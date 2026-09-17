"""
B2d -- Joint windowed space-time attention (deep)

As B2w, a deeper and narrower shape: d_model = 256 over 6 layers.
Mean-pool readout.

One of seven transformer architectures released with the paper. All seven share
an identical training pipeline; they differ only in the model definition
(section 3) and in the CONFIG block below.

A demo block at the end of CONFIG points the script at the small dataset
bundled with this repository. Set DEMO_MODE = False there, or delete the block
altogether, to train on the full dataset as in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import sys
from tqdm import tqdm

# ==========================================================================
# CONFIG
# ==========================================================================
MODEL_ID = "B2d"

# --- Input data -------------------------------------------------------------
# Edit the path below to point at your own data for a real run. See load_split()
# and the bundled demo data for the required input structure, or adapt the
# loading block to your data.
DATA_DIR = "path/to/the/input/data"

# --- Architecture -----------------------------------------------------------
D_MODEL   = 256
N_LAYERS  = 6
N_HEADS   = 4
DIM_FF    = 256
MASK_MODE = 'window'
WINDOW_R  = 1
RELPOS    = False
READOUT   = 'mean'

# --- Data / optimisation ----------------------------------------------------
BATCH_SIZE = 128
PATCH_SIZE = 7
NCH        = 4
NEPOCH     = 300
NHOLD      = 32
ILR        = 3e-4
DS         = 630 * 4
DR         = 0.98
SEED       = 0          # 0 = do not set a manual seed

# --- Learning-rate schedule -------------------------------------------------
WARMUP_STEPS = 630      # length of the warmup ramp, in optimiser steps
T0_EPOCHS    = 30       # cosine restart period, in units of WARMUP_STEPS

# ==========================================================================
# DEMO BLOCK
# Runs the script on the small dataset bundled with this repository. Set
# DEMO_MODE = False, or delete this whole block, to reproduce the paper runs on
# the full dataset. Nothing outside this block refers to the demo.
DEMO_MODE = True

if DEMO_MODE:
    DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "Transformer_demo4096")
    WARMUP_STEPS = 32   # one epoch over the 4096 demo training objects
# END DEMO BLOCK
# ==========================================================================


# --------------------------------------------------------------------------
# 1. Dataset
# --------------------------------------------------------------------------
class PaddedImageDataset(Dataset):
    def __init__(self, images, time, channel_id, labels, patch_size):
        H, W  = images.shape[-2], images.shape[-1]
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h), value=0.0)
        print(f"Image size after padding: {images.shape[-2]}x{images.shape[-1]} "
              f"(was {H}x{W}, padded by {pad_h}x{pad_w})")
        self.images     = images
        self.time       = time
        self.channel_id = channel_id
        self.labels     = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx], self.time[idx],
                self.channel_id[idx], self.labels[idx])


# --------------------------------------------------------------------------
# 2. Time embedding (Time2Vec)
# --------------------------------------------------------------------------
class Time2Vec(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(1, d_model)

    def forward(self, t):
        x        = self.W(t)
        trend    = x[:, :, :1]
        periodic = torch.sin(x[:, :, 1:])
        return torch.cat([trend, periodic], dim=-1)


# --------------------------------------------------------------------------
# 3. Model -- Joint windowed space-time attention (deep)
# --------------------------------------------------------------------------
# One attention over all T*P space-time tokens, with the connectivity
# restricted by MASK_MODE:
#   causal    (t', p') visible when t' <= t
#   window    as causal, and additionally the Chebyshev distance between patches
#             p and p' must be at most WINDOW_R -- full temporal reach, local
#             spatial reach
#   bandsync  the whole current observation, plus the same patch position in
#             every earlier observation
# RELPOS optionally adds a learned relative space-time position bias.
# Readout: mean over the current observation's patch tokens.

# Pre-norm encoder layer built directly on nn.MultiheadAttention.
# nn.TransformerEncoder's eval-only fast path can return NaN when given a
# finite additive float mask, so with RELPOS the attention is run manually to
# keep training and evaluation on the same code path.
class ManualEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.dp = nn.Dropout(dropout)

    def forward(self, x, mask):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), attn_mask=mask, need_weights=False)
        x = x + self.dp(a)
        x = x + self.dp(self.ff(self.n2(x)))
        return x


class ManualEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([ManualEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
                                     for _ in range(n_layers)])

    def forward(self, x, mask=None, is_causal=False):
        for l in self.layers:
            x = l(x, mask)
        return x


class JointMaskTransformer(nn.Module):
    def __init__(self, d_model, patch_size, H_pad, W_pad, Nch,
                 n_heads=4, n_layers=4, dim_feedforward=256,
                 mask_mode="window", window_r=1, relpos=False, readout="mean"):
        super().__init__()
        self.d_model = d_model; self.patch_size = patch_size
        self.Ph = H_pad // patch_size; self.Pw = W_pad // patch_size
        self.P = self.Ph * self.Pw
        self.mask_mode = mask_mode; self.window_r = window_r
        self.relpos = relpos; self.readout = readout
        # patch index -> (row, col) lookup
        pid = torch.arange(self.P)
        self.register_buffer("p_row", pid // self.Pw)
        self.register_buffer("p_col", pid %  self.Pw)
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU())
        self.patch_embed = nn.Conv2d(64, d_model, kernel_size=patch_size, stride=patch_size)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.P, d_model) * 0.02)
        self.time_embed = Time2Vec(d_model)
        self.band_scale = nn.Embedding(Nch, d_model); self.band_shift = nn.Embedding(Nch, d_model)
        nn.init.ones_(self.band_scale.weight); nn.init.zeros_(self.band_shift.weight)
        if relpos:
            # a learned additive bias needs ManualEncoder (see above)
            self.transformer = ManualEncoder(d_model, n_heads, dim_feedforward, n_layers, 0.1)
        else:
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                batch_first=True, dropout=0.1)
            self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        if relpos:
            self.MAXDT = 14
            self.rel_bias = nn.Parameter(
                torch.zeros(self.MAXDT, 2 * self.Ph - 1, 2 * self.Pw - 1))
        if readout == "xattn":
            self.read_q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.read_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
            self.read_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1))

    def _build_mask(self, T, device):
        P, Ph, Pw = self.P, self.Ph, self.Pw
        idx = torch.arange(T * P, device=device)
        ti = (idx // P); pi = idx % P
        ri = self.p_row.to(device)[pi]; ci = self.p_col.to(device)[pi]
        # pairwise (query i, key j)
        dt = ti.unsqueeze(1) - ti.unsqueeze(0)          # >=0 means key is past/current
        dr = ri.unsqueeze(1) - ri.unsqueeze(0)
        dc = ci.unsqueeze(1) - ci.unsqueeze(0)
        future = dt < 0
        if self.mask_mode == "causal":
            allow = ~future
        elif self.mask_mode == "window":
            spatial_ok = (dr.abs() <= self.window_r) & (dc.abs() <= self.window_r)
            allow = (~future) & spatial_ok
        elif self.mask_mode == "bandsync":
            same_t = dt == 0
            same_p = (dr == 0) & (dc == 0)
            allow = same_t | ((~future) & same_p)
        else:
            raise ValueError(self.mask_mode)
        am = torch.zeros(T * P, T * P, device=device)
        if self.relpos:
            dt_c = dt.clamp(0, self.MAXDT - 1)
            bias = self.rel_bias[dt_c, dr + (Ph - 1), dc + (Pw - 1)]
            am = am + bias
        am = am.masked_fill(~allow, -1e9)
        return am

    def _read_mask(self, T, P, device):
        qi = torch.arange(T, device=device).unsqueeze(1)
        kt = (torch.arange(T * P, device=device) // P).unsqueeze(0)
        allow = kt <= qi
        am = torch.zeros(T, T * P, device=device); am[~allow] = float('-inf')
        return am

    def forward(self, imgs, timestamps, band_ids):
        B, T, C, H, W = imgs.shape; P, d = self.P, self.d_model
        x = self.cnn_stem(imgs.reshape(B * T, 1, H, W))
        x = self.patch_embed(x).flatten(2).transpose(1, 2).reshape(B, T, P, d)
        x = x + self.spatial_pos
        x = x * self.band_scale(band_ids).unsqueeze(2) + self.band_shift(band_ids).unsqueeze(2)
        t_emb = self.time_embed(timestamps.unsqueeze(-1))
        x = x + t_emb.unsqueeze(2)
        x = x.reshape(B, T * P, d)
        attn_mask = self._build_mask(T, imgs.device)
        h = self.transformer(x, mask=attn_mask, is_causal=False)
        if self.readout == "xattn":
            q = self.read_q + t_emb
            rmask = self._read_mask(T, P, imgs.device)
            r, _ = self.read_attn(q, h, h, attn_mask=rmask, need_weights=False)
            h = self.read_norm(r)
        else:
            h = h.reshape(B, T, P, d).mean(dim=2)
        logits = self.classifier(h).squeeze(-1)
        return logits, torch.sigmoid(logits)


# --------------------------------------------------------------------------
# 4. Training loop
# --------------------------------------------------------------------------
def run_epoch(model, loader, device, Nt, optimizer=None, scheduler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0
    total_acc  = 0
    n = 0

    pbar = tqdm(loader, disable=not sys.stdout.isatty())

    with torch.set_grad_enabled(optimizer is not None):
        for imgs, t, b, label in pbar:
            imgs  = imgs.to(device)
            t     = t.to(device)
            b     = b.to(device)
            label = label.to(device)

            logits, probs = model(imgs, t, b)

            label_exp = label.unsqueeze(1).expand(-1, Nt).float()
            loss = F.binary_cross_entropy_with_logits(logits, label_exp)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            pred = (probs > 0.5).long()
            acc  = (pred == label.unsqueeze(1)).float().mean()

            total_loss += loss.item() * imgs.shape[0]
            total_acc  += acc.item()  * imgs.shape[0]
            n += imgs.shape[0]

            pbar.set_postfix(loss=total_loss / n, acc=total_acc / n)

    return total_loss / n, total_acc / n


def save_ckpt(model, optimizer, scheduler, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)


def load_ckpt(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


def adjust_learning_rate(optimizer, scheduler, ilr, ds, dr):
    step = scheduler.last_epoch
    old_lr0 = optimizer.param_groups[0]["initial_lr"]
    current_lr = optimizer.param_groups[0]["lr"]
    factor = 10 ** np.random.uniform(-3, 3)
    new_lr0 = float(np.clip(old_lr0 * factor, 1e-8, 1e-3))
    print("=" * 60)
    print("Adjusting LR schedule")
    print(f"step={step}")
    print(f"Old lr0 = {old_lr0:.4e}")
    print(f"Current lr = {current_lr:.4e}")
    print(f"New lr0 = {new_lr0:.4e}  factor={factor:.4e}")
    print("=" * 60)
    for g in optimizer.param_groups:
        g["initial_lr"] = new_lr0
        g["lr"] = new_lr0 * (dr ** (step / ds))
    new_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: dr ** (s / ds))
    new_scheduler.last_epoch = step
    return new_scheduler


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import shutil

    if SEED:
        torch.manual_seed(SEED); np.random.seed(SEED)

    print("CUDA:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    def load_split(data_dir, split):
        imgs  = torch.from_numpy(np.load(os.path.join(data_dir, f"{split}_imgs.npy"),  mmap_mode="r")).float()
        time  = torch.from_numpy(np.load(os.path.join(data_dir, f"{split}_time.npy"),  mmap_mode="r")).float()
        band  = torch.from_numpy(np.load(os.path.join(data_dir, f"{split}_band.npy"),  mmap_mode="r")).long()
        label = torch.from_numpy(np.load(os.path.join(data_dir, f"{split}_labels.npy"), mmap_mode="r")).long()
        return imgs, time, band, label

    data_dir = DATA_DIR
    print(f"[{MODEL_ID}] data_dir={data_dir}")

    train_imgs, train_time, train_band, train_labels = load_split(data_dir, "train")
    val_imgs,   val_time,   val_band,   val_labels   = load_split(data_dir, "val")
    test_imgs,  test_time,  test_band,  test_labels  = load_split(data_dir, "test")
    print("Test imgs shape:", test_imgs.shape)

    batch_size = BATCH_SIZE
    patch_size = PATCH_SIZE
    Nch        = NCH

    def make_loader(dataset, batch_size, shuffle=False, workers=None):
        if workers is None:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)

    Num_workers = None

    train_dataset = PaddedImageDataset(train_imgs, train_time, train_band, train_labels, patch_size)
    train_loader  = make_loader(train_dataset, batch_size, shuffle=False, workers=Num_workers)

    Nsam, Nt, _, H, W = train_imgs.shape
    H_pad = train_dataset.images.shape[-2]
    W_pad = train_dataset.images.shape[-1]

    val_dataset  = PaddedImageDataset(val_imgs, val_time, val_band, val_labels, patch_size)
    val_loader   = make_loader(val_dataset, batch_size, shuffle=False, workers=Num_workers)

    test_dataset = PaddedImageDataset(test_imgs, test_time, test_band, test_labels, patch_size)
    test_loader  = make_loader(test_dataset, batch_size, shuffle=False, workers=Num_workers)

    d_model = D_MODEL

    model = JointMaskTransformer(
        d_model         = d_model,
        patch_size      = patch_size,
        H_pad           = H_pad,
        W_pad           = W_pad,
        Nch             = Nch,
        n_heads         = N_HEADS,
        n_layers        = N_LAYERS,
        dim_feedforward = DIM_FF,
        mask_mode       = MASK_MODE,
        window_r        = WINDOW_R,
        relpos          = RELPOS,
        readout         = READOUT,
    ).to(device)

    ilr = ILR
    ds  = DS
    dr  = DR

    # Warmup ramp, then cosine warm-restarts with period T0 and peak ilr.
    warmup = WARMUP_STEPS
    T0     = WARMUP_STEPS * T0_EPOCHS
    print(f"LR schedule: warmup={warmup} steps, T0={T0} steps")

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        s = (step - warmup) % T0
        return 0.5 * (1.0 + np.cos(np.pi * s / T0))

    optimizer = torch.optim.Adam(model.parameters(), lr=ilr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Each architecture writes to its own run directory, so the seven scripts
    # can be run from the same working directory without overwriting each other.
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "runs", MODEL_ID) + os.sep
    CHECKPOINT_DIR = main_dir + "checkpoints/"
    TEMP_DIR = main_dir + "temp_models/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    metrics_file = main_dir + "metrics_file.txt"
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w") as f:
            f.write("#epoch train_loss train_acc val_loss val_acc\n")

    def save_metrics(epochs, stats):
        with open(metrics_file, "a") as f:
            for e, s in zip(epochs, stats):
                f.write(f"{e}\t{s[0]:.4f}\t{s[1]:.4f}\t{s[2]:.4f}\t{s[3]:.4f}\n")
        for e in epochs:
            shutil.move(f"{TEMP_DIR}/checkpoint_epoch_{e}.pt",
                        f"{CHECKPOINT_DIR}/checkpoint_epoch_{e}.pt")
            shutil.move(f"{TEMP_DIR}/model_epoch_{e}.pt",
                        f"{CHECKPOINT_DIR}/model_epoch_{e}.pt")

    Nepoch = NEPOCH
    Nhold = NHOLD

    best_val_loss = np.inf
    initial_epoch = 0
    failure_count = 0
    all_failure_count = 0
    temp_epochs = []
    temp_stats = []
    epoch = initial_epoch + 1

    while epoch < Nepoch:
        train_loss, train_acc = run_epoch(model, train_loader, device, Nt, optimizer, scheduler)
        val_loss, val_acc = run_epoch(model, val_loader, device, Nt)

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        temp_epochs.append(epoch)
        temp_stats.append([train_loss, train_acc, val_loss, val_acc])
        save_ckpt(model, optimizer, scheduler, epoch, f"{TEMP_DIR}/checkpoint_epoch_{epoch}.pt")
        torch.save(model, f"{TEMP_DIR}/model_epoch_{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_metrics(temp_epochs, temp_stats)
            failure_count = 0
            all_failure_count = 0
            save_ckpt(model, optimizer, scheduler, epoch, f"{CHECKPOINT_DIR}/ckpt_best.pt")
            torch.save(model, f"{CHECKPOINT_DIR}/model_best.pt")
            temp_epochs = []
            temp_stats = []
            initial_epoch = epoch
        else:
            failure_count += 1
            all_failure_count += 1
            print(f"No improvement {failure_count}/{Nhold} (best epoch={initial_epoch} val_loss={best_val_loss:.4f})")
            if failure_count >= Nhold:
                print(f"\n Rolling back to best model epoch={initial_epoch} \n")
                failure_count = 0
                load_ckpt(f"{CHECKPOINT_DIR}/ckpt_best.pt", model, optimizer, scheduler, device)
                shutil.rmtree(TEMP_DIR)
                os.makedirs(TEMP_DIR)
                if all_failure_count >= 3 * Nhold:
                    scheduler = adjust_learning_rate(optimizer, scheduler, ilr, ds, dr)
                    all_failure_count = 0
                epoch = initial_epoch + 1
                temp_epochs = []
                temp_stats = []
                continue
        epoch += 1
