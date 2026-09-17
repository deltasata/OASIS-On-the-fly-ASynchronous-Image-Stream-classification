# OASIS — on-the-fly classification of asynchronous image streams

Code for:

> **OASIS: On-the-fly ASynchronous Image Stream classification**

Seven transformer architectures that score each new observation of a multi-band,
irregularly sampled image time series as a lensed SN Ia or not. Inputs are the
image of the band that arrived, its observation time, and its band index.

## Data

The paper trains on 128 000 systems. `Transformer_demo4096/` ships a small subset
so the models run end to end.

| file | shape | contents |
|------|-------|----------|
| `<split>_imgs.npy` | `(N, 14, 1, 26, 26)` | one image per epoch, `float16` |
| `<split>_time.npy` | `(N, 14)` | normalised observation time per epoch |
| `<split>_band.npy` | `(N, 14)` | band index per epoch (0=*g*, 1=*r*, 2=*i*, 3=*z*) |
| `<split>_labels.npy` | `(N,)` | binary label |
| `label_and_details_<Split>.txt` | `(N, 2)` | class code, catalogue ID |
| `idxs_<Split>.npy` | `(N,)` | indices into the full dataset |

`<split>` is `train` / `val` / `test`.

| split | N | positive | negative |
|-------|---|----------|----------|
| train | 4096 | 2048 | 2048 |
| val | 1024 | 88 | 936 |
| test | 1024 | 84 | 940 |

Class codes `2` and `4` are positive, `0`, `-21`, `-22`, `-31` negative. The
class ratio of each split follows the full dataset: training balanced,
validation and test at the real ~8.5% positive rate. Images are the central
26×26 of the 59×59 stamps, to stay under GitHub's file limit.

## Models

`models/` holds one self-contained script per architecture. They share the same
dataset, training loop and hyperparameters, and differ only in the model.

| | attention over space and time | readout | width, depth |
|---|---|---|---|
| `A1-factorized_mean_pool.py` | factorized | mean pool | 256, 4 |
| `A2-factorized_query_readout.py` | factorized | learned query | 256, 4 |
| `A3-factorized_latent_array.py` | factorized | latent array | 256, 4 |
| `B1-joint_global.py` | joint, global | mean pool | 256, 4 |
| `B2w-joint_window_wide.py` | joint, 3×3 window | mean pool | 384, 4 |
| `B2d-joint_window_deep.py` | joint, 3×3 window | mean pool | 256, 6 |
| `B3-joint_pyramid.py` | joint, pyramid | time-conditioned | 256, 3+3 |

## Running it

```bash
python models/A2-factorized_query_readout.py
```

Everything you need to change is in the **CONFIG** block at the top. It ends
with a demo block:

```python
DEMO_MODE = True     # small demo set
```

Set `DEMO_MODE = False`, or delete the block, to train on the full dataset with
the published settings, and point `DATA_DIR` at your own data. Nothing outside
that block refers to the demo.

Training writes to `models/runs/<MODEL_ID>/`, keeps the best model by `val_loss`
and rolls back after `NHOLD` bad epochs.

**This is a smoke test, not a reproduction.**
