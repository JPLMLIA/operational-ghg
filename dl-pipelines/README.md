# Deep Learning Pipelines

This directory contains scripts for training and applying segmentation models
for plume detection.

The models are described in Section 4.4 and Supplemental Information Section 3 
of our preprint: https://arxiv.org/abs/2505.21806

## Pretrained models

Model weights are available at: https://doi.org/10.5281/zenodo.19014658
Training datasets are avialable at: https://doi.org/10.5281/zenodo.19011045

### AVIRIS-NG multicampign

Methane plume detection model trained on the CACH4, Permian, and COVID campaigns

Usage:
```
python predict_flightline_ucnet.py \
    --norm-max 4000
    --band 4
    --weights weights/ANG_multitask.pt
    --pool average \
```

### EMIT

Preliminary methane plume detection model trained on EMIT

Usage:
```
python predict_flightline_ucnet.py \
    --norm-max 4000
    --band 4
    --weights weights/EMIT_multitask.pt
    --pool max \
```
