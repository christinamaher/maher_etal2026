# Coordinated Human Prefrontal Dynamics During Learning

Code accompanying:

**Maher et al.**
[*Coordinated human prefrontal dynamics sustain task-state representations during learning*](https://www.biorxiv.org/content/10.64898/2026.05.03.722562v1.full.pdf)

This repository contains lightweight implementations of the core behavioral and neural analysis workflows described in the manuscript.

## Files

### `Neural_Analysis.py`

Core neural analysis functions used throughout the manuscript, including:

* permutation-based regression
* time-resolved regression
* cluster-based permutation testing
* PSI connectivity analyses
* mixed-effects modeling
* single-unit encoding analyses

### `Behavioral_Modeling.R`

Core reinforcement learning modeling functions, including:

* Uniform Attention RL (UA-RL)
* Selective Attention / ACL RL (SA-RL)
* model fitting
* held-out likelihood estimation
* leave-one-block-out cross-validation

## Notes

These scripts are intended to demonstrate the primary analysis logic and statistical workflows used in the manuscript. Patient data is not included in this repository.

LFP preprocessing was performed using the [LFPAnalysis](https://github.com/seqasim/LFPAnalysis) package.

