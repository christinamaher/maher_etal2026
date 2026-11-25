# Coordinated Human Prefrontal Dynamics During Learning

Code accompanying:

**Maher et al.**  
[*Coordinated human prefrontal dynamics sustain task-state representations during learning*](https://www.biorxiv.org/content/10.64898/2026.05.03.722562v1.full.pdf)

This repository contains lightweight implementations of the core behavioral and neural analysis workflows described in the manuscript. Simulated datasets and demonstration scripts are provided to illustrate the functionality of the analysis code.

## Repository Contents

### `Neural_Analysis.py`

Core neural analysis functions used throughout the manuscript, including:

* Permutation-based regression
* Time-resolved regression
* Cluster-based permutation testing
* PSI connectivity analyses
* Mixed-effects modeling
* Single-unit encoding analyses

### `Behavioral_Modeling.R`

Core reinforcement learning modeling functions, including:

* Uniform Attention RL (UA-RL)
* Selective Attention RL (SA-RL; also referred to as ACL)
* Model fitting
* Held-out likelihood estimation
* Leave-one-block-out cross-validation

### `demo/`

Simulated datasets and demonstration scripts:

* `simulated_behavioral_data.csv`: simulated trial-level behavioral data from a multidimensional reinforcement learning task.
* `simulated_neural_data.csv`: simulated neural power data and associated behavioral regressors.
* `behavioral_demo.R`: demonstration of reinforcement learning model fitting.
* `neural_demo.py`: demonstration of permutation-based neural encoding analyses.

## System Requirements

### Software Dependencies

The behavioral modeling code is implemented in R, and the neural analysis code is implemented in Python.

The Python code requires the following packages:

* NumPy
* pandas
* SciPy
* patsy
* statsmodels
* tqdm
* MNE-Python

The behavioral modeling code uses base R and the standard `stats` package. No additional R packages are required.

LFP preprocessing was performed using the [LFPAnalysis](https://github.com/seqasim/LFPAnalysis) package. This package is not required to run the demonstration analyses, which use simulated, preprocessed neural data.

### Tested Software Versions

The demonstration scripts have been tested using the following software versions:

| Software | Version |
|---|---|
| Operating system | [Insert version] |
| Python | [Insert version] |
| R | [Insert version] |
| NumPy | [Insert version] |
| pandas | [Insert version] |
| SciPy | [Insert version] |
| patsy | [Insert version] |
| statsmodels | [Insert version] |
| tqdm | [Insert version] |
| MNE-Python | [Insert version] |

### Hardware Requirements

The demonstration analyses are designed to run on a standard desktop or laptop computer. No specialized hardware or GPU acceleration is required.

## Installation Guide

Download or clone this repository and navigate to the repository directory.

### Python

Install the required Python dependencies using:

```bash
python -m pip install numpy pandas scipy patsy statsmodels tqdm mne
```

### R

Install R from the [Comprehensive R Archive Network (CRAN)](https://cran.r-project.org/).

No additional R packages are required to run the behavioral modeling code.

### Typical Installation Time

Installation on a standard desktop computer takes approximately [insert installation time], depending on the existing software environment and internet connection.

## Demonstration

Small simulated behavioral and neural datasets are provided in the `demo/` directory to demonstrate the core analysis workflows.

All demonstration data are entirely synthetic and do not contain data from human participants.

### Behavioral Modeling

The simulated behavioral dataset contains 648 trials from six simulated participants performing a multidimensional reinforcement learning task. Each participant completes six blocks of 18 trials.

Choices are generated using attention-weighted feature values and a softmax choice rule.

To run the behavioral modeling demonstration, execute the following command from the repository root:

```bash
Rscript demo/behavioral_demo.R
```

**Expected output:** Fitted model parameters, negative log likelihoods, and Bayesian information criterion (BIC) values for the UA-RL and SA-RL models.

**Expected runtime:** [Insert runtime] on a standard desktop computer.

### Neural Analysis

The simulated neural dataset contains 3,888 observations from 12 simulated electrodes, together with associated behavioral regressors and time-point identifiers.

A negative relationship between expected value and neural power was imposed during simulation, with additional random noise, to demonstrate the neural encoding analysis.

To run the neural analysis demonstration, execute the following command from the repository root:

```bash
python demo/neural_demo.py
```

**Expected output:** Electrode-level regression coefficients, permutation-derived z-scores, and associated p-values. The demonstration is designed to illustrate the recovery of the imposed relationship between expected value and neural power.

**Expected runtime:** [Insert runtime] on a standard desktop computer.

## Instructions for Use

### Behavioral Modeling

The functions in `Behavioral_Modeling.R` accept trial-level behavioral data containing:

* Block identifiers
* Chosen option indices
* Reward outcomes
* Relevant feature dimensions
* Feature identities for each available option

Trials should be ordered chronologically within each block. For datasets containing multiple participants, model fitting should be performed separately for each participant.

Users can specify the relevant column names, model parameters, feature dimensions, and fitting settings through the function arguments.

The script includes documentation describing the required inputs, optional arguments, and returned outputs.

### Neural Analysis

The functions in `Neural_Analysis.py` accept observation-level neural data and user-specified regression formulas.

Depending on the analysis, input data should include:

* Neural activity measurements
* Behavioral or computational regressors
* Electrode or single-unit identifiers
* Participant identifiers
* Time-point identifiers for time-resolved analyses

Users can specify regression formulas, grouping variables, regressors to permute, and the number of permutations through the function arguments.

The script includes documentation describing the required inputs, optional arguments, and returned outputs.

## Data Availability and Reproducibility

These scripts are intended to demonstrate the primary analysis logic and statistical workflows used in the manuscript.

Patient data are not included in this repository. The simulated datasets are provided solely to demonstrate the functionality of the analysis code and are not intended to reproduce the quantitative findings reported in the manuscript.

Detailed descriptions of the behavioral models, neural analyses, and statistical procedures are provided in the Methods section of the accompanying manuscript.

## License


## Citation

If you use or adapt this code, please cite:

Maher et al. *Coordinated human prefrontal dynamics sustain task-state representations during learning.*

https://doi.org/10.64898/2026.05.03.722562

