# modelling_package

# README.md
# Modelling Package

## Overview
This package provides implementations of the Rescorla-Wagner model and related computational functions for learning rate and precision estimation. It is designed to analyze behavioral data using a model-based approach.

## Installation
To install the package, navigate to the directory and run:
```sh
pip install -e .
```

## Usage
```python
from modelling_package.rescorla_wagner import rescorla_wagner_model
from modelling_package.likelihood import likelihood_lr_prec
from modelling_package.utils import generate_lr, generate_prec, prob_choice_lr_prec
```

## Functions
- `rescorla_wagner_model(lr, outcome, initial_belief)`: Implements the Rescorla-Wagner learning model.
- `likelihood_lr_prec(probability_choice_given_lr_prec)`: Computes likelihood from choice probabilities.
- `generate_lr(lr_min, lr_max, total_number)`: Generates learning rates within a range.
- `generate_prec(prec_min, prec_max, total_number)`: Generates precision parameters within a range.
- `prob_choice_lr_prec(belief_list, actual_choice, p)`: Computes the probability of a choice given beliefs and precision.

