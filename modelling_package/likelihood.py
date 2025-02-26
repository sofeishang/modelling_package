# likelihood.py
import numpy as np

def likelihood_lr_prec(probability_choice_given_lr_prec):
    return np.exp(np.sum(np.log(probability_choice_given_lr_prec)))

def posterior (likelihood_matrix):
    """
    out.posterior=out.posterior./(sum(sum(out.posterior)));
    """
    denominator = likelihood_matrix.values.sum()
    likelihood_matrix=(likelihood_matrix/denominator)
    return likelihood_matrix
