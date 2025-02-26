# utils.py
import numpy as np
from scipy.special import i0

def normalize_pi_neg_pi(value):
    return (value + np.pi) % (2 * np.pi) - np.pi

def normalize_0_2pi(value):
    return value % (2 * np.pi)

def von_mises_pdf(mu, kappa, choice_angle):
    return np.exp(kappa * np.cos(choice_angle - mu)) / (2 * np.pi * i0(kappa))

def inverse_logit(x):
    return 1 / (1 + np.exp(-x))

def logit(x):
    return np.log(x / (1 - x))

def generate_choice(belief_mu, precision_kappa):
    return [np.random.vonmises(b, precision_kappa, 1).tolist()[0] for b in belief_mu]

def generate_lr(lr_min, lr_max, total_number):
    lrpoints_sampled = np.linspace(logit(lr_min), logit(lr_max), total_number)
    return lrpoints_sampled, inverse_logit(lrpoints_sampled)

def generate_prec(prec_min, prec_max, total_number):
    precpoints_sampled = np.linspace(np.log(prec_min), np.log(prec_max), total_number)
    return precpoints_sampled, np.exp(precpoints_sampled)

def prob_choice_lr_prec (belief_list, actual_choice, p):
    i = 0
    to_ret = []
    while i < len(actual_choice):
        b = belief_list[i]
        c = actual_choice[i]
        prob = von_mises_pdf(b, p, c)
        to_ret.append(prob)
        i = i + 1
    return to_ret