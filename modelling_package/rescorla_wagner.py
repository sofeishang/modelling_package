import numpy as np
import math

def normalize_pi_neg_pi(value):
    return (value + np.pi) % (2 * np.pi) - np.pi

def normalize_0_2pi(value):
    return value % (2 * np.pi)

def mindist(angle_1, angle_2):
    if angle_2 > angle_1:
        return (angle_2 - angle_1) if (angle_2 - angle_1) < math.pi else (angle_2 - angle_1) - 2 * math.pi
    else:
        return (angle_2 - angle_1) if (angle_1 - angle_2) < math.pi else (angle_2 - angle_1) + 2 * math.pi


def rescorla_wagner_model(lr, outcome, initial_belief):
    next_belief_list = [initial_belief]
    for i in range(1, len(outcome)):
        next_belief = next_belief_list[i-1] + lr * (mindist(next_belief_list[i-1], outcome[i-1]))
        next_belief = normalize_pi_neg_pi(next_belief)
        next_belief = normalize_0_2pi(next_belief)
        next_belief_list.append(next_belief)
    return next_belief_list

def rescorla_wagner_model_change_angle(lr, outcome, initial_belief):
    next_belief_list = [initial_belief]
    for i in range(1, len(outcome)):
        next_belief = next_belief_list[i-1] + lr * (mindist(next_belief_list[i-1], outcome[i-1]))
        next_belief = normalize_pi_neg_pi(next_belief)
        next_belief_list.append(next_belief)
    return next_belief_list

def learnt_info(coin_angle, change_position, sensitivity):
    learnt = np.array(coin_angle) + sensitivity * np.array(change_position)
    learnt = [normalize_pi_neg_pi(l) for l in learnt]
    learnt = [normalize_0_2pi(l) for l in learnt]
    return learnt
