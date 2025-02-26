# fit_model_lr_prec_bic_raw_lr.py
import numpy as np
import pandas as pd
import itertools
from modelling_package.rescorla_wagner import rescorla_wagner_model
from modelling_package.likelihood import likelihood_lr_prec, posterior
from modelling_package.utils import generate_lr, generate_prec, prob_choice_lr_prec, inverse_logit, logit
lr_range = [0.001, 0.999]
lr_num = 99
prec_range = [0.01, 100]
prec_num = 99

data = pd.read_csv("simulated_data.csv")

logit_lr, lr_parameter = generate_lr (lr_range[1], lr_range[0], lr_num)
logged_precision, precision_parameter = generate_prec(prec_range[0], prec_range[1], prec_num)


study_block = ["HighVolatility_HighNoise", "HighVolatility_LowNoise", 
               "LowVolatility_HighNoise", "LowVolatility_LowNoise"]

id_list = data.id.unique().tolist()

fitted_lr_list = []
fitted_prec_list = []
track_id = []
track_block = []
model_liklihood = []

permutation = [x for x in itertools.product(lr_parameter, precision_parameter)]

for id in id_list:
    print (id)
    for b in study_block:
        df = data[(data.id == id)]
        outcome = df["normalized_coin_angle"].tolist()
        actual_choice = df["normalized_position_predicted"].tolist()
        all_likelihood = []
        for parameters in permutation:
            lr = parameters[0]
            precision = parameters[1]
            belief = rescorla_wagner_model(lr, outcome, outcome[0])
            choice_prob = prob_choice_lr_prec(belief, actual_choice, precision)
            likelihood = likelihood_lr_prec(choice_prob)
            all_likelihood.append(likelihood)
        all_likelihood_df= pd.DataFrame({'parameters': permutation, 'likelihood': all_likelihood})
        all_likelihood_df[['lr','precision']] = pd.DataFrame(all_likelihood_df.parameters.tolist(), index= all_likelihood_df.index)
        
        # each column represents a unique precision value
        # each row represents a unique lr value
        all_likelihood_matrix = all_likelihood_df.pivot(index='lr', columns='precision', values='likelihood') # the same
        posterior_distribution = posterior(all_likelihood_matrix) # the same
        marginal_lr = posterior_distribution.sum(axis=1).tolist() # sum by row
        marginal_precision =posterior_distribution.sum(axis=0).tolist() # sum by col
        lr = np.dot(marginal_lr, np.flip(logit_lr))
        prec = np.dot(marginal_precision, logged_precision)
        
        model_fitted_lr = inverse_logit(lr)
        model_fitted_prec = np.exp(prec)
        fitted_lr_list.append(inverse_logit(lr))
        fitted_prec_list.append(np.exp(prec))
        track_id.append(id)
        track_block.append(b)
        
        # model likelihood
        outcome = df["normalized_coin_angle"].tolist()
        actual_choice = df["normalized_position_predicted"].tolist()
        predicted_belief = rescorla_wagner_model(model_fitted_lr, outcome, outcome[0])
        predicted_choice_prob = prob_choice_lr_prec(predicted_belief, actual_choice, model_fitted_prec)
        predicted_likelihood = likelihood_lr_prec(predicted_choice_prob)
        
        model_liklihood.append(predicted_likelihood)


fitted = pd.DataFrame({"fitted_lr":fitted_lr_list, 
"fitted_prec": fitted_prec_list,
"model_liklihood": model_liklihood,
"id": track_id,
"block": track_block})

fitted_lr = fitted[["id", "block", "fitted_lr"]].pivot(index = 'id', columns="block", values = ["fitted_lr"])
fitted_lr = fitted_lr["fitted_lr"].reset_index()
fitted_lr.rename(columns = {"HighVolatility_HighNoise": "HighVolatility_HighNoise_lr", 
                            "HighVolatility_LowNoise": "HighVolatility_LowNoise_lr",
                            "LowVolatility_HighNoise": "LowVolatility_HighNoise_lr",
                            "LowVolatility_LowNoise": "LowVolatility_LowNoise_lr"}, inplace = True)
fitted_lr['HighVolatility_HighNoise_lr_logit'] =  fitted_lr['HighVolatility_HighNoise_lr'].apply(logit)
fitted_lr['HighVolatility_LowNoise_lr_logit'] =  fitted_lr['HighVolatility_LowNoise_lr'].apply(logit)
fitted_lr['LowVolatility_HighNoise_lr_logit'] =  fitted_lr['LowVolatility_HighNoise_lr'].apply(logit)
fitted_lr['LowVolatility_LowNoise_lr_logit'] =  fitted_lr['LowVolatility_LowNoise_lr'].apply(logit)

fitted_prec = fitted[["id", "block", "fitted_prec"]].pivot(index = 'id', columns="block", values = ["fitted_prec"])
fitted_prec = fitted_prec["fitted_prec"].reset_index()
fitted_prec.rename(columns = {"HighVolatility_HighNoise": "HighVolatility_HighNoise_prec", 
                            "HighVolatility_LowNoise": "HighVolatility_LowNoise_prec",
                            "LowVolatility_HighNoise": "LowVolatility_HighNoise_prec",
                            "LowVolatility_LowNoise": "LowVolatility_LowNoise_prec"}, inplace = True)

fitted_prec['HighVolatility_HighNoise_prec_log'] =  fitted_prec['HighVolatility_HighNoise_prec'].transform("log")
fitted_prec['HighVolatility_LowNoise_prec_log'] =  fitted_prec['HighVolatility_LowNoise_prec'].transform("log")
fitted_prec['LowVolatility_HighNoise_prec_log'] =  fitted_prec['LowVolatility_HighNoise_prec'].transform("log")
fitted_prec['LowVolatility_LowNoise_prec_log'] =  fitted_prec['LowVolatility_LowNoise_prec'].transform("log")

modelfit = pd.merge(fitted_lr, fitted_prec, on = "id")
number_sub = modelfit.id.nunique()

likelihood = fitted[["id", "block", "model_liklihood"]].pivot(index = 'id', columns="block", values = ["model_liklihood"])
likelihood = likelihood["model_liklihood"].reset_index()
likelihood.rename(columns = {"high": "high_likelihood", 
                            "low": "low_likelihood"}, inplace = True)

modelfit = pd.merge(fitted_lr, fitted_prec, on = "id")
modelfit = pd.merge(modelfit, likelihood, on = "id")

number_sub = modelfit.id.nunique()

modelfit.to_csv("simulated_data_model_fit.csv" )