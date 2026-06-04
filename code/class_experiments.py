import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt  
from scipy.stats import beta, t
from sklearn.model_selection import KFold
from typing import Literal, Callable
from itertools import chain, combinations
import cvxpy as cp
from sklearn.metrics.pairwise import pairwise_kernels
import math
import warnings
from shap_mml import ShapMML
import random
import math
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
import scipy.special as sc
random.seed(123)

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

def svr_learning_fn(X, Y):
    if not np.any(X): return np.mean(Y)
    
    model = TransformedTargetRegressor(
        regressor=SVR(kernel='rbf', C=10.0, epsilon=0.1),
        transformer=StandardScaler()
    )
    model.fit(X, Y)
    return model

def svr_predict_fn(X, model):
    if not hasattr(model, 'predict'): return np.full(X.shape[0], model)
    return model.predict(X)

def get_beta_weights(p, a, b):
    weights = np.array([beta.pdf((r + 0.5) / p, a=a, b=b) for r in range(p)])
    norm = np.sum([weights[r] * math.comb(p-1, r) for r in range(p)])
    return weights / norm if norm > 0 else None


def get_delta_weights(p, target="d"):
    weights = np.zeros(p)
    if target == "d":
        weights[-1] = 1.0
    elif target == "1":
        weights[0] = 1.0  
    return weights

def generate_dynamic_synergistic_regression(n=600, n_modalities=7, attrs_per_modality=3, seed=42):
    rng = np.random.default_rng(seed)
    d = n_modalities * attrs_per_modality
    X = rng.normal(0, 1, size=(n, d))
    condition_var = X[:, 0]
    
    signal_A = rng.normal(0, 5, size=n)
    signal_B = rng.normal(0, 5, size=n)
    
    # Pathway A: Needs all 3 to cancel noise
    noise_A1 = rng.normal(0, 5, size=n)
    noise_A2 = rng.normal(0, 5, size=n)
    X[:, 1*attrs_per_modality] = signal_A + noise_A1
    X[:, 2*attrs_per_modality] = signal_A + noise_A2
    X[:, 3*attrs_per_modality] = signal_A - (noise_A1 + noise_A2)
    
    # Pathway B: Needs all 3 to cancel noise
    noise_B1 = rng.normal(0, 5, size=n)
    noise_B2 = rng.normal(0, 5, size=n)
    X[:, 4*attrs_per_modality] = signal_B + noise_B1
    X[:, 5*attrs_per_modality] = signal_B + noise_B2
    X[:, 6*attrs_per_modality] = signal_B - (noise_B1 + noise_B2)
    
    Y = np.where(condition_var > 0, 10 * signal_A, 10 * signal_B)
    Y += rng.normal(0, 1, size=n)
    
    modalities = {m: list(range(m * attrs_per_modality, (m + 1) * attrs_per_modality)) for m in range(n_modalities)}
    return X, Y, modalities


def generate_dynamic_redundant_regression(n=600, n_modalities=7, attrs_per_modality=3, seed=42):
    rng = np.random.default_rng(seed)
    d = n_modalities * attrs_per_modality
    X = rng.normal(0, 1, size=(n, d))
    condition_var = X[:, 0]
    
    signal_A = rng.normal(0, 5, size=n)
    signal_B = rng.normal(0, 5, size=n)
    
    # Pathway A: Redundant copies (Any 1 of these 3 gives the full signal)
    X[:, 1*attrs_per_modality] = signal_A + rng.normal(0, 0.1, size=n)
    X[:, 2*attrs_per_modality] = signal_A + rng.normal(0, 0.1, size=n)
    X[:, 3*attrs_per_modality] = signal_A + rng.normal(0, 0.1, size=n)
    
    # Pathway B: Redundant copies (Any 1 of these 3 gives the full signal)
    X[:, 4*attrs_per_modality] = signal_B + rng.normal(0, 0.1, size=n)
    X[:, 5*attrs_per_modality] = signal_B + rng.normal(0, 0.1, size=n)
    X[:, 6*attrs_per_modality] = signal_B + rng.normal(0, 0.1, size=n)
    
    # Target Y: Routed by the condition variable
    Y = np.where(condition_var > 0, 10 * signal_A, 10 * signal_B)
    Y += rng.normal(0, 1, size=n)
    
    modalities = {m: list(range(m * attrs_per_modality, (m + 1) * attrs_per_modality)) for m in range(n_modalities)}
    return X, Y, modalities

NUM_TRIALS = 10
N_SAMPLES = 600
N_MODALITIES = 7
Q_LIST = np.arange(0, N_MODALITIES + 1)
ALPHA_LEVEL = 0.05

methods_to_compare = {
    "Marginal Std": {"mode": "marg_std", "params": None, "fmt": {"color": "lightblue", "marker": "s", "ls": "--"}},
    "Cond Std": {"mode": "cond", "params": None, "fmt": {"color": "blue", "marker": "o", "ls": "-"}},
    "Cond Skew Small (1,16)": {"mode": "cond", "params": (1, 16), "fmt": {"color": "purple", "marker": "D", "ls": "-."}},
    "Cond Skew Large (16,1)": {"mode": "cond", "params": (16, 1), "fmt": {"color": "red", "marker": "^", "ls": ":"}},
    "Cond Auto": {"mode": "cond_auto", "params": None, "fmt": {"color": "green", "marker": "*", "ls": "-"}}
}

all_scores = {m: np.zeros((NUM_TRIALS, len(Q_LIST))) for m in methods_to_compare.keys()}

print("Starting Synergistic Experiment...", flush=True)
for trial in range(NUM_TRIALS):
    print(f"  Trial {trial + 1} / {NUM_TRIALS}", flush=True)
    X, Y, MODALITIES = generate_dynamic_synergistic_regression(n=N_SAMPLES, seed=trial)
    X = StandardScaler().fit_transform(X)
    split_idx = int(N_SAMPLES * 0.5)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_true_test = X[split_idx:], Y[split_idx:]

    shapmml_model = ShapMML(
        x=X_train, y=Y_train, modalities=MODALITIES,
        learning_fn=svr_learning_fn, predict_fn=svr_predict_fn,
        loss_fn=lambda y, yhat: (y - yhat) ** 2,
        task_type="regression", alpha=ALPHA_LEVEL, split=0.5
    )
    shapmml_model.train()
    shapmml_model.marginal_calibrate()
    
    std_shap = shapmml_model.compute_marginal_contributions(custom_weights=None)
    
    for scheme_name, config in methods_to_compare.items():
        mode = config["mode"]
        params = config["params"]
        
        if mode == "marg_std":
            y_preds = []
            for q in Q_LIST:
                if q == 0:
                    y_preds.append(np.full(len(Y_true_test), np.mean(Y_train)))
                    continue
                tau_upper = 1 - ALPHA_LEVEL / (2 * q)
                marg_upper = np.quantile(std_shap, tau_upper, axis=0)
                pos_idx = np.where(marg_upper > 0)[0]
                if len(pos_idx) == 0:
                    y_preds.append(np.full(len(Y_true_test), np.mean(Y_train)))
                else:
                    ranked = pos_idx[np.argsort(marg_upper[pos_idx])[::-1]]
                    S = tuple(sorted(shapmml_model.modality_list[j] for j in ranked[:min(q, len(ranked))].tolist()))
                    y_preds.append(svr_predict_fn(X_test * shapmml_model.mask(S), shapmml_model.mu[S]))
            for i, q in enumerate(Q_LIST): all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, y_preds[i])

        elif mode == "cond":
            if params == "delta_d":
                custom_w = get_delta_weights(shapmml_model.p, "d")
            elif params == "delta_1":
                custom_w = get_delta_weights(shapmml_model.p, "1")
            elif isinstance(params, tuple):
                custom_w = get_beta_weights(shapmml_model.p, params[0], params[1])
            else:
                custom_w = None
                
            shapmml_model.shapley_values = shapmml_model.compute_marginal_contributions(custom_weights=custom_w)
            
            for i, q in enumerate(Q_LIST):
                if q == 0:
                    all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, np.full(len(Y_true_test), np.mean(Y_train)))
                    continue
                best_params = shapmml_model.tune_hyperparameters(lambda1_grid=[1e-3, 1e-2], lambda2_grid=[1e-3, 1e-2], q=q, dim_reduce="svd", n_folds=3)
                shapmml_model.lambda1 = best_params["lambda1"]
                shapmml_model.lambda2 = best_params["lambda2"]
                shapmml_model.conditional_calibrate(q, dim_reduce="svd")
                y_cond, _ = shapmml_model.predict_optimal_modalities(X_test)
                all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, y_cond)

        elif mode == "cond_auto":
            beta_grid = [
                "std",            # Standard SHAP
                "delta_1",        # Extreme Skew Small (Empty Coalition)
                "delta_p",        # Extreme Skew Large (Grand Coalition)
                (1, 32), (1, 16), (1, 4), (1, 2),  
                (1, 1), (2, 2), (4, 4),            
                (2, 1), (4, 1), (16, 1), (32, 1)   
            ]
            
            for i, q in enumerate(Q_LIST):
                if q == 0:
                    all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, np.full(len(Y_true_test), np.mean(Y_train)))
                    continue
                
                # 1. Automatically find the best weights and Lambdas
                best_params = shapmml_model.tune_hyperparameters(
                    beta_grid=beta_grid, 
                    lambda1_grid=[1e-3, 1e-2], 
                    lambda2_grid=[1e-3, 1e-2], 
                    q=q, 
                    dim_reduce="svd", 
                    n_folds=3
                )
                
                # 2. Apply the winning weights
                if best_params["beta"] == "std":
                    custom_w = None
                elif best_params["beta"] == "delta_1":
                    custom_w = np.zeros(shapmml_model.p)
                    custom_w[0] = 1.0
                elif best_params["beta"] == "delta_p":
                    custom_w = np.zeros(shapmml_model.p)
                    custom_w[-1] = 1.0
                else:
                    a, b = best_params["beta"]
                    custom_w = get_beta_weights(shapmml_model.p, a, b)
                
                shapmml_model.shapley_values = shapmml_model.compute_marginal_contributions(custom_weights=custom_w)
                
                # 3. Apply the winning Lambdas and fit the final conditional layer
                shapmml_model.lambda1 = best_params["lambda1"]
                shapmml_model.lambda2 = best_params["lambda2"]
                shapmml_model.conditional_calibrate(q, dim_reduce="svd")
                
                # 4. Predict and Record
                y_cond, _ = shapmml_model.predict_optimal_modalities(X_test)
                all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, y_cond)

final_means = {m: np.mean(all_scores[m], axis=0) for m in all_scores.keys()}
final_stderr = {m: np.std(all_scores[m], axis=0) / np.sqrt(NUM_TRIALS) for m in all_scores.keys()}

print("\n" + "="*95)
print(f"SUMMARY TABLE")
print("="*95)

header = f"{'Method':<25} | " + " | ".join([f"q={q:<7}" for q in Q_LIST])
print(header)
print("-" * len(header))

for scheme_name in methods_to_compare.keys():
    row = f"{scheme_name:<25} | "
    means = final_means[scheme_name]
    stds = final_stderr[scheme_name]
    row_vals = []
    for m, s in zip(means, stds):
        row_vals.append(f"{m:5.1f} ± {s:<3.1f}")
    print(row + " | ".join(row_vals))
print("="*95)


plt.figure(figsize=(10, 6))
jitters = np.linspace(-0.15, 0.15, len(methods_to_compare))

for idx, (scheme_name, config) in enumerate(methods_to_compare.items()):
    fmt = config["fmt"]
    plt.errorbar(Q_LIST + jitters[idx], final_means[scheme_name], yerr=final_stderr[scheme_name],
                 marker=fmt["marker"], linestyle=fmt["ls"], linewidth=2, capsize=4, 
                 color=fmt["color"], label=scheme_name)

plt.xlabel("Max Number of Modalities (q)")
plt.ylabel(f"Test MSE over {NUM_TRIALS} Trials")
plt.xticks(Q_LIST)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("../output/synergistic_svr.pdf")  
plt.show()



NUM_TRIALS = 10
N_SAMPLES = 600
N_MODALITIES = 7
Q_LIST = np.arange(0, N_MODALITIES + 1)
ALPHA_LEVEL = 0.05

methods_to_compare = {
    "Marginal Std": {"mode": "marg_std", "params": None, "fmt": {"color": "lightblue", "marker": "s", "ls": "--"}},
    "Cond Std": {"mode": "cond", "params": None, "fmt": {"color": "blue", "marker": "o", "ls": "-"}},
    "Cond Skew Small (1,16)": {"mode": "cond", "params": (1, 16), "fmt": {"color": "purple", "marker": "D", "ls": "-."}},
    "Cond Skew Large (16,1)": {"mode": "cond", "params": (16, 1), "fmt": {"color": "red", "marker": "^", "ls": ":"}},
    "Cond Auto": {"mode": "cond_auto", "params": None, "fmt": {"color": "green", "marker": "*", "ls": "-"}}
}

all_scores = {m: np.zeros((NUM_TRIALS, len(Q_LIST))) for m in methods_to_compare.keys()}

print("Starting Redundant Experiment...", flush=True)
for trial in range(NUM_TRIALS):
    print(f"  Trial {trial + 1} / {NUM_TRIALS}", flush=True)
    X, Y, MODALITIES = generate_dynamic_redundant_regression(n=N_SAMPLES, seed=trial)
    X = StandardScaler().fit_transform(X)
    split_idx = int(N_SAMPLES * 0.5)
    X_train, Y_train, X_test, Y_true_test = X[:split_idx], Y[:split_idx], X[split_idx:], Y[split_idx:]

    shapmml_model = ShapMML(
        x=X_train, y=Y_train, modalities=MODALITIES,
        learning_fn=svr_learning_fn, predict_fn=svr_predict_fn,
        loss_fn=lambda y, yhat: (y - yhat) ** 2,
        task_type="regression", alpha=ALPHA_LEVEL, split=0.5
    )
    shapmml_model.train()
    shapmml_model.marginal_calibrate()
    std_shap = shapmml_model.compute_marginal_contributions(custom_weights=None)
    
    for scheme_name, config in methods_to_compare.items():
        mode, params = config["mode"], config["params"]
        
        if mode == "marg_std":
            for i, q in enumerate(Q_LIST):
                if q == 0:
                    all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, np.full(len(Y_true_test), np.mean(Y_train)))
                    continue
                marg_upper = np.quantile(std_shap, 1 - ALPHA_LEVEL / (2 * q), axis=0)
                pos_idx = np.where(marg_upper > 0)[0]
                if len(pos_idx) == 0:
                    all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, np.full(len(Y_true_test), np.mean(Y_train)))
                else:
                    S = tuple(sorted(shapmml_model.modality_list[j] for j in pos_idx[np.argsort(marg_upper[pos_idx])[::-1]][:q].tolist()))
                    all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, svr_predict_fn(X_test * shapmml_model.mask(S), shapmml_model.mu[S]))

        elif mode == "cond":
            if params == "delta_d":
                custom_w = get_delta_weights(shapmml_model.p, "d")
            elif params == "delta_1":
                custom_w = get_delta_weights(shapmml_model.p, "1")
            elif isinstance(params, tuple):
                custom_w = get_beta_weights(shapmml_model.p, params[0], params[1])
            else:
                custom_w = None

            shapmml_model.shapley_values = shapmml_model.compute_marginal_contributions(custom_weights=custom_w)
            
            for i, q in enumerate(Q_LIST):
                if q == 0:
                    all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, np.full(len(Y_true_test), np.mean(Y_train)))
                    continue
                best_params = shapmml_model.tune_hyperparameters(lambda1_grid=[1e-3, 1e-2], lambda2_grid=[1e-3, 1e-2], q=q, dim_reduce="svd", n_folds=3)
                shapmml_model.lambda1 = best_params["lambda1"]
                shapmml_model.lambda2 = best_params["lambda2"]
                shapmml_model.conditional_calibrate(q, dim_reduce="svd")
                y_cond, _ = shapmml_model.predict_optimal_modalities(X_test)
                all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, y_cond)

        elif mode == "cond_auto":
            beta_grid = [
                "std",            # Standard SHAP
                "delta_1",        # Extreme Skew Small (Empty Coalition)
                "delta_p",        # Extreme Skew Large (Grand Coalition)
                (1, 32), (1, 16), (1, 4), (1, 2),  
                (1, 1), (2, 2), (4, 4),            
                (2, 1), (4, 1), (16, 1), (32, 1)   
            ]
            
            for i, q in enumerate(Q_LIST):
                if q == 0:
                    all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, np.full(len(Y_true_test), np.mean(Y_train)))
                    continue
                
                # 1. Automatically find the best weights and Lambdas
                best_params = shapmml_model.tune_hyperparameters(
                    beta_grid=beta_grid, 
                    lambda1_grid=[1e-3, 1e-2], 
                    lambda2_grid=[1e-3, 1e-2], 
                    q=q, 
                    dim_reduce="svd", 
                    n_folds=3
                )
                
                # 2. Apply the winning weights
                
                if best_params["beta"] == "std":
                    custom_w = None
                elif best_params["beta"] == "delta_1":
                    custom_w = np.zeros(shapmml_model.p)
                    custom_w[0] = 1.0
                elif best_params["beta"] == "delta_p":
                    custom_w = np.zeros(shapmml_model.p)
                    custom_w[-1] = 1.0
                else:
                    a, b = best_params["beta"]
                    custom_w = get_beta_weights(shapmml_model.p, a, b)
                
                shapmml_model.shapley_values = shapmml_model.compute_marginal_contributions(custom_weights=custom_w)
                
                # 3. Apply the winning Lambdas and fit the final conditional layer
                shapmml_model.lambda1 = best_params["lambda1"]
                shapmml_model.lambda2 = best_params["lambda2"]
                shapmml_model.conditional_calibrate(q, dim_reduce="svd")
                
                # 4. Predict and Record
                y_cond, _ = shapmml_model.predict_optimal_modalities(X_test)
                all_scores[scheme_name][trial, i] = mean_squared_error(Y_true_test, y_cond)

final_means = {m: np.mean(all_scores[m], axis=0) for m in all_scores.keys()}
final_stderr = {m: np.std(all_scores[m], axis=0) / np.sqrt(NUM_TRIALS) for m in all_scores.keys()}

print("\n" + "="*95)
print(f"SUMMARY TABLE")
print("="*95)

header = f"{'Method':<25} | " + " | ".join([f"q={q:<7}" for q in Q_LIST])
print(header)
print("-" * len(header))

for scheme_name in methods_to_compare.keys():
    row = f"{scheme_name:<25} | "
    means = final_means[scheme_name]
    stds = final_stderr[scheme_name]
    row_vals = []
    for m, s in zip(means, stds):
        row_vals.append(f"{m:5.1f} ± {s:<3.1f}")
    print(row + " | ".join(row_vals))
print("="*95)

plt.figure(figsize=(10, 6))
jitters = np.linspace(-0.15, 0.15, len(methods_to_compare))

for idx, (scheme_name, config) in enumerate(methods_to_compare.items()):
    fmt = config["fmt"]
    plt.errorbar(Q_LIST + jitters[idx], final_means[scheme_name], yerr=final_stderr[scheme_name],
                 marker=fmt["marker"], linestyle=fmt["ls"], linewidth=2, capsize=4, 
                 color=fmt["color"], label=scheme_name)

plt.xlabel("Max Number of Modalities (q)")
plt.ylabel(f"Test MSE over {NUM_TRIALS} Trials")
plt.xticks(Q_LIST)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("../output/redundant_svr.pdf")  
plt.show()
