import cvxpy as cp
from sklearn.metrics.pairwise import pairwise_kernels
from itertools import chain, combinations
import numpy as np
from typing import Callable
import math
from scipy.stats import beta, t
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from typing import Literal
import pandas as pd
import warnings

def get_beta_weights(p, a, b):
        weights = np.array([beta.pdf((r + 0.5) / p, a=a, b=b) for r in range(p)])
        norm = np.sum([weights[r] * math.comb(p-1, r) for r in range(p)])
        return weights / norm if norm > 0 else None

class ShapMML:
    def __init__(
            self, 
            x : np.ndarray,
            y : np.ndarray,
            modalities: dict[int, list[int]],
            learning_fn : Callable,
            predict_fn : Callable,
            loss_fn : Callable,
            task_type : Literal["regression", "classification"],
            lambda1 : float = None,  # Regularization for Kernel component
            lambda2 : float = None,  # Regularization for Bias/Linear component
            alpha = 0.05,           # Target coverage (e.g., 0.05 for 95th percentile)
            split : float = 0.5,
            covariates: list[int] = None,
        ):
        self.shape = x.shape[1:]
        self.n = len(y)
        self.split = split
        self.m = int(split * self.n)
        self.n_cal = self.n - self.m
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.x_train = self.x[:self.m]
        self.x_calib = self.x[self.m:]
        self.y_train = self.y[:self.m]
        self.y_calib = self.y[self.m:]
        self.modalities = modalities
        self.covariates = covariates
        self.p = len(modalities)
        modality_list = list(modalities.keys())
        self.modality_list = modality_list
        self.modality_power_set = list(chain.from_iterable(combinations(modality_list, j) for j in range(len(modality_list) + 1)))
        self.learning_fn = learning_fn
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.loss_fn = loss_fn
        self.h = {} # Will store the predictors
        self.predict_fn = predict_fn
        self.task_type = task_type
        self.output_dim = len(np.unique(self.y_train)) if self.task_type == 'classification' else 1
        self.utility = dict()

    def mask(self, S):
        mask = np.zeros(self.shape, dtype=np.float32)
        if S is not None:
            for j in S:
                for idx in self.modalities[j]:
                    mask[idx] = 1.0
        return mask


    def train(self):
        mu = dict()
        for S in self.modality_power_set:
            x_train_mask = self.x_train * self.mask(S)
            mu[S] = self.learning_fn(x_train_mask, self.y_train)
        self.mu = mu

    def t_p_value(self, shapley_values):
        n_cal, p = shapley_values.shape
        alpha = self.alpha

        rows = []

        for j in range(p):
            phi_j = shapley_values[:, j]

            ell = int(np.ceil((n_cal + 1) * alpha / 2))
            ell = max(ell, 1)

            u = int(np.ceil((n_cal + 1) * (1 - alpha / 2)))
            u = min(u, n_cal)

            # Lower conformal bound
            L_j = np.sort(phi_j)[ell - 1]
            U_j = np.sort(phi_j)[u - 1]
            # Conformal p-value
            # p_value = (1 + np.sum(phi_j <= 0.0)) / (n_cal + 1)
            mean_j = np.mean(phi_j)
            std_j = np.std(phi_j, ddof=1)
            t_stat = mean_j / (std_j / np.sqrt(n_cal))
            if std_j < 1e-12:
                p_value = 1.0
            else:
                p_value = 1 - t.cdf(t_stat, df=n_cal - 1)  # one-sided test H0: mean <= 0
            rows.append({
                "modality": j,
                "lower_bound": L_j,
                "upper_bound": U_j,
                "p_value": p_value
               # "reject_H0_at_alpha": L_j > 0
            })

        return pd.DataFrame(rows)
    
    def compute_marginal_contributions(self, custom_weights=None):
        """Computes feature attributions given a weight vector for each coalition size."""
        if not self.utility:
            raise ValueError("Run marginal_calibrate() first to compute utilities.")
        
        p = self.p
        shapley_values = np.zeros((self.n_cal, p), dtype = np.float32)

        if custom_weights is None:
            fact = np.array([math.factorial(i) for i in range(p+1)], dtype=np.float32)
            custom_weights = [fact[r] * fact[p - r - 1] / fact[p] for r in range(p)]

        for i in range(self.p):
            others = [j for j in range(p) if j != i]
            for r in range(len(others)+1):
                weight = np.float32(custom_weights[r])
                if weight == 0: continue

                for subset in combinations(others, r):
                    S = tuple(sorted(self.modality_list[j] for j in subset))
                    S_i = tuple(sorted(self.modality_list[j] for j in subset + (i,)))
                    v_S = self.utility.get(S, 0)
                    v_Si = self.utility.get(S_i, 0)
                    shapley_values[:, i] += weight * (v_Si - v_S)
        return shapley_values

    def marginal_calibrate(self):
        """Computes and caches the utility map so we can quickly calculate different SHAP schemes."""
        baseline = self.loss_fn(self.y_calib, self.predict_fn(self.x_calib * self.mask(()), self.mu[()]
))
        for S in self.modality_power_set:
            if S == ():
                loss = baseline
            else:
                x_calib_mask = self.x_calib * self.mask(S)
                y_pred = self.predict_fn(x_calib_mask, self.mu[S])
                loss = self.loss_fn(self.y_calib, y_pred)
            self.utility[S] = baseline - loss

        # Default to standard SHAP values initially
        self.shapley_values = self.compute_marginal_contributions(custom_weights=None)
        return self.shapley_values

    def tune_hyperparameters(self, lambda1_grid, lambda2_grid, q, beta_grid=None, dim_reduce="svd", n_folds=3, random_state=0):
        if q == 0: 
            return {"beta": "std" if beta_grid else "current", "lambda1": 0.0, "lambda2": 0.0}
        
        X = self.x_calib
        Y = self.y_calib  # We need the true labels for the validation folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # Track the lowest downstream loss instead of highest utility
        best_loss = np.inf
        best_params = {"beta": None, "lambda1": None, "lambda2": None}

        actual_beta_grid = beta_grid if beta_grid else ["current"]

        for beta_params in actual_beta_grid:
            # 1. Compute custom SHAP values used to TRAIN the conditional model
            if beta_params == "current":
                Phi_current = self.shapley_values
            elif beta_params == "std":
                custom_w = None
                Phi_current = self.compute_marginal_contributions(custom_weights=custom_w)
            elif beta_params == "delta_1":
                custom_w = np.zeros(self.p)
                custom_w[0] = 1.0
                Phi_current = self.compute_marginal_contributions(custom_weights=custom_w)
            elif beta_params == "delta_p":
                custom_w = np.zeros(self.p)
                custom_w[-1] = 1.0
                Phi_current = self.compute_marginal_contributions(custom_weights=custom_w)
            else:
                a, b = beta_params
                custom_w = get_beta_weights(self.p, a, b)
                Phi_current = self.compute_marginal_contributions(custom_weights=custom_w)

            # 2. Tune the Lambdas for this specific SHAP distribution
            for lam1 in lambda1_grid:
                for lam2 in lambda2_grid:
                    fold_losses = []
                    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                        # X_train, Phi_train are used to fit the RKHS model
                        X_train, Phi_train = X[train_idx], Phi_current[train_idx]
                        
                        # X_val, y_val are used to test the downstream predictive performance
                        X_val, y_val = X[val_idx], Y[val_idx]
                        
                        self.lambda1, self.lambda2 = lam1, lam2
                        
                        # Cache ID omits beta to reuse SVD/Kernel matrices across beta grid searches
                        fold_data_id = f"tuning_fold_{fold_idx}"
                        
                        # Calibrate on the inner training fold
                        self.conditional_calibrate(q=q, dim_reduce=dim_reduce, X_calib=X_train, Phi_calib=Phi_train, cache_id=fold_data_id)
                        
                        # Predict SHAP quantiles on the inner validation fold
                        h_val = self.predict(X_val, cache_key=(dim_reduce, None, None, fold_data_id))
                        
                        # --- Evaluate True Downstream Loss ---
                        n_val = len(X_val)
                        if self.task_type == 'classification':
                            y_val_pred = np.zeros((n_val, self.output_dim), dtype=np.float32)
                        else:
                            y_val_pred = np.zeros((n_val,), dtype=np.float32)
                            
                        masked_X_val = np.zeros_like(X_val, dtype=np.float32)
                        groups = {}
                        
                        # Map each validation sample to its optimally selected modality subset S
                        for i in range(n_val):
                            scores_i = h_val[i, :]
                            pos_idx = np.where(scores_i > 0)[0]
                            
                            if len(pos_idx) == 0:
                                S = ()
                            else:
                                ranked = pos_idx[np.argsort(scores_i[pos_idx])[::-1]]
                                min_q = min(q, len(ranked))
                                selected_indices = ranked[:min_q]
                                mylist = selected_indices.tolist()
                                S = tuple(sorted(self.modality_list[j] for j in mylist))
                                
                            masked_X_val[i] = X_val[i] * self.mask(S)
                            
                            if S not in groups:
                                groups[S] = []
                            groups[S].append(i)
                            
                        # Batch predict using the pre-cached models for each selected subset
                        for S, indices in groups.items():
                            model_mu = self.mu[S]
                            x_group = masked_X_val[indices]
                            
                            if x_group.ndim == X_val.ndim - 1:
                                x_group = x_group[None, ...]
                                
                            y_group_pred = self.predict_fn(x_group, model_mu)
                            
                            if self.task_type == 'classification':
                                y_val_pred[indices, :] = y_group_pred
                            else:
                                y_val_pred[indices] = y_group_pred
                                
                        # Calculate the actual predictive loss
                        fold_loss_array = self.loss_fn(y_val, y_val_pred)
                        fold_losses.append(np.mean(fold_loss_array))
                    
                    mean_loss = np.mean(fold_losses)
                    
                    # Update best parameters if downstream loss is minimized
                    if mean_loss < best_loss:
                        best_loss = mean_loss
                        best_params = {"beta": beta_params, "lambda1": lam1, "lambda2": lam2}
                        
        return best_params
    
    def tune_independent_pinball(self, lambda1_grid, lambda2_grid, q=1, dim_reduce="svd", n_folds=3, random_state=123):
        """Tunes lambda1 and lambda2 independently for each modality using validation Pinball Loss."""
        self.q = q
        tau = 1 - self.alpha / (2 * q)
        
        # 1. Fix SHAP values to uniform weights
        Phi = self.compute_marginal_contributions(custom_weights=None)
        X = self.x_calib
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Track the best lambdas and lowest pinball loss for EACH modality independently
        best_lambdas = {j: {"lambda1": None, "lambda2": None} for j in range(self.p)}
        best_losses = {j: np.inf for j in range(self.p)}
        
        for lam1 in lambda1_grid:
            for lam2 in lambda2_grid:
                # Temporarily set global lambdas for the conditional_calibrate call
                self.lambda1 = lam1
                self.lambda2 = lam2
                
                fold_losses = {j: [] for j in range(self.p)}
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                    X_train, Phi_train = X[train_idx], Phi[train_idx]
                    X_val, Phi_val = X[val_idx], Phi[val_idx]
                    
                    fold_data_id = f"tuning_fold_{fold_idx}_pinball"
                    
                    # Fit on training fold
                    self.conditional_calibrate(q=q, dim_reduce=dim_reduce, X_calib=X_train, Phi_calib=Phi_train, cache_id=fold_data_id)
                    
                    # Predict on validation fold
                    h_val = self.predict(X_val, cache_key=(dim_reduce, None, None, fold_data_id))
                    
                    # Calculate Pinball Loss for each modality independently
                    for j in range(self.p):
                        u_j = Phi_val[:, j] - h_val[:, j]
                        pinball = np.maximum(tau * u_j, (tau - 1) * u_j)
                        fold_losses[j].append(np.mean(pinball))
                
                # Check if this lambda pair beat the current best for any individual modality
                for j in range(self.p):
                    mean_loss = np.mean(fold_losses[j])
                    if mean_loss < best_losses[j]:
                        best_losses[j] = mean_loss
                        best_lambdas[j]["lambda1"] = lam1
                        best_lambdas[j]["lambda2"] = lam2
                        
        return best_lambdas

    def conditional_calibrate(self, 
                              q, 
                              dim_reduce: Literal["pca", "svd"], 
                              X_calib = None, 
                              Phi_calib = None, 
                              n_components = None,
                              condition_on: tuple[int] | None = None,
                              verbose=False,
                              cache_id=None):
        # ---------------------------------------------------------
        # Semi-Parametric RKHS Quantile Regression
        # Model: h_j(x) = K(x, X_calib)^T * eta_j + x^T * beta_j
        # ---------------------------------------------------------
        if q == 0:
            self.q = 0
            self.selected_modalities_ = []
            return

        if X_calib is None:
            X_calib = self.x_calib
            n_cal = self.n_cal
            if cache_id is None:
                cache_id = "global_calibration_set"
        else:
            n_cal = X_calib.shape[0]
            if cache_id is None:
                if X_calib is self.x_calib:
                    cache_id = "global_calibration_set"
                else:
                    cache_id = id(X_calib)
        if Phi_calib is None:
            Phi_calib = self.shapley_values
        if X_calib is not None and Phi_calib is None:
            raise ValueError("Phi_calib must be provided when X_calib is provided")

        # Prepare Data
        X_calib_full = X_calib
        if self.covariates is not None:
            cond_mask = np.zeros(self.shape)
            cond_mask[self.covariates] = 1.0
            X_calib = X_calib_full * cond_mask

        else:
            X_calib = X_calib_full


        assert X_calib.shape[0] == Phi_calib.shape[0], (
            X_calib.shape, Phi_calib.shape
        )

        X_gamma = X_calib.reshape(n_cal, -1)

        if not hasattr(self, "_cond_cache"):
            self._cond_cache = {}

        cache_key = (
            dim_reduce,
            n_components,
            condition_on,
            cache_id
            )
        
        self._last_cache_key = cache_key


        
        if cache_key not in self._cond_cache:
            # Center the data
            mu = np.mean(X_gamma, axis=0)
            X_gamma_centered = X_gamma - mu

            if dim_reduce == "pca":
                pca = PCA(n_components=n_components, whiten=False, random_state=0)
                V_prime_calib = pca.fit_transform(X_gamma_centered)
                Q = None
            elif dim_reduce == "svd":
                _, _, Vh = np.linalg.svd(X_gamma_centered, full_matrices=False)
                Q = Vh.T 
                for k in range(Q.shape[1]):
                    idx = np.argmax(np.abs(Q[:, k]))
                    if Q[idx, k] < 0:
                        Q[:, k] *= -1


                pca = None
                V_prime_calib = X_gamma_centered @ Q 

                if n_components is not None:
                    Q = Q[:, :n_components]
                    V_prime_calib = V_prime_calib[:, :n_components]

            else:
                raise ValueError("dim_reduce must be 'pca' or 'svd'")
        
            # Kernel Matrix (RBF)
            dim_v_prime = V_prime_calib.shape[1]
            gamma = 1.0 / (dim_v_prime if dim_v_prime > 0 else 1)
            K = pairwise_kernels(V_prime_calib, V_prime_calib, metric='rbf', gamma=gamma)
            K += 1e-2 * np.eye(n_cal)

            
            self._cond_cache[cache_key] = {
                "mu": mu,
                "Q": Q,
                "V_prime_calib": V_prime_calib,
                "K": K,
                "pca": pca,
                "gamma": gamma
            }
        
        cache = self._cond_cache[cache_key]
        mu = cache["mu"]
        Q = cache["Q"]
        V_prime_calib = cache["V_prime_calib"]
        K = cache["K"]
        pca = cache["pca"]
        gamma = cache["gamma"]

        dim_v_prime = V_prime_calib.shape[1]

        # storage for warm starts
        if not hasattr(self, "_warm_start"):
            self._warm_start = {}

        # Convert lambdas to tuples if they are dicts so they can be hashed
        l1_hashable = tuple(sorted(self.lambda1.items())) if isinstance(self.lambda1, dict) else self.lambda1
        l2_hashable = tuple(sorted(self.lambda2.items())) if isinstance(self.lambda2, dict) else self.lambda2

        warm_key = (cache_key, l1_hashable, l2_hashable)

        eta_opt = np.zeros((n_cal, self.p))
        beta_opt = np.zeros((dim_v_prime, self.p))
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-4 * np.eye(n_cal))

        for j in range(self.p):
            eta_j = cp.Variable(n_cal)
            beta_j = cp.Variable(dim_v_prime)

            pred_j = K @ eta_j + V_prime_calib @ beta_j
            u_j = Phi_calib[:, j] - pred_j

            tau = 1 - self.alpha / (2 * q)
            pinball = cp.maximum(tau * u_j, (tau - 1) * u_j)
            loss = cp.sum(pinball) / n_cal

            

            # Check if lambdas are specific to modality j, otherwise use global float
            lam1_j = self.lambda1[j] if isinstance(self.lambda1, dict) else self.lambda1
            lam2_j = self.lambda2[j] if isinstance(self.lambda2, dict) else self.lambda2

            reg = lam1_j * cp.sum_squares(L.T @ eta_j) \
                + lam2_j * cp.sum_squares(beta_j)
            eps = 1e-8
            reg += eps * cp.sum_squares(cp.hstack([beta_j, eta_j]))

            prob = cp.Problem(cp.Minimize(loss + reg))
            solved = False
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Try OSQP
                try:
                    prob.solve(
                        solver=cp.OSQP,
                        warm_start=False,
                        eps_abs=1e-5,  
                        eps_rel=1e-5,  
                        max_iter=10000, 
                        verbose=verbose,
                        polish=False,
                        adaptive_rho=True
                    )
                    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        solved = True
                except (cp.error.SolverError, ValueError):
                    pass 

                # Try CLARABEL
                if not solved:
                    try:
                        prob.solve(solver=cp.CLARABEL, verbose=verbose)
                        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                            solved = True
                    except (cp.error.SolverError, ValueError, AttributeError):
                        # AttributeError in case old CVXPY version doesn't have CLARABEL
                        pass

                # Try ECOS
                if not solved:
                    try:
                        prob.solve(
                            solver=cp.ECOS,
                            warm_start=True,
                            abstol=1e-5,
                            reltol=1e-5,
                            verbose=verbose
                        )
                        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                            solved = True
                    except (cp.error.SolverError, ValueError):
                        pass
                    
                # Try SCS 
                if not solved:
                    try:
                        prob.solve(
                            solver=cp.SCS,
                            warm_start=False,
                            eps=1e-4, 
                            max_iters=5000,
                            verbose=verbose
                        )
                        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                            solved = True
                    except (cp.error.SolverError, ValueError):
                        pass

            if not solved:
                print(f"Warning: Optimization failed for modality {j}. Status: {prob.status}")
                eta_opt[:, j] = 0
                beta_opt[:, j] = 0
            else:
                # Safely access value only if solved
                if eta_j.value is not None:
                    eta_opt[:, j] = eta_j.value
                    beta_opt[:, j] = beta_j.value
                else:
                    eta_opt[:, j] = 0
                    beta_opt[:, j] = 0


        # save solution 
        self._warm_start[warm_key] = (eta_opt.copy(), beta_opt.copy())

        # Create predictor closure
        def create_predictor(eta_col, beta_col, V_prime_calib_ref):
            def predictor(x_test, Q_ref, mu_ref, pca_ref, gamma_ref):
                n_test = x_test.shape[0]
                x_test_flat = x_test.reshape(n_test, -1)
                X_phi_centered_test = x_test_flat - mu_ref
                
                if Q_ref is None:
                    V_prime_test = pca_ref.transform(X_phi_centered_test)
                else:
                    V_prime_test = X_phi_centered_test @ Q_ref

                K_test = pairwise_kernels(V_prime_test, V_prime_calib_ref, metric='rbf', gamma=gamma_ref)
                rkhs_term = K_test @ eta_col
                linear_term = V_prime_test @ beta_col
                return rkhs_term + linear_term
            return predictor

        h = {}
        for j in range(self.p):
            h[j] = create_predictor(eta_opt[:, j], beta_opt[:, j], V_prime_calib)
            
        self.h = h
        self.q = q

    def predict(self, x_test, cache_key = None):
        if cache_key is None:
            if not hasattr(self, "_last_cache_key"):
                raise RuntimeError("No calibrated model available")
            cache_key = self._last_cache_key

        if cache_key not in self._cond_cache:
             raise KeyError(f"Cache key {cache_key} not found. Available keys: {list(self._cond_cache.keys())}")
        
        # Uses the stored transformation parameters Q and mu
        Q_ref = self._cond_cache[cache_key]['Q']
        mu_ref = self._cond_cache[cache_key]['mu']
        pca_ref = self._cond_cache[cache_key]['pca']
        gamma_ref = self._cond_cache[cache_key]['gamma']
        
        n_test = x_test.shape[0]
        preds = np.zeros((n_test, self.p))
        
        for i in range(self.p):
            # Pass the transformation parameters Q and mu to the predictor function
            preds[:, i] = self.h[i](x_test, Q_ref, mu_ref, pca_ref, gamma_ref)
            
        return preds
    
    def t_p_value_conditional(self, condition_on, condition_level, cache_key = None):

        if isinstance(condition_on, int):
            condition_on = (condition_on,)

        columns = condition_on
        k = len(columns)


        # --- coerce first ---
        condition_level = np.asarray(condition_level, dtype=float)

        # scalar -> vector
        if condition_level.ndim == 0:
            condition_level = np.full(k, condition_level)

        assert condition_level.shape == (k,), (
            condition_level.shape, k
        )



        x_star = self.x_calib.copy() 
        x_star[:, columns] = condition_level
        conf_shap = self.predict(x_star)
        
        n_cal, p = conf_shap.shape
        alpha = self.alpha

        rows = []

        for j in range(p):
            phi_j = conf_shap[:, j]

            # Conformal p-value
            conformal_p_value = (1 + np.sum(phi_j <= 0.0)) / (n_cal + 1)
            mean_j = np.mean(phi_j)
            std_j = np.std(phi_j, ddof=1)
            t_stat = mean_j / (std_j / np.sqrt(n_cal))
            if std_j < 1e-12:
                global_p_value = 1.0
            else:
                global_p_value = 1 - t.cdf(t_stat, df=n_cal - 1)  # one-sided test H0: mean <= 0
            rows.append({
                "modality": j,
                "conformal_p_value": conformal_p_value,
                "global_p_value": global_p_value
               # "reject_H0_at_alpha": L_j > 0
            })

        return pd.DataFrame(rows), conf_shap
    
    def select_and_mask(self, x_test: np.ndarray) -> tuple[np.ndarray, list]:
        
        n_test = x_test.shape[0]
        
        # 1. Predict the conditional quantile h_j for all modalities (h returns (n_test, p))
        # This uses the method implemented in the previous step.
        h_scores = self.predict(x_test) 
        
        # Initialize output storage
        masked_x_test = np.zeros_like(x_test, dtype=np.float32)
        selected_modalities_list = []

        for i in range(n_test):
            # 2. Select the top q modalities based on h_scores
            
            # Get the scores for observation i
            scores_i = h_scores[i, :]
            
            pos_idx = np.where(scores_i > 0)[0]

            if len(pos_idx) == 0:
                selected_modalities_list.append(())
                continue

            ranked = pos_idx[np.argsort(scores_i[pos_idx])[::-1]]
            min_q = min(self.q, len(ranked))
            selected_indices = ranked[:min_q]
            
            # S is the set of modality indices for this observation
            mylist = selected_indices.tolist()
            S = tuple(sorted(self.modality_list[j] for j in mylist))
            
            selected_modalities_list.append(S)
            
            # 3. Apply the mask for the selected set S
            # The mask generation needs the context of the modalities dictionary.
            # We need to reshape the mask to apply it to x_test[i]
            modality_mask = self.mask(S)
            
            # Apply mask and store the result
            masked_x_test[i] = x_test[i] * modality_mask

        return masked_x_test, selected_modalities_list
    
    def predict_optimal_modalities(self, x_test: np.ndarray) -> np.ndarray:
        if self.q == 0:
            n_test = x_test.shape[0]

            if self.task_type == "classification":
                # Predict the marginal class distribution
                class_probs = np.bincount(
                    self.y_train.astype(int),
                    minlength=self.output_dim
                ).astype(float)
                class_probs /= class_probs.sum()

                y_pred = np.tile(class_probs, (n_test, 1))
            else:
                # Regression: marginal mean
                y_pred = np.full(n_test, self.y_train.mean())

            return y_pred, [tuple()] * n_test


        # Select the top modalities and mask the input
        masked_x_test, selected_modalities_list = self.select_and_mask(x_test)
                
        
        # Initialize predictions (assuming output shape matches y_train)
        n_test = x_test.shape[0]
        if self.task_type == 'classification':
            y_pred = np.zeros((n_test, self.output_dim), dtype=np.float32)
        else:
            y_pred = np.zeros((n_test,), dtype=np.float32)  # scalar per sample


        
        # Group test observations by the selected modality set S
        # This avoids re-running prediction for the same trained model mu_S
        groups = {}
        for i, S in enumerate(selected_modalities_list):
            if S not in groups:
                groups[S] = []
            groups[S].append(i)
            
        # 2. Iterate through groups and make predictions
        for S, indices in groups.items():
            # S_cols = tuple(self.modality_list[j] for j in S)
            S_cols = S
            model_mu = self.mu[S_cols]
            x_group = masked_x_test[indices]

            # --- enforce batch dimension ---
            if x_group.ndim == x_test.ndim - 1:
                x_group = x_group[None, ...]
            # -----------------------------------

            y_group_pred = self.predict_fn(x_group, model_mu)
            
            if self.task_type == 'classification':
                y_pred[indices, :] = y_group_pred
            else:
                y_pred[indices] = y_group_pred

 
        return y_pred, selected_modalities_list
