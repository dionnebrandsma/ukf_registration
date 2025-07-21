import utils
import numpy as np
from tqdm import tqdm
from scipy.linalg import cholesky, block_diag
from scipy.spatial.transform import Rotation as R

class UKFRegistration:
    def __init__(self, alpha=0.25, beta=3, kappa=500, var_trans=10, var_theta=10, isotropic=True, max_iter=100, threshold=0.3):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.var_trans = var_trans
        self.var_theta = var_theta
        self.isotropic = isotropic
        
        self.max_iter = max_iter
        self.threshold = threshold
        self.anneal_var = 1
        self.anneal_rate = 0.98

    def register(self, source: np.ndarray, target: np.ndarray, add_noise=True):
        assert source.shape[0] == 3 and target.shape[0] == 3

        Y_breve = target.copy()
        U_breve = source.copy()

        # State vector and covariance initialization
        xest_k = np.zeros(6)
        Pest_xk = np.eye(6)
        n = len(xest_k)

        lambda_ = self.alpha**2 * (n + self.kappa) - n

        # Measurement noise model
        if self.isotropic:
            var_yacute = 3
            Sigma_yacute = var_yacute * np.eye(3)
        else:
            Sigma_u = np.diag([3, 3, 3])
            Sigma_y = np.diag([10, 100, 1])
            Sigma_yacute = np.eye(3) @ Sigma_u @ np.eye(3).T + Sigma_y
        
        N_point = np.size(U_breve, axis=1)
        n_yacute = np.random.multivariate_normal([0,0,0], Sigma_yacute, N_point).T
        U = U_breve + n_yacute if add_noise else U_breve

        # Process noise model
        Sigma_x_k = np.diag([self.var_trans]*3 + [self.var_theta]*3)

        T_all = np.eye(4)
        best_T = np.eye(4)
        best_rmse = np.inf
        U_transformed = U.copy()

        for iter in tqdm(range(self.max_iter)):
            # Reset parameters
            xest_k = np.zeros(6)
            Pest_xk = np.eye(6)
            Sigma_x_k = Sigma_x_k * self.anneal_var

            # New point insertion
            point = min(iter + 1, N_point)
            rand_idx = np.random.choice(N_point, point, replace=False)
            U_k = U_transformed[:, rand_idx]

            # Closest point operator (k-d tree search)
            Y_k, distances = utils.find_correspondences(U_k, Y_breve)
            mean_dist = np.mean(distances)

            for k in range(point):
                # --- State vector prediction ---
                xest_kmin1 = xest_k # For convenience
                Pest_xkmin1 = Pest_xk
                
                # Compute sigma points
                xtilde_k, w_m, w_c = self._compute_sigma_points(xest_kmin1, Pest_xkmin1, lambda_)
                
                # Recover mu and sigma of the transformed sigma points
                xest_bar_k, Pest_bar_xk = self._recover_gaussian(xtilde_k, w_m, w_c)
                Pest_bar_xk += Sigma_x_k
                
                # --- State vector update ---
                u_k = U_k[:, :k+1]
                ytilde_k = self._h_transform(xtilde_k, u_k) # Transform estimated state to measurement
                yest_bar_k, P_yk = self._recover_gaussian(ytilde_k, w_m, w_c) # Recover gaussian in measurement space
                
                # Add P_yk with measurement noise
                temp = [1.25 * Sigma_yacute for _ in range(k+1)]
                Sigma_yacute_k = block_diag(*temp)
                temp = [10 * mean_dist * np.eye(3) for _ in range(k+1)]
                Sigma_int_k = block_diag(*temp)
                P_yk += Sigma_yacute_k + Sigma_int_k
                
                # Compute P_xkyk
                P_xkyk = sum(w_c[j] * np.outer(xtilde_k[:, j] - xest_bar_k, ytilde_k[:, j] - yest_bar_k) for j in range(xtilde_k.shape[1]))
                
                # Compute Kalman gain
                K_k = np.linalg.solve(P_yk, P_xkyk.T).T
                
                # Update state vector
                xest_k = xest_bar_k + K_k @ (Y_k[:, :k+1].flatten() - yest_bar_k)
                Pest_xk = Pest_bar_xk - K_k @ P_yk @ K_k.T

            R_xest_bar_theta = R.from_euler('ZYX', np.deg2rad(xest_k[3:6])).as_matrix()
            T_all = np.vstack((np.hstack((R_xest_bar_theta, xest_k[:3, None])), [0, 0, 0, 1])) @ T_all

            U_transformed = R_xest_bar_theta @ U_transformed + xest_k[:3, None]

            if mean_dist < best_rmse:
                best_rmse = mean_dist
                best_T = T_all.copy()

            # Convergence check
            if mean_dist < self.threshold:
                break
            self.anneal_var *= self.anneal_rate

        return best_T, best_rmse

    def _compute_sigma_points(self, mu, sigma, lambda_):
        n = len(mu)
        sigma_points = np.zeros((n, 2*n + 1))
        sigma_points[:, 0] = mu
        sqrt_matrix = np.sqrt(n + lambda_) * cholesky(sigma, lower=True)
        sigma_points[:, 1:n+1] = mu[:, None] + sqrt_matrix
        sigma_points[:, n+1:] = mu[:, None] - sqrt_matrix

        w_m = np.full(2*n + 1, 1/(2*(n + lambda_)))
        w_m[0] = lambda_/(n + lambda_)
        w_c = np.copy(w_m)
        w_c[0] += 1 - self.alpha**2 + self.beta
        return sigma_points, w_m, w_c

    def _recover_gaussian(self, sigma_points, w_m, w_c):
        mu = np.sum(w_m * sigma_points, axis=1)
        sigma = np.zeros((len(mu), len(mu)))
        for j in range(sigma_points.shape[1]):
            diff = sigma_points[:, j] - mu
            sigma += w_c[j] * np.outer(diff, diff)
        return mu, sigma

    def _h_transform(self, sigma_points, u_i):
        ytilde_k = []
        for j in range(sigma_points.shape[1]):
            t = sigma_points[:3, j]
            angles = sigma_points[3:6, j]
            R_j = R.from_euler('ZYX', np.deg2rad(angles)).as_matrix()
            y = R_j @ u_i + t[:, None]
            ytilde_k.append(y.flatten())
        return np.array(ytilde_k).T


