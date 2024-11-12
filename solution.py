import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import GPy
import scipy.special
import matplotlib.pyplot as plt
import time
# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
MAX_t = 20


# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # Storage for observations
        self.X = np.empty((0, 1))  # inputs
        self.Y = np.empty((0, 1))  # objective values (logP)
        self.V = np.empty((0, 1))  # constraint values (SA)

        # GP for objective function (logP) and constraint function (SA)
        self.kernel_f = GPy.kern.Matern52(input_dim=1, variance=0.5, lengthscale=1.0)
        self.kernel_v = GPy.kern.Add([
            GPy.kern.Linear(input_dim=1),
            GPy.kern.Matern52(input_dim=1, variance=0.5, lengthscale=1.0)
        ])

        self.model_f = None
        self.model_v = None

        self.UNIFORM_MODE = True
        self.t = 0  # time step

    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        
        if self.UNIFORM_MODE:
            res = np.linspace(*DOMAIN[0], 20)[self.t]
        else:
            res = self.optimize_acquisition_function()

        self.t += 1
        return res

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN, approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        """
        Below, you can find the quantitative details of this problem.
        • The domain is X = [0, 10].
        • The noise perturbing the observation is Gaussian with standard deviation of = 0.15 and
        0 = 0.0001 for logP and SA, respectively.
        • The mapping f can be effectively modeled with a Matérn with smoothness parameter v =2.5 or a RBF kernel with variance 0.5, lengthscale 10, 1 or 0.5. To achieve the best result, we recommend tuning the kernel and the lengthscale parameter.
        • The mapping u can be effectively modeled with an additive kernel composed of a Linear kernel and a Matérn with smoothness parameter v = 2.5 or a RBF kernel with variance v2, lengthscale 10, 1 or 0.5. And the prior mean should be 4. To achieve the best result, we recommend tuning the kernel and the lengthscale parameter.
        • The maximum tolerated SA is K = 4.
        """

        # Skip if no model exists yet
        if self.model_f is None or self.model_v is None:
            return 0.0

        # Get predictions from both GPs
        mu_f, var_f = self.model_f.predict(x)
        mu_v, var_v = self.model_v.predict(x)

        # Standard deviations
        sigma_f = np.sqrt(var_f)
        sigma_v = np.sqrt(var_v)

        # Probability of constraint satisfaction (v <= SAFETY_THRESHOLD)
        # Using CDF of normal distribution
        prob_safe = 0.5 * (1 + scipy.special.erf((SAFETY_THRESHOLD - mu_v) / (np.sqrt(2) * sigma_v)))

        # Expected Improvement with constraint
        z = (mu_f - np.max(self.Y)) / sigma_f
        ei = sigma_f * (z * 0.5 * (1 + scipy.special.erf(z / np.sqrt(2))) + 1/np.sqrt(2*np.pi) * np.exp(-0.5 * z**2))
        
        # Constrained Expected Improvement
        return float(ei * prob_safe)

    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        print(f"Adding observation: x={x}, f={f}, v={v}")
        x = np.atleast_2d(x)
        self.X = np.vstack((self.X, x))
        self.Y = np.vstack((self.Y, f))
        self.V = np.vstack((self.V, v))
        
        # Update or create the GP models
        if self.model_f is None:
            self.model_f = GPy.models.GPRegression(self.X, self.Y, self.kernel_f)
            self.model_v = GPy.models.GPRegression(self.X, self.V, self.kernel_v, mean_function=GPy.mappings.Constant(input_dim=1, output_dim=1, value=4.0))
        else:
            self.model_f.set_XY(self.X, self.Y)
            self.model_v.set_XY(self.X, self.V)
        
        # Optimize the hyperparameters (optional, but recommended)
        # self.model_f.optimize_restarts(num_restarts=1)
        # self.model_v.optimize_restarts(num_restarts=1)
        self.model_f.optimize()
        self.model_v.optimize()

        if self.t == MAX_t:  # Last iteration, self.t is 0-indexed
            self.dump_model_data()

    def dump_model_data(self):
        # Sample posterior at N points across domain
        X_test = np.linspace(*DOMAIN[0], 100).reshape(-1, 1)
        
        # Get predictions for objective and constraint
        mu_f, var_f = self.model_f.predict(X_test)
        mu_v, var_v = self.model_v.predict(X_test)
        
        sigma_f = np.sqrt(var_f)
        sigma_v = np.sqrt(var_v)
        
        print("\nPosterior at final iteration:")
        print("x,f_mean,f_std,v_mean,v_std")
        for i in range(0, len(X_test)):
            print(f"{X_test[i,0]:.3f},{mu_f[i,0]:.3f},{sigma_f[i,0]:.3f},{mu_v[i,0]:.3f},{sigma_v[i,0]:.3f}")

        # Print observed data
        print("\nObserved data:")
        print("x,f,v")
        for i in range(len(self.X)):
            print(f"{self.X[i,0]:.3f},{self.Y[i,0]:.3f},{self.V[i,0]:.3f}")

        time.sleep(0.5)
        raise Exception("Stop here")


    

    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # Find points that are likely to be safe
        X_test = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 1000).reshape(-1, 1)
        mu_v, var_v = self.model_v.predict(X_test)
        sigma_v = np.sqrt(var_v)
        
        # Use scipy.special.erf for vectorized operations
        prob_safe = 0.5 * (1 + scipy.special.erf((SAFETY_THRESHOLD - mu_v) / (np.sqrt(2) * sigma_v)))
        safe_idx = prob_safe >= 0.95  # Conservative threshold for safety
        
        if not np.any(safe_idx):
            # If no safe points found, return the safest observed point
            safe_obs_idx = self.V.flatten() <= SAFETY_THRESHOLD
            if np.any(safe_obs_idx):
                best_safe_idx = np.argmax(self.Y[safe_obs_idx])
                return float(self.X[safe_obs_idx][best_safe_idx])
            return float(self.X[np.argmin(self.V)])  # Return safest point if no safe points
            
        # Among likely safe points, find the one with highest predicted objective
        X_safe = X_test[safe_idx].reshape(-1, 1)  # Ensure 2D array shape
        mu_f, _ = self.model_f.predict(X_safe)
        best_idx = np.argmax(mu_f)
        
        return float(X_safe[best_idx])
        

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        # Plot objective posterior
        self.model_f.plot()
        plt.savefig('objective_posterior.png')
        plt.close()

        # Plot constraint posterior
        self.model_v.plot()
        plt.savefig('constraint_posterior.png')
        plt.close()


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function recommend_next must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
