import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from sklearn.metrics import mean_squared_error


class Kernel:
    def __init__(self, name, params=None, left=None, right=None, operation=None):

        self.name = name  # e.g., 'SE', 'Periodic'
        self.params = params or {}
        self.left = left
        self.right = right
        self.operation = operation  # 'add' or 'mul'

    def __call__(self, x1, x2):

        if self.operation == "add":
            return self.left(x1, x2) + self.right(x1, x2)

        elif self.operation == "mul":
            return self.left(x1, x2) * self.right(x1, x2)

        else:
            return self.kernel_function(x1, x2)

    def kernel_function(self, x1, x2):

        # Implement the basic kernels

        if self.name == "SE":

            lengthscale = self.params.get("lengthscale", 1.0)
            variance = self.params.get("variance", 1.0)
            sqdist = np.subtract.outer(x1, x2) ** 2
            return variance * np.exp(-0.5 * sqdist / lengthscale**2)

        elif self.name == "Periodic":

            lengthscale = self.params.get("lengthscale", 1.0)
            variance = self.params.get("variance", 1.0)
            period = self.params.get("period", 1.0)
            diff = np.subtract.outer(x1, x2)
            return variance * np.exp(
                -2 * (np.sin(np.pi * diff / period) ** 2) / lengthscale**2
            )

        elif self.name == "Linear":

            variance = self.params.get("variance", 1.0)
            return variance * np.outer(x1, x2)

        elif self.name == "WhiteNoise":

            noise_level = self.params.get("noise_level", 1e-3)
            return noise_level * np.eye(len(x1))

        else:

            raise ValueError(f"Unknown kernel {self.name}")

    def get_params(self):

        # Recursively get parameters
        if self.operation in ["add", "mul"]:
            return {**self.left.get_params(), **self.right.get_params()}

        else:
            return {self.name: self.params}

    def set_params(self, params):

        # Recursively set parameters
        if self.operation in ["add", "mul"]:
            self.left.set_params(params)
            self.right.set_params(params)

        else:

            if self.name in params:
                self.params.update(params[self.name])

    def __str__(self):

        if self.operation == "add":
            return f"({self.left} + {self.right})"

        elif self.operation == "mul":
            return f"({self.left} * {self.right})"

        else:
            return self.name


class Particle:
    def __init__(self, kernel):

        self.kernel = kernel
        self.weight = 1.0  # Initialize with uniform weight

    def copy(self):
        return deepcopy(self)

    def __str__(self):
        return str(self.kernel)


class SMCStructureLearning:
    def __init__(self, num_particles=10, num_rejuvenation_steps=1, max_kernel_depth=3):

        self.num_particles = num_particles
        self.num_rejuvenation_steps = num_rejuvenation_steps
        self.max_kernel_depth = max_kernel_depth
        self.particles = []
        self.weights = []

    def initialize_particles(self):

        for _ in range(self.num_particles):

            # Sample initial kernel structures and parameters
            kernel = self.sample_initial_kernel()
            particle = Particle(kernel)
            self.particles.append(particle)
            self.weights.append(1.0 / self.num_particles)

    def sample_initial_kernel(self):

        # Define possible base kernels
        base_kernels = ["SE", "Periodic", "Linear"]
        name = random.choice(base_kernels)
        params = self.sample_params(name)
        return Kernel(name, params)

    def sample_params(self, kernel_name):

        if kernel_name == "SE":

            return {
                "lengthscale": np.random.gamma(2.0, 1.0),
                "variance": np.random.gamma(2.0, 1.0),
            }

        elif kernel_name == "Periodic":

            return {
                "lengthscale": np.random.gamma(2.0, 1.0),
                "variance": np.random.gamma(2.0, 1.0),
                "period": np.random.uniform(0.5, 2.0),
            }

        elif kernel_name == "Linear":

            return {"variance": np.random.gamma(2.0, 1.0)}

        else:

            return {}

    def reweight_particles(self, x, y):

        total_weight = 0.0

        for i, particle in enumerate(self.particles):

            # Compute the marginal likelihood
            likelihood = self.compute_marginal_likelihood(particle.kernel, x, y)
            self.weights[i] *= likelihood
            total_weight += self.weights[i]

        # Normalize weights
        self.weights = [w / total_weight for w in self.weights]

    def compute_marginal_likelihood(self, kernel, x, y):

        # Compute the GP marginal likelihood for given kernel and data
        try:

            K = kernel(x, x) + 1e-3 * np.eye(len(x))  # Increased jitter
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_likelihood = -0.5 * np.dot(y, alpha)
            log_likelihood -= np.sum(np.log(np.diagonal(L)))
            log_likelihood -= len(x) / 2 * np.log(2 * np.pi)

            return np.exp(log_likelihood)

        except np.linalg.LinAlgError:

            # Handle numerical errors by returning a very small likelihood
            return 1e-10

    def resample_particles(self):

        indices = np.random.choice(
            range(self.num_particles), size=self.num_particles, p=self.weights
        )

        self.particles = [self.particles[i].copy() for i in indices]
        self.weights = [1.0 / self.num_particles] * self.num_particles

    def rejuvenate_particles(self, x, y):

        for particle in self.particles:

            # Apply MCMC moves to kernel structure and parameters
            self.mcmc_move(particle, x, y)

    def mcmc_move(self, particle, x, y):

        # Propose a new kernel by adding, deleting, or modifying
        new_kernel = self.propose_new_kernel(particle.kernel)
        
        # Compute acceptance probability
        current_likelihood = self.compute_marginal_likelihood(particle.kernel, x, y)
        new_likelihood = self.compute_marginal_likelihood(new_kernel, x, y)
        acceptance_ratio = min(1, new_likelihood / current_likelihood)

        if random.uniform(0, 1) < acceptance_ratio:

            # Accept the new kernel
            particle.kernel = new_kernel

    def propose_new_kernel(self, kernel):

        # Randomly choose an operation: add, multiply, or change parameters
        operations = ["add", "mul", "change"]
        
        # Limit operations based on current kernel depth
        current_depth = self.get_kernel_depth(kernel)

        if current_depth >= self.max_kernel_depth:

            operations = ["change"]  # Only allow parameter changes

        operation = random.choice(operations)

        if operation in ["add", "mul"]:

            # Combine with a new base kernel
            new_base_kernel = self.sample_initial_kernel()
            new_kernel = Kernel(
                name=None, left=kernel, right=new_base_kernel, operation=operation
            )

            return new_kernel

        elif operation == "change":

            # Modify parameters slightly, ensuring positivity
            new_kernel = deepcopy(kernel)
            params = new_kernel.get_params()

            for k_name in params:

                for param in params[k_name]:
                    
                    perturbation = np.random.normal(0, 0.1)
                    new_value = params[k_name][param] * np.exp(perturbation)
                    
                    # Ensure parameters remain positive
                    new_value = max(new_value, 1e-3)
                    params[k_name][param] = new_value

            new_kernel.set_params(params)

            return new_kernel

        else:

            return kernel

    def get_kernel_depth(self, kernel):

        if kernel.operation in ["add", "mul"]:

            return 1 + max(
                self.get_kernel_depth(kernel.left), self.get_kernel_depth(kernel.right)
            )

        else:

            return 1

    def run_smc(self, data_sequence):

        self.initialize_particles()

        for t, (x_t, y_t) in enumerate(data_sequence):

            print(f"Time step {t+1}/{len(data_sequence)}")

            # Reweight particles
            self.reweight_particles(x_t, y_t)

            # Compute Effective Sample Size (ESS)
            ess = 1.0 / sum(w**2 for w in self.weights)

            print(f"Effective Sample Size (ESS): {ess:.2f}")

            # Resample if ESS is below threshold
            if ess < self.num_particles / 2:

                print("Resampling particles...")
                self.resample_particles()

                # Rejuvenate particles
                for _ in range(self.num_rejuvenation_steps):

                    self.rejuvenate_particles(x_t, y_t)

            # Optional: normalize weights again

            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]


def partition_data(x, y, num_partitions):

    x_partitions = np.array_split(x, num_partitions)
    y_partitions = np.array_split(y, num_partitions)
    data_sequence = list(zip(x_partitions, y_partitions))

    return data_sequence


def plot_results_with_test(x_train, y_train, x_test, y_test, best_kernel):

    # Combine training and test data
    x_all = np.concatenate([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])

    # Sort data

    sorted_indices = np.argsort(x_all)
    x_all = x_all[sorted_indices]
    y_all = y_all[sorted_indices]

    # Create test points over the entire range
    x_pred = np.linspace(np.min(x_all), np.max(x_all), 500)

    # Compute covariance matrices
    K = best_kernel(x_train, x_train) + 1e-3 * np.eye(len(x_train))
    K_s = best_kernel(x_train, x_pred)
    K_ss = best_kernel(x_pred, x_pred) + 1e-3 * np.eye(len(x_pred))

    # Compute the mean and covariance of the GP posterior

    try:

        L = np.linalg.cholesky(K)

    except np.linalg.LinAlgError:

        print("Cholesky decomposition failed. Adding more jitter.")
        K += 1e-2 * np.eye(len(x_train))
        L = np.linalg.cholesky(K)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Predictive mean
    mu_s = K_s.T @ alpha

    # Predictive variance
    v = np.linalg.solve(L, K_s)
    var_s = np.diag(K_ss) - np.sum(v**2, axis=0)
    std_s = np.sqrt(var_s)

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(x_train, y_train, "kx", label="Training data")
    plt.plot(x_test, y_test, "ro", label="Test data")
    plt.plot(x_pred, mu_s, "b", label="Predictive mean")
    plt.fill_between(
        x_pred,
        mu_s - 2 * std_s,
        mu_s + 2 * std_s,
        color="blue",
        alpha=0.2,
        label="Confidence interval",
    )

    plt.legend()
    plt.title("GP Regression with Best Kernel")
    plt.xlabel("Time")
    plt.ylabel("Log(y)")
    plt.show()


def compute_gp_posterior(best_kernel, x_train, y_train, x_pred):

    K = best_kernel(x_train, x_train) + 1e-2 * np.eye(len(x_train))  # Increased jitter
    K_s = best_kernel(x_train, x_pred)
    K_ss = best_kernel(x_pred, x_pred) + 1e-2 * np.eye(len(x_pred))

    try:

        L = np.linalg.cholesky(K)

    except np.linalg.LinAlgError:

        # Handle non-positive definite matrix
        print("Cholesky decomposition failed. Adjusting jitter.")
        
        K += 1e-2 * np.eye(len(x_train))  # Increase jitter further
        L = np.linalg.cholesky(K)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Predictive mean
    mu_s = K_s.T @ alpha

    # Predictive variance
    v = np.linalg.solve(L, K_s)
    var_s = np.diag(K_ss) - np.sum(v**2, axis=0)
    std_s = np.sqrt(var_s)

    return mu_s, std_s



