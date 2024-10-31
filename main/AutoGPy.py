
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from sklearn.metrics import mean_squared_error
from scipy.stats import gamma, uniform
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit
from jax import lax
from typing import Dict, Optional, Tuple, List
from jax.tree_util import register_pytree_node_class


def safe_cholesky(K):
    jitter = 1e-6
    def try_cholesky(K):
        return jnp.linalg.cholesky(K)

    def add_jitter(K):
        return jnp.linalg.cholesky(K + jitter * jnp.eye(K.shape[0]))

    is_pd = jnp.all(jnp.linalg.eigvalsh(K) > 0)
    L = lax.cond(is_pd, try_cholesky, add_jitter, K)
    return L


@register_pytree_node_class
class Kernel:
    """
    Represents a kernel function for Gaussian Processes.

    Parameters
    ----------
    name : str
        The name of the base kernel ('SE', 'Periodic', 'Linear', etc.).
    params : dict, optional
        Parameters for the kernel function.
    left : Kernel, optional
        The left child kernel (for composite kernels).
    right : Kernel, optional
        The right child kernel (for composite kernels).
    operation : str, optional
        The operation to combine kernels ('add' or 'mul').
    """

    def __init__(
        self,
        name: Optional[str],
        params: Optional[Dict[str, float]] = None,
        left: Optional['Kernel'] = None,
        right: Optional['Kernel'] = None,
        operation: Optional[str] = None
    ):
        self.name = name
        self.params = params or {}
        self.left = left
        self.right = right
        self.operation = operation

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        if self.operation == "add":
            return self.left(x1, x2) + self.right(x1, x2)
        elif self.operation == "mul":
            return self.left(x1, x2) * self.right(x1, x2)
        else:
            return self.kernel_function(x1, x2)

    def kernel_function(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        x1 = jnp.asarray(x1)
        x2 = jnp.asarray(x2)
        if self.name == "SE":
            lengthscale = self.params.get("lengthscale", 1.0)
            variance = self.params.get("variance", 1.0)
            sqdist = (x1[:, None] - x2[None, :]) ** 2
            return variance * jnp.exp(-0.5 * sqdist / lengthscale**2)
        else:
            raise ValueError(f"Unknown kernel {self.name}")

    def get_params(self) -> Dict[str, Dict[str, float]]:
        if self.operation in ["add", "mul"]:
            left_params = self.left.get_params()
            right_params = self.right.get_params()
            return {**left_params, **right_params}
        else:
            return {self.name: self.params}

    def set_params(self, params: Dict[str, Dict[str, float]]):
        if self.operation in ["add", "mul"]:
            self.left.set_params(params)
            self.right.set_params(params)
        else:
            if self.name in params:
                self.params.update(params[self.name])

    def __str__(self) -> str:
        if self.operation == "add":
            return f"({self.left} + {self.right})"
        elif self.operation == "mul":
            return f"({self.left} * {self.right})"
        else:
            return self.name

    # Required methods for JAX pytrees
    def tree_flatten(self):
        children = (self.left, self.right, self.params)
        aux_data = (self.name, self.operation)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        name, operation = aux_data
        left, right, params = children
        return cls(name, params, left, right, operation)


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

    def initialize_particles(self, key):
        keys = random.split(key, self.num_particles)
        for i in range(self.num_particles):
            kernel = self.sample_initial_kernel(keys[i])
            particle = Particle(kernel)
            self.particles.append(particle)
            self.weights.append(1.0 / self.num_particles)

    def sample_initial_kernel(self, key):
        # Adjusted to use JAX random functions
        base_kernels = ["SE", "Periodic", "Linear"]
        operations = ["add", "mul"]
        key, subkey = random.split(key)
        depth = random.randint(subkey, (), 1, self.max_kernel_depth + 1)
        kernel = self.build_random_kernel(depth, base_kernels, operations, key)
        return kernel

    def build_random_kernel(self, depth, base_kernels, operations, key):
        if depth == 1:
            key, subkey = random.split(key)
            index = random.randint(subkey, (), 0, len(base_kernels))
            name = base_kernels[index]
            params = self.sample_params(name, key)
            return Kernel(name, params)
        else:
            key, op_key, left_key, right_key = random.split(key, 4)
            op_index = random.randint(op_key, (), 0, len(operations))
            operation = operations[op_index]
            left = self.build_random_kernel(
                depth - 1, base_kernels, operations, left_key)
            right = self.build_random_kernel(
                depth - 1, base_kernels, operations, right_key)
            return Kernel(None, None, left, right, operation)

    def sample_params(self, key, kernel_name):
        if kernel_name == "SE":
            key, subkey1, subkey2 = random.split(key, 3)
            return {
                "lengthscale": gamma.sample(2.0, 1.0, subkey1),
                "variance": gamma.sample(2.0, 1.0, subkey2),
            }

    def reweight_particles(self, x, y):
        log_weights = []
        for i, particle in enumerate(self.particles):
            log_likelihood = self.compute_log_marginal_likelihood(
                particle.kernel, x, y)
            log_weight = log_likelihood  # Since initial weights are uniform in log-space
            log_weights.append(log_weight)

        # Normalize log-weights to prevent underflow
        max_log_weight = np.max(log_weights)
        normalized_log_weights = [lw - max_log_weight for lw in log_weights]
        weights = np.exp(normalized_log_weights)
        total_weight = np.sum(weights)
        self.weights = weights / total_weight  # Normalize to sum to 1

    def compute_log_marginal_likelihood(self, kernel, x, y):
        K = kernel(x, x) + 1e-6 * jnp.eye(len(x))
        L = safe_cholesky(K)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
        log_likelihood = -0.5 * jnp.dot(y, alpha)
        log_likelihood -= jnp.sum(jnp.log(jnp.diagonal(L)))
        log_likelihood -= len(x) / 2 * jnp.log(2 * jnp.pi)
        return log_likelihood

    def resample_particles(self, key):
        cumsum_weights = jnp.cumsum(self.weights)
        indices = jnp.searchsorted(cumsum_weights, random.uniform(key, shape=(self.num_particles,)))
        self.particles = [self.particles[i].copy() for i in indices]
        self.weights = jnp.ones(self.num_particles) / self.num_particles

    def rejuvenate_particles(self, x, y, key):
        for i, particle in enumerate(self.particles):
            # Apply MCMC moves to kernel structure and parameters
            for _ in range(self.num_rejuvenation_steps):
                key, subkey = random.split(key)
                self.mcmc_move(particle, x, y, subkey)

    def mcmc_move(self, particle, x, y):
        # Choose between SUBTREE-REPLACE and DETACH-ATTACH moves
        move_type = random.choice(["subtree_replace", "detach_attach"])

        if move_type == "subtree_replace":
            new_kernel, log_q_forward, log_q_backward = self.subtree_replace_move(
                particle.kernel)
        else:
            new_kernel, log_q_forward, log_q_backward = self.detach_attach_move(
                particle.kernel)

        # Compute prior probabilities
        log_p_current = self.compute_log_prior(particle.kernel)
        log_p_new = self.compute_log_prior(new_kernel)

        # Compute log-likelihoods
        current_log_likelihood = self.compute_log_marginal_likelihood(
            particle.kernel, x, y)
        new_log_likelihood = self.compute_log_marginal_likelihood(
            new_kernel, x, y)

        # Compute acceptance probability in log-space
        log_acceptance_ratio = (
            new_log_likelihood + log_p_new + log_q_backward
            - (current_log_likelihood + log_p_current + log_q_forward)
        )

        # Convert log-acceptance ratio to probability
        acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))

        if random.uniform(key) < acceptance_probability:
            # Accept the new kernel
            particle.kernel = new_kernel
            # Update parameters using HMC
            particle.kernel = self.hmc_update(particle.kernel, x, y, key)

    def compute_log_prior(self, kernel):
        # Recursively compute the log prior probability of the kernel structure and parameters
        if kernel.operation in ["add", "mul"]:
            log_p_left = self.compute_log_prior(kernel.left)
            log_p_right = self.compute_log_prior(kernel.right)
            # Assume uniform prior over operations
            log_p_operation = np.log(0.5)
            return log_p_left + log_p_right + log_p_operation
        else:
            # Base kernel prior
            log_p_structure = np.log(1.0 / 3.0)  # Assuming three base kernels
            # Parameter priors
            log_p_params = 0.0
            for param_name, value in kernel.params.items():
                if param_name == "lengthscale" or param_name == "variance":
                    log_p_params += gamma.logpdf(value, 2.0, scale=1.0)
                elif param_name == "period":
                    log_p_params += uniform.logpdf(value, 0.5, 1.5)
            return log_p_structure + log_p_params

    def subtree_replace_move(self, kernel, key):
        # Implement SUBTREE-REPLACE move
        # Select a random subtree to replace

        paths = self.get_subtree_paths(kernel)
        key, subkey = random.split(key)
        selected_index = random.choice(subkey, len(paths))
        selected_path = paths[selected_index]

        # Extract the subtree at the selected path
        subtree = self.get_subtree(kernel, selected_path)

        # Propose a new subtree from the prior
        new_subtree = self.sample_initial_kernel()

        # Build the new kernel by replacing the subtree
        new_kernel = self.replace_subtree(kernel, selected_path, new_subtree)

        # Compute proposal probabilities (assuming symmetric)
        log_q_forward = 0.0  # Since proposals are from the prior
        log_q_backward = 0.0  # Symmetric proposal

        return new_kernel, log_q_forward, log_q_backward

    def detach_attach_move(self, kernel):
        # Implement DETACH-ATTACH move
        # For simplicity, we'll treat this similar to SUBTREE-REPLACE but can be expanded
        return self.subtree_replace_move(kernel)

    def get_subtree_paths(self, kernel, path=()):
        # Recursively get all possible paths to subtrees
        paths = [path]
        if kernel.operation in ["add", "mul"]:
            paths += self.get_subtree_paths(kernel.left, path + ("left",))
            paths += self.get_subtree_paths(kernel.right, path + ("right",))
        return paths

    def get_subtree(self, kernel, path):
        # Navigate the tree according to the path
        for p in path:
            if p == "left":
                kernel = kernel.left
            elif p == "right":
                kernel = kernel.right
        return kernel

    def replace_subtree(self, kernel, path, new_subtree):
        # Recursively rebuild the kernel with the new subtree
        if not path:
            return new_subtree
        else:
            direction = path[0]
            if direction == "left":
                left = self.replace_subtree(kernel.left, path[1:], new_subtree)
                return Kernel(
                    name=None,
                    left=left,
                    right=kernel.right,
                    operation=kernel.operation,
                )
            elif direction == "right":
                right = self.replace_subtree(
                    kernel.right, path[1:], new_subtree)
                return Kernel(
                    name=None,
                    left=kernel.left,
                    right=right,
                    operation=kernel.operation,
                )
            else:
                return kernel

    def hmc_update(self, kernel, x, y, key):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        # Update the parameters using HMC

        def model():
            params = {}
            if kernel.operation in ["add", "mul"]:
                left_kernel = self.hmc_update(kernel.left, x, y)
                right_kernel = self.hmc_update(kernel.right, x, y)
                return Kernel(
                    name=None,
                    left=left_kernel,
                    right=right_kernel,
                    operation=kernel.operation,
                )
            else:
                # Sample parameters using their priors
                if kernel.name == "SE":
                    lengthscale = numpyro.sample(
                        "lengthscale", dist.Gamma(2.0, 1.0)
                    )
                    variance = numpyro.sample("variance", dist.Gamma(2.0, 1.0))
                    params = {"lengthscale": lengthscale, "variance": variance}
                elif kernel.name == "Periodic":
                    lengthscale = numpyro.sample(
                        "lengthscale", dist.Gamma(2.0, 1.0)
                    )
                    variance = numpyro.sample("variance", dist.Gamma(2.0, 1.0))
                    period = numpyro.sample("period", dist.Uniform(0.5, 2.0))
                    params = {
                        "lengthscale": lengthscale,
                        "variance": variance,
                        "period": period,
                    }
                elif kernel.name == "Linear":
                    variance = numpyro.sample("variance", dist.Gamma(2.0, 1.0))
                    params = {"variance": variance}
                else:
                    params = kernel.params

                # Compute the GP likelihood
                updated_kernel = Kernel(kernel.name, params)
                K = updated_kernel(x, x) + 1e-6 * np.eye(len(x))
                numpyro.sample(
                    "y",
                    dist.MultivariateNormal(
                        loc=np.zeros(len(x)), covariance_matrix=K),
                    obs=y,
                )
                return updated_kernel

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=200, num_samples=100)
        mcmc.run(key)
        samples = mcmc.get_samples()
        updated_kernel = self.extract_parameters_from_samples(kernel, samples)
        return updated_kernel

    def extract_parameters_from_samples(self, kernel, samples):
        if kernel.operation in ["add", "mul"]:
            left_kernel = self.extract_parameters_from_samples(
                kernel.left, samples)
            right_kernel = self.extract_parameters_from_samples(
                kernel.right, samples)
            return Kernel(
                name=None,
                left=left_kernel,
                right=right_kernel,
                operation=kernel.operation,
            )
        else:
            params = {k: float(v.mean()) for k, v in samples.items() if k in kernel.params}

            return Kernel(kernel.name, params)

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
            self.rejuvenate_particles(x_t, y_t)

            # Normalize weights after rejuvenation
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
    x_pred = jnp.linspace(jnp.min(x_all), jnp.max(x_all), 500)


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

    K = best_kernel(x_train, x_train) + 1e-2 * \
        np.eye(len(x_train))  # Increased jitter
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


def predict_with_particles(self, x_train, y_train, x_test):
    mu_all = []
    for particle, weight in zip(self.particles, self.weights):
        mu, _ = compute_gp_posterior(particle.kernel, x_train, y_train, x_test)
        mu_all.append(weight * mu)
    mu_final = np.sum(mu_all, axis=0)
    return mu_final
