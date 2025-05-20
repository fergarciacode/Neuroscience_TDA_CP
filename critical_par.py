import numpy as np 
import scipy.io as sio
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

data_file_path = '/content/SCmatrices_mammals.mat'  # Replace with your actual file path
data = sio.loadmat(data_file_path)

def simulate_w_cp_fast(adj_matrix, lambda_val, max_time=100, initial_active_ratio=0.1):
    N = adj_matrix.shape[0]
    state = np.zeros(N, dtype=int)
    state[np.random.choice(N, int(initial_active_ratio * N), replace=False)] = 1

    t = 0
    times = [0]
    active_counts = [np.sum(state)]

    while t < max_time and np.any(state):
        active = state == 1
        inactive = state == 0

        events = []
        rates = []

        # Recovery events
        active_indices = np.where(active)[0]
        events += [('recover', i) for i in active_indices]
        rates += [1.0] * len(active_indices)

        # Infection events
        inactive_indices = np.where(inactive)[0]
        weights_matrix = adj_matrix[inactive_indices]
        total_weights = weights_matrix.sum(axis=1)
        active_weights = weights_matrix @ state

        mask = (total_weights > 0) & (active_weights > 0)
        targets = inactive_indices[mask]
        infect_rates = lambda_val * (active_weights[mask] / total_weights[mask])

        events += [('infect', i) for i in targets]
        rates += infect_rates.tolist()

        if not rates:
            break

        rates = np.array(rates)
        R = rates.sum()
        t += np.random.exponential(1.0 / R)
        times.append(t)

        chosen_index = np.random.choice(len(events), p=rates / R)
        action, node = events[chosen_index]

        state[node] = 0 if action == 'recover' else 1
        active_counts.append(np.sum(state))

    return times, active_counts, np.any(state)

def trial_survives_weighted(args):
    adj_matrix, lambda_val, max_time, init_ratio = args
    _, active_counts, _ = simulate_w_cp_fast(adj_matrix, lambda_val, max_time, init_ratio)
    return active_counts[-1] > 0

from concurrent.futures import ProcessPoolExecutor

def find_lambda_critical_w_par(adj_matrix, lambda_low, lambda_high, tolerance=0.01, max_time=100, trials=12):
    def is_supercritical(lambda_val):
        # Generate random initial ratios outside the worker
        init_ratios = np.random.uniform(0.1, 0.8, size=trials)
        args_list = [(adj_matrix, lambda_val, max_time, r) for r in init_ratios]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(trial_survives_weighted, args_list))

        return sum(results) / trials > 0.5

    while abs(lambda_high - lambda_low) > tolerance:
        lambda_mid = (lambda_low + lambda_high) / 2.0
        if is_supercritical(lambda_mid):
            lambda_high = lambda_mid
        else:
            lambda_low = lambda_mid

    return (lambda_low + lambda_high) / 2.0

def compute_critvalues(start_index, end_index, tol = 0.005):
  """
  Processes connectome data to compute cliques, generate plots, and update a DataFrame.

  Args:
    start_index: The starting index for processing.
    end_index: The ending index for processing.

  Returns:
    numpy array with the critical values
  """
  crits = [tol]
   
  matrices = data["conn_mat"] 

  for index in tqdm(range(start_index, end_index + 1), desc="Processing connectomes"):
    # print("index ", index)
    x = matrices[:, :, index]
    x = x/x.mean()
    crit_v = find_lambda_critical_w_par(x, lambda_low=1.0, lambda_high=2.0, tolerance=tol)
    crits.append(crit_v)

  crits = np.array(crits)
  return crits