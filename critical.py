import numpy as np 
import scipy.io as sio
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

data_file_path = '/content/SCmatrices_mammals.mat'  # Replace with your actual file path
data = sio.loadmat(data_file_path)

def simulate_weighted_contact_process(adj_matrix, lambda_val, max_time=100, initial_active_ratio=0.1):
    """
    Simulates the weighted Contact Process using Gillespie's algorithm.

    Parameters:
    - adj_matrix: 2D numpy array (float weights between 0 and 1)
    - lambda_val: infection rate λ
    - max_time: simulation end time
    - initial_active_ratio: fraction of initially active nodes

    Returns:
    - times: list of times at which events happened
    - active_counts: number of active nodes at each time
    """


    # Añadir normalizacion /np.mean()
    N = adj_matrix.shape[0]
    #state is a list of the active nodes
    state = np.zeros(N, dtype=int)

    # Initial active nodes
    initially_active = random.sample(range(N), int(initial_active_ratio * N))
    state[initially_active] = 1

    t = 0
    times = [0]
    active_counts = [np.sum(state)]

    while t < max_time and np.any(state):
        events = []
        rates = []

        for i in range(N):
            if state[i] == 1:
                # Recovery event (rate = 1)
                events.append(('recover', i))
                rates.append(1.0)
            else:
                # Infection rate depends on active neighbors & weights
                weights = adj_matrix[i]
                total_weight = np.sum(weights)
                if total_weight == 0:
                    continue  # no neighbors
                active_weight = np.dot(weights, state)
                if active_weight > 0:
                    rate = lambda_val * (active_weight / total_weight)
                    events.append(('infect', i))
                    rates.append(rate)

        if not rates:
            break

        R = sum(rates)
        delta_t = np.random.exponential(1.0 / R)
        t += delta_t
        times.append(t)

        event = random.choices(events, weights=rates, k=1)[0]
        action, node = event

        if action == 'recover':
            state[node] = 0
        elif action == 'infect':
            state[node] = 1

        active_counts.append(np.sum(state))

    return times, active_counts, np.any(state)

def find_lambda_critical_w(adj_matrix, lambda_low, lambda_high, tolerance=0.01, max_time=100, trials = 5):
    """
    Estimates the critical lambda using a binary search method.

    Parameters:
    - adj_matrix: numpy adjacency matrix
    - lambda_low: known subcritical lambda
    - lambda_high: known supercritical lambda
    - tolerance: stop when high - low < tolerance
    - max_time: max simulation time
    - trials: number of runs per lambda guess

    Returns:
    - lambda_c: estimated critical lambda
    """
    def is_supercritical(lambda_val):
        survived = 0
        for _ in range(trials):
            _, active_counts, aa = simulate_weighted_contact_process(adj_matrix, lambda_val, max_time)
            if active_counts[-1] > 0:
                survived += 1
        return survived / trials > 0.5

    while abs(lambda_high - lambda_low) > tolerance:
        lambda_mid = (lambda_low + lambda_high) / 2.0
        if is_supercritical(lambda_mid):
            lambda_high = lambda_mid
        else:
            lambda_low = lambda_mid

    return (lambda_low + lambda_high) / 2.0

def compute_critvalues(start_index, end_index, tol = 0.01):
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
    print("index ", index)
    x = matrices[:, :, index]
    x = x/x.mean()
    crit_v = find_lambda_critical_w(x, lambda_low=1.0, lambda_high=2.0, tolerance=tol)
    crits.append(crit_v)

  crits = np.array(crits)
  return crits