import networkx as nx
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import gc

# threshold = 0.3
# percentile_cut = 10 
# 1. Load the data:
data_file_path = '/content/SCmatrices_mammals.mat'  # Replace with your actual file path
data = sio.loadmat(data_file_path)

# Create a folder to save the plots
plots_folder = '/content/clique_plots/'
os.makedirs(plots_folder, exist_ok=True)

def process_connectome_data(start_index, end_index,p_cut,  cliques_df=None):
  """
  Processes connectome data to compute cliques, generate plots, and update a DataFrame.

  Args:
    start_index: The starting index for processing.
    end_index: The ending index for processing.
    cliques_df: (Optional) Existing Pandas DataFrame to append results to.

  Returns:
    pandas.DataFrame: The updated DataFrame with clique data.
  """
  
  if cliques_df is None:
    cliques_df = pd.DataFrame(columns=['num_simplices', 'max_order_simplex', 'index_study'])

  matrices = data["conn_mat"] 

  for index in tqdm(range(start_index, end_index + 1), desc="Processing connectomes"):
    print("index ", index)
    x = matrices[:, :, index]
    x = x/x.mean()
    x_flat = x.flatten()
    x_flat = np.array(x_flat[x_flat > 0])
    threshold = np.percentile(x_flat, p_cut)
    x[x <= threshold] = 0
    G = nx.from_numpy_array(x)  

    G_ig = ig.Graph.TupleList(G.edges(), directed=False)
    # maxs = G_ig.largest_cliques()
    # k_max = np.shape(maxs)[1]
    k_max = G_ig.clique_number()

    if k_max < 18:
      all_cliques = list(tqdm(G_ig.cliques(), desc=f"Finding cliques for index {index}"))
      # Plotting for k_max < 18
      counts = Counter(len(clique) - 1 for clique in all_cliques)
      orders = sorted(counts.keys())
      counts_by_order = [counts[k] for k in orders]

      plt.figure(figsize=(6, 4))
      plt.plot(orders, counts_by_order, marker='o', linestyle='-')
      plt.xlabel("Order (k) of simplex")
      plt.ylabel("Number of simplices")
      plt.title("Clique Complex: Simplex Counts by Order, Graph : " + str(index))
      plt.xticks(orders)
      plt.grid(True)
            # Save the plot
      plot_filename = os.path.join(plots_folder, f'clique_plot_{index}.png')
      plt.savefig(plot_filename)
      plt.close()  # Close the figure to prevent display

      df = pd.DataFrame({
        'num_simplices': [np.array(counts_by_order)],
        'max_order_simplex': [len(counts_by_order)], 
        'index_study' : [int(index)]
      })

      cliques_df = pd.concat([cliques_df, df]) 
      gc.collect()
      del all_cliques     

    elif k_max < 22:
      # Modified else part for k_max > 17: Store only clique counts
      maxs = G_ig.largest_cliques()
      max_size = np.shape(maxs)[1]

      clique_counts = []  # Store clique counts as a list

      for k in tqdm(range(1, max_size + 1), desc="Processing clique sizes"):
        cliques_k = G_ig.cliques(min=k, max=k)
        num_cliques_of_size_k = len(cliques_k)
        clique_counts.append(num_cliques_of_size_k)
        print(f"Number of cliques of size {k}: {num_cliques_of_size_k}")
        del cliques_k
        gc.collect()

      # Plotting for k_max > 17
      orders = list(range(1, max_size + 1))  # or list(range(max_size))

      plt.figure(figsize=(6, 4))
      plt.plot(orders, clique_counts, marker='o', linestyle='-')
      plt.xlabel("Order (k) of simplex")
      plt.ylabel("Number of simplices")
      plt.title(f"Clique Complex: Simplex Counts by Order, Graph: {index} (k_max > 17)")
      plt.xticks(orders)  # Ensure ticks at each integer order
      plt.grid(True)

      # Save the plot
      plot_filename = os.path.join(plots_folder, f'clique_plot_{index}.png')
      plt.savefig(plot_filename)
      plt.close()

      # Update DataFrame (adapted to use clique_counts list)
      df = pd.DataFrame({
        'num_simplices': [np.array(clique_counts)],
        'max_order_simplex': [max_size],
        'index_study' : [int(index)]
      })

      cliques_df = pd.concat([cliques_df, df])
      gc.collect()
    
    else: 
      print("ORDER BIGGER THAN 22, skipped, ORDER :", k_max)
      continue

  return cliques_df
