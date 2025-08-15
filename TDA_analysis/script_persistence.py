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
import gudhi as gd

# threshold = 0.3
# percentile_cut = 10 

# 1. Load the data:
data_file_path = '/content/SCmatrices_mammals.mat'  # Replace with your actual file path
data = sio.loadmat(data_file_path)

# Create a folder to save the plots
plots_folder = '/content/persistence_plots/'
os.makedirs(plots_folder, exist_ok=True)

def create_persistence(start_index, end_index, p_cut, betti_df=None):
  """
  Processes connectome data to compute cliques, generate plots, and update a DataFrame.

  Args:
    start_index: The starting index for processing.
    end_index: The ending index for processing.
    cliques_df: (Optional) Existing Pandas DataFrame to append results to.

  Returns:
    pandas.DataFrame: The updated DataFrame with clique data.
  """
  
  if betti_df is None:
    betti_df = pd.DataFrame(columns=['betti_numbers', 'index_study'])

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
    k_max = G_ig.clique_number()

    in_matrix = x
    # Convert to networkx graph
    G_nx = nx.from_numpy_array(in_matrix)

    # Convert to igraph for efficient clique finding
    G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)

    # Create a SimplexTree
    simplex_tree = gd.SimplexTree()

    # Insert vertices
    for i in range(in_matrix.shape[0]):
        simplex_tree.insert([i], filtration=0)

    # Find and insert cliques as higher-dimensional simplices
    max_clique_size = G_ig.clique_number()  # Get maximum clique size

    for k in range(2, max_clique_size + 1):  # Iterate through clique sizes
        cliques_of_size_k = G_ig.cliques(min=k, max=k)
        for clique in cliques_of_size_k:
            # Get the maximum weight of edges within the clique for filtration value
            max_weight = 0
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    weight = in_matrix[clique[i], clique[j]]
                    max_weight = max(max_weight, weight)
            
            simplex_tree.insert(clique, filtration=max_weight)  # Insert clique with filtration

    # Now you have a SimplexTree with higher-dimensional simplices.
    # You can compute persistence, Betti numbers, etc.
    # betti_colors = {
    #     0: "tab:blue",
    #     1: "tab:orange",
    #     2: "tab:green",
    #     3: "tab:red",
    #     4: "tab:purple",
    #     5: "tab:brown",
    #     6: "tab:pink",
    #     7: "tab:gray",
    #     8: "tab:olive",
    #     9: "tab:cyan",
    #     10: "gold"
    # }

    # persistence = simplex_tree.persistence()
    # max_dim = max(dim for dim, _ in persistence)

    # # Prepare vertical figure: barcode on top, Betti curves below
    # fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # # --- Custom barcode plot ---
    # for dim in range(max_dim + 1):
    #     intervals = simplex_tree.persistence_intervals_in_dimension(dim)
    #     color = betti_colors.get(dim, "black")
    #     for idx, (birth, death) in enumerate(intervals):
    #         death_plot = death if np.isfinite(death) else t_max + (t_max - t_min) * 0.05
    #         axes[0].hlines(y=dim + idx * 0.1, xmin=birth, xmax=death_plot,
    #                       colors=color, linewidth=2)
    # axes[0].set_ylabel("Homology dim")
    # axes[0].set_yticks(range(max_dim + 1))
    # axes[0].set_yticklabels([f"β{d}" for d in range(max_dim + 1)])

    # # --- Betti curves ---
    # # Get bounds for filtration axis
    # all_times = []
    # for dim in range(max_dim + 1):
    #     intervals = simplex_tree.persistence_intervals_in_dimension(dim)
    #     if len(intervals) > 0:
    #         all_times.extend(intervals.flatten())
    # all_times = np.array(all_times)
    # all_times = all_times[np.isfinite(all_times)]
    # t_min, t_max = (all_times.min(), all_times.max()) if all_times.size else (0.0, 1.0)
    # grid = np.linspace(t_min, t_max, 200)

    # for dim in range(max_dim + 1):
    #     intervals = simplex_tree.persistence_intervals_in_dimension(dim)
    #     betti_vals = [
    #         np.sum((intervals[:, 0] <= t) & (intervals[:, 1] > t))
    #         for t in grid
    #     ]
    #     axes[1].step(grid, betti_vals, where='post', color=betti_colors.get(dim, "black"), label=f"β{dim}")

    # axes[1].set_xlabel("Filtration value")
    # axes[1].set_ylabel("Betti number")
    # axes[1].legend()

    # plt.tight_layout()
    # plot_filename = os.path.join(plots_folder, f'persistence_and_betti_{index}.png')
    # plt.savefig(plot_filename)
    # plt.close()

    ##### SECOND WAY 
    
    persistence = simplex_tree.persistence()

    # code for persistence diagrams
    # colour is different for the 2 plots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- Top: persistence barcode ---
    gd.plot_persistence_barcode(persistence, axes=axes[0])
    axes[0].set_ylabel("Homology dim")
    axes[0].set_xlabel("")  # hide x-label for top plot

    # --- Bottom: Betti curves ---
    max_dim = max(dim for dim, _ in persistence)
    all_times = []
    for dim in range(max_dim + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(intervals) > 0:
            all_times.extend(intervals.flatten())
    all_times = np.array(all_times)
    all_times = all_times[np.isfinite(all_times)]
    t_min, t_max = (all_times.min(), all_times.max()) if all_times.size else (0.0, 1.0)
    grid = np.linspace(t_min, t_max, 200)

    for dim in range(max_dim + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        betti_vals = [
            np.sum((intervals[:, 0] <= t) & (intervals[:, 1] > t))
            for t in grid
        ]
        axes[1].step(grid, betti_vals, where='post', label=f"β{dim}")

    axes[1].set_xlabel("Filtration value")
    axes[1].set_ylabel("Betti number")
    axes[1].legend()

    plt.tight_layout()
    plot_filename = os.path.join(plots_folder, f'persistence_and_betti_{index}.png')
    plt.savefig(plot_filename)
    plt.close()

    #### First Code  
    
    # # gd.plot_persistence_barcode(persistence)

    # # # plt.figure(figsize=(6, 4))

    # # plt.xlabel("Normalized Weight Connection")
    # # plt.ylabel("Cavity order")

    # # plot_filename = os.path.join(plots_folder, f'persistence_plot_{index}.png')
    # # plt.savefig(plot_filename)
    # # plt.close()  # Close the figure to prevent display

    df = pd.DataFrame({
      'betti_numbers': [simplex_tree.betti_numbers()], 
      'index_study' : [int(index)]
    })

    betti_df = pd.concat([betti_df, df]) 
    gc.collect()
    del simplex_tree

  return betti_df



