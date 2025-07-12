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
    persistence = simplex_tree.persistence()
    gd.plot_persistence_barcode(persistence)

    # plt.figure(figsize=(6, 4))

    plt.xlabel("Normalized Weight Connection")
    plt.ylabel("Cavity order")

    plot_filename = os.path.join(plots_folder, f'persistence_plot_{index}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to prevent display

    df = pd.DataFrame({
      'betti_numbers': [simplex_tree.betti_numbers()], 
      'index_study' : [int(index)]
    })

    betti_df = pd.concat([betti_df, df]) 
    gc.collect()
    del simplex_tree

  return betti_df
