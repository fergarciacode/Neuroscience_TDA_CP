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
from scipy.special import comb # Importante: para el coeficiente binomial

# ... (el resto de tus importaciones y la carga de datos se mantienen igual)
data_file_path = '/content/SCmatrices_mammals.mat' 
data = sio.loadmat(data_file_path) 
plots_folder = '/content/clique_plots/' 
os.makedirs(plots_folder, exist_ok=True)


def process_connectome_data(start_index, end_index, p_cut, cliques_df=None):
    """
    Processes connectome data to compute NORMALIZED cliques, generate plots, and update a DataFrame.
    """
    if cliques_df is None:
        # Añadimos una columna para los conteos normalizados
        cliques_df = pd.DataFrame(columns=['num_simplices', 'normalized_simplices', 'max_order_simplex', 'index_study'])

    matrices = data["conn_mat"]
    for index in tqdm(range(start_index, end_index + 1), desc="Processing connectomes"):
        print("index ", index)
        x = matrices[:, :, index]
        x = x / x.mean()
        x_flat = x.flatten()
        x_flat = np.array(x_flat[x_flat > 0])
        threshold = np.percentile(x_flat, p_cut)
        x[x <= threshold] = 0
        
        G = nx.from_numpy_array(x)
        num_nodes = G.number_of_nodes() # Obtenemos el número de nodos 'n'
        
        G_ig = ig.Graph.TupleList(G.edges(), directed=False)
        k_max = G_ig.clique_number()

        # --- Lógica principal para contar y normalizar ---
        
        # Obtenemos los conteos de símplices (cliques de tamaño k+1)
        all_cliques = list(G_ig.cliques(min=1)) 
        counts = Counter(len(clique) - 1 for clique in all_cliques)
        orders = sorted(counts.keys())
        counts_by_order = [counts[k] for k in orders]
        
        # --- NUEVA SECCIÓN: Normalización del conteo ---
        normalized_counts = []
        for k, count in zip(orders, counts_by_order):
            # 'k' es el orden del símplice (k=0 son nodos, k=1 son aristas, etc.)
            # El tamaño del clique es k+1
            # Calculamos (n choose k+1)
            max_possible_simplices = comb(num_nodes, k + 1, exact=True)
            if max_possible_simplices > 0:
                normalized_counts.append(count / max_possible_simplices)
            else:
                normalized_counts.append(0.0)
        # --- FIN DE LA NUEVA SECCIÓN ---

        # Graficar los conteos NORMALIZADOS
        plt.figure(figsize=(6, 4))
        plt.plot(orders, normalized_counts, marker='o', linestyle='-') # Usamos los datos normalizados
        plt.xlabel("Order (k) of simplex")
        plt.ylabel("Normalized Number of simplices") # Etiqueta del eje Y actualizada
        plt.title(f"Normalized Simplex Counts by Order, Graph: {index}")
        plt.xticks(orders)
        plt.grid(True)
        
        plot_filename = os.path.join(plots_folder, f'clique_plot_normalized_{index}.png')
        plt.savefig(plot_filename)
        plt.close()

        # Actualizar el DataFrame con ambas métricas
        df = pd.DataFrame({
            'num_simplices': [np.array(counts_by_order)],
            'normalized_simplices': [np.array(normalized_counts)], # Nueva columna
            'max_order_simplex': [len(counts_by_order)],
            'index_study': [int(index)]
        })
        cliques_df = pd.concat([cliques_df, df], ignore_index=True)
        
        gc.collect()
        del all_cliques

    return cliques_df
