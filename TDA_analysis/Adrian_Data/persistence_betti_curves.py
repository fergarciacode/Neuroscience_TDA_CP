import networkx as nx
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import gc
import gudhi as gd
from sklearn.metrics import auc # Para calcular el Área Bajo la Curva (AUC)

# --- 1. CONFIGURACIÓN INICIAL ---
# Define la ruta a tus datos y la carpeta de salida
data_file_path = '/content/SCmatrices_mammals.mat' # Reemplaza con tu ruta
plots_folder = '/content/persistence_plots/' 
os.makedirs(plots_folder, exist_ok=True)

# Carga los datos del archivo .mat
try:
    data = sio.loadmat(data_file_path)
    matrices = data["conn_mat"]
    print("Datos cargados correctamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {data_file_path}")
    # Salir o manejar el error como prefieras
    exit()

# --- 2. DEFINICIÓN DE LA FUNCIÓN PRINCIPAL ---

def create_persistence(start_index, end_index, metrics_df=None):
    """
    Procesa datos de conectomas para calcular la persistencia homológica 
    y extraer un conjunto de métricas topológicas resumidas.

    Args:
        start_index (int): Índice de la primera matriz a procesar.
        end_index (int): Índice de la última matriz a procesar.
        metrics_df (pd.DataFrame, optional): DataFrame existente para añadir resultados. 
                                            Si es None, se creará uno nuevo.

    Returns:
        pd.DataFrame: Un DataFrame de pandas con las métricas topológicas para cada conectoma.
    """
    if metrics_df is None:
        # DataFrame para almacenar todas las métricas extraídas
        metrics_df = pd.DataFrame(columns=[
            'index_study', 'AUC_b0', 'AUC_b1', 'AUC_b2', 
            'TP_b0', 'TP_b1', 'TP_b2', 'percolation_density',
            'peak_b1', 'peak_density_b1',
            'peak_b2', 'peak_density_b2' # Columnas para el pico de B2 añadidas
        ])

    for index in tqdm(range(start_index, end_index + 1), desc="Procesando conectomas"):
        print(f"Procesando índice: {index}")
        
        # --- Preprocesamiento de la matriz ---
        x = matrices[:, :, index]
        # Normalización por la media para hacer los pesos comparables
        if x.mean() > 0:
            x = x / x.mean()
        
        # --- Creación del SimplexTree (Método robusto compatible con versiones antiguas/nuevas de Gudhi) ---
        G_nx = nx.from_numpy_array(x)
        
        # 1. Crea un SimplexTree vacío
        simplex_tree = gd.SimplexTree()
        # 2. Inserta los nodos (vértices)
        for node in G_nx.nodes():
            simplex_tree.insert([node], filtration=0.0)
        # 3. Inserta las aristas con su peso de filtración NEGATIVO
        #    (para que conexiones fuertes [peso alto] aparezcan primero [filtración baja])
        for u, v, data in G_nx.edges(data=True):
            weight = data.get('weight', 0.0)
            simplex_tree.insert([u, v], filtration=-weight)
        # 4. Expande a símplices de mayor dimensión (cliques)
        max_dim_expansion = 5 # Límite para la construcción de cliques
        simplex_tree.expansion(max_dim_expansion)

        # --- Cálculo de la persistencia y curvas de Betti ---
        simplex_tree.persistence()
        
        all_times = []
        intervals_b0 = simplex_tree.persistence_intervals_in_dimension(0)
        # Excluimos el componente infinito de B0 para tener una escala de filtración más representativa
        finite_times_b0 = intervals_b0[np.isfinite(intervals_b0[:, 1])]
        if len(finite_times_b0) > 0:
            all_times.extend(finite_times_b0.flatten())

        for dim in range(1, max_dim_expansion + 1):
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            if len(intervals) > 0:
                all_times.extend(intervals.flatten())
        
        all_times = np.array(all_times)
        all_times = all_times[np.isfinite(all_times)]
        t_min, t_max = (all_times.min(), all_times.max()) if all_times.size > 0 else (0.0, 1.0)
        grid = np.linspace(t_min, t_max, 200)

        betti_curves = {}
        for dim in range(max_dim_expansion + 1):
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            betti_vals = [np.sum((intervals[:, 0] <= t) & (intervals[:, 1] > t)) for t in grid]
            betti_curves[dim] = np.array(betti_vals)

        # --- Extracción de Métricas Topológicas ---
        summary_metrics = {'index_study': int(index)}

        # 1. Área Bajo la Curva (AUC) para B0, B1, B2
        summary_metrics['AUC_b0'] = auc(grid, betti_curves.get(0, [0]))
        summary_metrics['AUC_b1'] = auc(grid, betti_curves.get(1, [0]))
        summary_metrics['AUC_b2'] = auc(grid, betti_curves.get(2, [0]))
        
        # 2. Persistencia Total (TP) para B0, B1, B2
        for dim in range(3):
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            death_vals = np.copy(intervals[:, 1])
            death_vals[np.isinf(death_vals)] = t_max
            lifespans = death_vals - intervals[:, 0]
            summary_metrics[f'TP_b{dim}'] = np.sum(lifespans)

        # 3. Densidad de Percolación (caída brusca de B0)
        b0_curve = betti_curves.get(0)
        if b0_curve is not None and len(b0_curve) > 1 and np.ptp(b0_curve) > 0:
            drop_index = np.argmin(np.diff(b0_curve)) 
            summary_metrics['percolation_density'] = grid[drop_index]
        else:
            summary_metrics['percolation_density'] = np.nan

        # 4. Pico de B1 y la densidad donde ocurre
        b1_curve = betti_curves.get(1)
        if b1_curve is not None and len(b1_curve) > 0 and b1_curve.max() > 0:
            peak_index = np.argmax(b1_curve)
            summary_metrics['peak_b1'] = b1_curve[peak_index]
            summary_metrics['peak_density_b1'] = grid[peak_index]
        else:
            summary_metrics['peak_b1'] = 0
            summary_metrics['peak_density_b1'] = np.nan

        # 5. Pico de B2 y la densidad donde ocurre
        b2_curve = betti_curves.get(2)
        if b2_curve is not None and len(b2_curve) > 0 and b2_curve.max() > 0:
            peak_index_b2 = np.argmax(b2_curve)
            summary_metrics['peak_b2'] = b2_curve[peak_index_b2]
            summary_metrics['peak_density_b2'] = grid[peak_index_b2]
        else:
            summary_metrics['peak_b2'] = 0
            summary_metrics['peak_density_b2'] = np.nan

        # Añadir las métricas al DataFrame
        df = pd.DataFrame([summary_metrics])
        metrics_df = pd.concat([metrics_df, df], ignore_index=True)
        
        # Liberar memoria
        gc.collect()
        del simplex_tree

    return metrics_df

# --- 3. EJECUCIÓN DEL ANÁLISIS ---
# Llama a la función para un rango de índices (ej. los primeros 10 conectomas)
# Nota: end_index es inclusivo, así que de 0 a 9 procesa 10 matrices.
final_metrics_df = create_persistence(start_index=0, end_index=9)

# Muestra los primeros resultados del DataFrame
print("\n--- Resultados Finales ---")
print(final_metrics_df.head())

# Opcional: Guardar los resultados en un archivo CSV
final_metrics_df.to_csv('/content/topological_metrics.csv', index=False)
print("\nResultados guardados en 'topological_metrics.csv'")
