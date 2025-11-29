#!/usr/bin/env python
# coding: utf-8

# # Análisis de Resultados - Tablas Resumen por Método
# 
# Este notebook presenta las tablas resumen de rendimiento para cada método de clasificación (KNN, SVM, RF, NB) con diferentes conjuntos de datos.

# In[13]:


import pandas as pd
import numpy as np
import os
import glob
from IPython.display import display, Markdown

methods = ['KNN', 'SVM', 'RF', 'NB']
norm_types = ['original', 'norm', 'std']
pca_variants = ['', '_PCA80', '_PCA95']

# Función para crear tabla resumen para un método específico
def create_summary_table(method):
    results_by_dataset = {}

    # Iterar sobre todos los tipos de normalización y variantes PCA
    for norm in norm_types:
        for pca in pca_variants:
            dataset_name = f"{norm}{pca}"

            # Leer los 5 archivos CSV correspondientes a los 5 folds
            fold_data = []
            for fold in range(1, 6):
                file_pattern = f"cross_validation_evaluation/{method}_{dataset_name}_{fold}.csv"
                if os.path.exists(file_pattern):
                    df = pd.read_csv(file_pattern)
                    # Convertir a diccionario para facilitar el acceso
                    metrics_dict = dict(zip(df['metric'], df['value']))
                    fold_data.append(metrics_dict)

            # Calcular media y desviación típica para cada métrica
            all_metrics = fold_data[0].keys()
            dataset_results = {}

            for metric in all_metrics:
                values = [fold[metric] for fold in fold_data if metric in fold]
                if values:
                    mean = np.mean(values)
                    std = np.std(values, ddof=1)  # Usar ddof=1 para desviación estándar muestral
                    # Formato LaTeX: media $\pm$ desviación
                    if np.isnan(std) or np.isclose(std, 0):
                        dataset_results[metric] = f"{mean:.4f}"
                    else:
                        dataset_results[metric] = f"{mean:.4f} $\\pm$ {std:.4f}"

            results_by_dataset[dataset_name] = dataset_results

    # Crear DataFrame con los resultados
    df_summary = pd.DataFrame(results_by_dataset).T

    # Ordenar columnas en un orden lógico
    desired_order = ['Exactitud', 'Precisión', 'Recall', 'F1-score', 'Sensibilidad', 
                    'Especificidad', 'Tasa de Falsos Positivos', 'Tasa de Falsos Negativos', 'AUC']

    column_order = [col for col in desired_order if col in df_summary.columns]
    df_summary = df_summary[column_order]

    csv_filename = f'cross_validation_results/{method}_summary_table.csv'
    df_summary.to_csv(csv_filename, index=True)

    latex_filename = f'cross_validation_results/{method}_summary_table.tex'
    latex_str = df_summary.to_latex(
        index=True,
        escape=False,
        caption=f'Tabla resumen del método {method}',
        label=f'tab:{method}_summary',
        column_format='l' + 'c' * len(df_summary.columns)
    )

    with open(latex_filename, 'w', encoding='utf-8') as f:
        f.write(latex_str)

    return df_summary


# ## Tabla Resumen - KNN
# 
# A continuación se presenta la tabla resumen del método **K-Nearest Neighbors (KNN)** con todos los conjuntos de datos considerados. Cada celda muestra el formato "media $\pm$ desviación típica" para cada métrica.

# In[14]:


knn_table = create_summary_table('KNN')

display(knn_table)


# ## Tabla Resumen - SVM
# 
# A continuación se presenta la tabla resumen del método **Support Vector Machine (SVM)** con todos los conjuntos de datos considerados.

# In[15]:


svm_table = create_summary_table('SVM')

display(svm_table)


# ## Tabla Resumen - Random Forest
# 
# A continuación se presenta la tabla resumen del método **Random Forest (RF)** con todos los conjuntos de datos considerados.

# In[16]:


rf_table = create_summary_table('RF')
if rf_table is not None:
    display(rf_table)
else:
    print("No se encontraron datos para RF")


# ## Tabla Resumen - Naive Bayes
# 
# A continuación se presenta la tabla resumen del método **Naive Bayes (NB)** con todos los conjuntos de datos considerados.

# In[17]:


nb_table = create_summary_table('NB')

display(nb_table)


# ## Tabla Resumen Global - F1-score
# 
# Tabla comparativa de F1-score con todos los métodos y conjuntos de datos.

# In[18]:


def create_f1_comparison_table():
    f1_results = {}

    for method in methods:
        method_f1 = {}

        for norm in norm_types:
            for pca in pca_variants:
                dataset_name = f"{norm}{pca}"

                fold_values = []
                for fold in range(1, 6):
                    file_pattern = f"cross_validation_evaluation/{method}_{dataset_name}_{fold}.csv"
                    if os.path.exists(file_pattern):
                        df = pd.read_csv(file_pattern)
                        f1_row = df[df['metric'] == 'F1-score']
                        if not f1_row.empty:
                            fold_values.append(f1_row['value'].values[0])

                if fold_values:
                    mean = np.mean(fold_values)
                    std = np.std(fold_values, ddof=1)
                    if np.isnan(std) or np.isclose(std, 0):
                        method_f1[dataset_name] = f"{mean:.4f}"
                    else:
                        method_f1[dataset_name] = f"{mean:.4f} $\\pm$ {std:.4f}"

        f1_results[method] = method_f1

    df_f1_comparison = pd.DataFrame(f1_results)

    os.makedirs('cross_validation_results', exist_ok=True)

    csv_filename = 'cross_validation_results/f1_comparison_table.csv'
    df_f1_comparison.to_csv(csv_filename, index=True)

    latex_filename = 'cross_validation_results/f1_comparison_table.tex'
    latex_str = df_f1_comparison.to_latex(
        index=True,
        escape=False,
        caption='Tabla comparativa F1-score por método y conjunto de datos',
        label='tab:f1_comparison',
        column_format='l' + 'c' * len(df_f1_comparison.columns)
    )

    with open(latex_filename, 'w', encoding='utf-8') as f:
        f.write(latex_str)

    return df_f1_comparison

f1_comparison = create_f1_comparison_table()
display(f1_comparison)

