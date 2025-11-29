#!/usr/bin/env python
# coding: utf-8

# # Ensemble Methods
# 
# Este notebook implementa diferentes métodos de ensemble para combinar las predicciones de múltiples modelos de machine learning y mejorar el rendimiento predictivo.
# 
# Los métodos implementados son:
# 1. **Votación (Voting)**: La clase predicha es la que obtiene más votos de los modelos individuales
# 2. **Media (Mean)**: Combina las probabilidades predichas usando la media
# 3. **Mediana (Median)**: Combina las probabilidades predichas usando la mediana

# ## 1. Método de Votación (Voting Ensemble)
# 
# El método de votación combina las predicciones de múltiples modelos mediante un sistema de votos. La clase que recibe más votos es seleccionada como la predicción final del ensemble.
# 
# **Características:**
# - **Votación mayoritaria**: La clase más votada es la predicción final
# - **Manejo de empates**: En caso de empate, se utilizan las probabilidades predichas como ponderación
# - **Ponderación por confianza**: Cada voto se pondera por la probabilidad que el modelo asignó a esa clase
# 
# **Parámetros:**
# - `predictions_list`: Lista de arrays numpy con las predicciones de cada modelo (valores de clase)
# - `probabilities_list`: Lista de arrays numpy con las probabilidades predichas por cada modelo (opcional, usado para resolver empates)
# 
# **Retorna:**
# - Array numpy con las predicciones finales del ensemble

# In[1]:


import numpy as np
from collections import Counter

def voting_ensemble(predictions_list: list, probabilities_list: list = None) -> np.ndarray:
    """
    Implementa un ensemble de votación que combina las predicciones de múltiples modelos.

    Args:
        predictions_list: Lista de arrays numpy con las predicciones de cada modelo.
                         Cada array tiene shape (n_samples,) con las clases predichas.
        probabilities_list: Lista opcional de arrays numpy con las probabilidades predichas.
                           Cada array tiene shape (n_samples, n_classes).
                           Se utiliza para ponderar en caso de empates.

    Returns:
        Array numpy con las predicciones finales del ensemble (shape: n_samples,)

    Ejemplo:
        >>> pred1 = np.array([0, 1, 1, 0])
        >>> pred2 = np.array([0, 1, 0, 0])
        >>> pred3 = np.array([1, 1, 0, 0])
        >>> predictions = [pred1, pred2, pred3]
        >>> voting_ensemble(predictions)
        array([0, 1, 0, 0])  # [0:2 votos, 1:2 votos (empate), 0:2 votos, 0:3 votos]
    """

    # Verificar que hay al menos una predicción
    if not predictions_list or len(predictions_list) == 0:
        raise ValueError("predictions_list no puede estar vacía")

    # Convertir todas las predicciones a arrays numpy
    predictions_array = np.array(predictions_list)  # shape: (n_models, n_samples)
    n_models, n_samples = predictions_array.shape

    # Array para almacenar las predicciones finales
    final_predictions = np.zeros(n_samples, dtype=int)

    # Para cada muestra, contar los votos
    for i in range(n_samples):
        # Obtener las predicciones de todos los modelos para esta muestra
        sample_predictions = predictions_array[:, i]

        # Contar los votos para cada clase
        vote_counts = Counter(sample_predictions)

        # Obtener el número máximo de votos
        max_votes = max(vote_counts.values())

        # Obtener las clases que tienen el máximo de votos
        tied_classes = [cls for cls, count in vote_counts.items() if count == max_votes]

        # Si hay empate y tenemos probabilidades, usar ponderación
        if len(tied_classes) > 1 and probabilities_list is not None:
            # Calcular la suma de probabilidades ponderadas para cada clase empatada
            weighted_probs = {}

            for cls in tied_classes:
                total_prob = 0.0

                # Sumar las probabilidades de los modelos que votaron por esta clase
                for model_idx in range(n_models):
                    if predictions_array[model_idx, i] == cls:
                        # Obtener la probabilidad que el modelo asignó a esta clase
                        prob = probabilities_list[model_idx][i, int(cls)]
                        total_prob += prob

                weighted_probs[cls] = total_prob

            # Seleccionar la clase con mayor suma de probabilidades
            final_predictions[i] = max(weighted_probs.items(), key=lambda x: x[1])[0]

        else:
            # Si no hay empate o no hay probabilidades, tomar la clase más votada
            # En caso de empate sin probabilidades, Counter devuelve una arbitrariamente
            final_predictions[i] = vote_counts.most_common(1)[0][0]

    return final_predictions


# ## 2. Método de Media (Mean Ensemble)
# 
# El método de media combina las probabilidades predichas por múltiples modelos calculando su promedio.
# 
# **Características:**
# - Combina las probabilidades de cada clase
# - La clase con mayor probabilidad promedio es seleccionada
# 
# (Por implementar)

# In[2]:


def mean_ensemble(probabilities_list: list) -> np.ndarray:
    probabilities_array = np.array(probabilities_list)
    mean_probabilities = np.mean(probabilities_array, axis=0)
    final_predictions = np.argmax(mean_probabilities, axis=1)
    return final_predictions


# ## 3. Método de Mediana (Median Ensemble)
# 
# El método de mediana combina las probabilidades predichas por múltiples modelos calculando su mediana.
# 
# **Características:**
# - Más robusto ante valores atípicos que la media
# - Combina las probabilidades de cada clase usando la mediana
# - La clase con mayor probabilidad mediana es seleccionada
# 
# (Por implementar)

# In[3]:


def median_ensemble(probabilities_list: list) -> np.ndarray:
    """
    Implementa un ensemble basado en la mediana de las probabilidades predichas.

    Args:
        probabilities_list: Lista de arrays numpy con las probabilidades predichas.
                           Cada array tiene shape (n_samples, n_classes).

    Returns:
        Array numpy con las predicciones finales del ensemble (shape: n_samples,)
    """
    # TODO: Implementar
    pass


# ## Ejemplo de uso del Voting Ensemble
# 
# A continuación se muestra un ejemplo simple de cómo funciona el método de votación con datos simulados.

# In[4]:


# Ejemplo 1: Caso simple sin empates
print("=" * 60)
print("EJEMPLO 1: Votación sin empates")
print("=" * 60)

# Simulamos predicciones de 4 modelos (KNN, SVM, NB, RF) para 5 muestras
pred_knn = np.array([0, 1, 1, 0, 2])
pred_svm = np.array([0, 1, 0, 0, 2])
pred_nb = np.array([1, 1, 0, 0, 1])
pred_rf = np.array([0, 1, 0, 1, 2])

predictions = [pred_knn, pred_svm, pred_nb, pred_rf]

print("\nPredicciones de cada modelo:")
print(f"KNN:  {pred_knn}")
print(f"SVM:  {pred_svm}")
print(f"NB:   {pred_nb}")
print(f"RF:   {pred_rf}")

result = voting_ensemble(predictions)
print(f"\nPredicción del ensemble: {result}")
print("\nAnálisis por muestra:")
for i in range(len(result)):
    votes = [p[i] for p in predictions]
    print(f"  Muestra {i}: votos={votes} → clase {result[i]}")

# Ejemplo 2: Caso con empates y ponderación
print("\n" + "=" * 60)
print("EJEMPLO 2: Votación con empates y ponderación")
print("=" * 60)

# Predicciones con empate en algunas muestras
pred_knn = np.array([0, 1])
pred_svm = np.array([1, 0])
pred_nb = np.array([0, 1])
pred_rf = np.array([1, 0])

predictions = [pred_knn, pred_svm, pred_nb, pred_rf]

# Probabilidades para resolver empates
# Cada modelo tiene probabilidades para 2 clases (0 y 1)
prob_knn = np.array([[0.9, 0.1], [0.2, 0.8]])  # muestra 0: 90% clase 0, muestra 1: 80% clase 1
prob_svm = np.array([[0.4, 0.6], [0.7, 0.3]])  # muestra 0: 60% clase 1, muestra 1: 70% clase 0
prob_nb = np.array([[0.8, 0.2], [0.3, 0.7]])   # muestra 0: 80% clase 0, muestra 1: 70% clase 1
prob_rf = np.array([[0.3, 0.7], [0.6, 0.4]])   # muestra 0: 70% clase 1, muestra 1: 60% clase 0

probabilities = [prob_knn, prob_svm, prob_nb, prob_rf]

print("\nPredicciones de cada modelo:")
print(f"KNN:  {pred_knn}")
print(f"SVM:  {pred_svm}")
print(f"NB:   {pred_nb}")
print(f"RF:   {pred_rf}")

print("\nProbabilidades de cada modelo:")
for i, (name, prob) in enumerate(zip(['KNN', 'SVM', 'NB', 'RF'], probabilities)):
    print(f"{name}: {prob}")

result_weighted = voting_ensemble(predictions, probabilities)
print(f"\nPredicción del ensemble (con ponderación): {result_weighted}")

print("\nAnálisis por muestra:")
for i in range(len(result_weighted)):
    votes = [p[i] for p in predictions]
    print(f"  Muestra {i}: votos={votes} (empate 2-2)")

    # Mostrar suma de probabilidades para cada clase
    for cls in [0, 1]:
        prob_sum = sum([probabilities[j][i, cls] for j in range(len(predictions)) if predictions[j][i] == cls])
        print(f"    Clase {cls}: suma de probabilidades = {prob_sum:.2f}")

    print(f"    → Predicción final: clase {result_weighted[i]}")


# ## Prueba del Ensemble con Modelos Pre-entrenados
# 
# A continuación, probamos el método de votación con los modelos reales que ya han sido entrenados, sin necesidad de reentrenarlos.

# In[5]:


import joblib
import pandas as pd
import os
import sys

# Importar las funciones de evaluación
sys.path.append('/workspaces/EnsembleAlgorithms')
from eval import evaluate_model

transformations = ["norm", "original", "std"]
PCA_values = [0, 80, 95]

print("="*80)
print("ENSEMBLE - MÉTODO DE VOTACIÓN CON MODELOS PRE-ENTRENADOS")
print("="*80)

# Aplicar ensemble de votación para cada combinación de transformación y PCA
for iteration in range(1, 6):
    for transform in transformations:
        for pca_value in PCA_values:
            print(f"\n--- Iteración {iteration} - Transformación: {transform} - PCA: {pca_value} ---")
            
            # Construir nombres de los archivos de datos y modelos
            if pca_value > 0:
                test_file = f"cross_validation_data/test{iteration}_{transform}_PCA{pca_value}.csv"
                model_suffix = f"{transform}_PCA{pca_value}_{iteration}"
            else:
                test_file = f"cross_validation_data/test{iteration}_{transform}.csv"
                model_suffix = f"{transform}_{iteration}"
            
            # Verificar que existe el archivo de test
            if not os.path.exists(test_file):
                print(f"  Archivo de test no encontrado: {test_file}")
                continue
            
            # Cargar datos de test
            df_test = pd.read_csv(test_file)
            X_test = df_test.iloc[:, :-1].astype(float)
            y_test = df_test.iloc[:, -1].astype(int).to_numpy()
            
            # Listas para almacenar predicciones y probabilidades de cada modelo
            predictions_list = []
            probabilities_list = []
            
            # Cargar y predecir con cada modelo (KNN, SVM, RF, NB)
            model_types = ['KNN', 'SVM', 'RF', 'NB']
            models_loaded = 0
            
            for model_type in model_types:
                model_path = f"cross_validation_models/{model_type}_{model_suffix}.pkl"
                
                if os.path.exists(model_path):
                    # Cargar modelo
                    model = joblib.load(model_path)
                    
                    # Obtener predicciones
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)
                    
                    predictions_list.append(y_pred)
                    probabilities_list.append(y_prob)
                    models_loaded += 1
                else:
                    print(f"  Modelo no encontrado: {model_path}")
            
            # Si se cargaron al menos 2 modelos, aplicar ensemble
            if models_loaded >= 2:
                print(f"  Modelos cargados: {models_loaded}/{len(model_types)}")
                
                y_prob_mean = np.mean(probabilities_list, axis=0)
                
                y_pred_voting = voting_ensemble(predictions_list, probabilities_list)
                evaluate_model(y_test, y_pred_voting, y_prob_mean, f"VotingEnsemble_{model_suffix}")
                print(f"  ✓ VotingEnsemble evaluado")
                
                y_pred_mean = mean_ensemble(probabilities_list)
                evaluate_model(y_test, y_pred_mean, y_prob_mean, f"MeanEnsemble_{model_suffix}")
                print(f"  ✓ MeanEnsemble evaluado")
            else:
                print(f"  ✗ No se pudieron cargar suficientes modelos para el ensemble")

print("\n" + "="*80)
print("ENSEMBLE COMPLETADO")
print("="*80)

