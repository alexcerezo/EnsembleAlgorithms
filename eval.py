#!/usr/bin/env python
# coding: utf-8

# # Evaluación de los modelos
#
# Evaluamos los modelos con las diferentes métricas mencionadas en la práctica:
# | Métrica | Definición |
# |---|---|
# | F1-score (Fm) | Fm = 2 × (PR × RC) / (PR + RC) |
# | Sensibilidad (S) | S = TP / (TP + FN) |
# | Exactitud (Acc) | Acc = (TP + TN) / (TP + FP + FN + TN) |
# | Especificidad (SP) | SP = TN / (FP + TN) |
# | Recall (RC) | RC = TP / (TP + FN) |
# | Precisión (PR) | PR = TP / (TP + FP) |
# | Tasa de falsos negativos (FNR) | FNR = FN / (TP + FN) |
# | Tasa de falsos positivos (FPR) | FPR = FP / (FP + TN) |
#
# Para ello, usamos una función que se encarga de obtener todas las métricas excepto la curva ROC y el AUC.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    sensitivities = []
    specificities = []
    fprs = []
    fnrs = []

    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivities.append(sensitivity)

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(specificity)

        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fprs.append(fpr)

        fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
        fnrs.append(fnr)

    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    avg_fpr = np.mean(fprs)
    avg_fnr = np.mean(fnrs)

    results = pd.DataFrame(
        {
            "metric": [
                "Sensibilidad",
                "Exactitud",
                "Especificidad",
                "Recall",
                "Precisión",
                "Tasa de Falsos Negativos",
                "Tasa de Falsos Positivos",
                "F1-score",
            ],
            "value": [
                avg_sensitivity,
                accuracy,
                avg_specificity,
                recall,
                precision,
                avg_fnr,
                avg_fpr,
                f1,
            ],
        }
    )

    return results


# A continuación, generamos la curva ROC y el AUC (área bajo la curva).

# In[ ]:


from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional


def get_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, name: str, show: Optional[bool] = False
) -> float:
    # Obtenemos el número de clases
    n_classes = len(np.unique(y_true))

    # Clasificación multiclase - usar estrategia One-vs-Rest
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)

    # Calcular ROC y AUC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcular macro-average AUC
    macro_roc_auc = np.mean(list(roc_auc.values()))

    # Graficar curvas ROC para cada clase
    plt.figure(figsize=(10, 8))
    colors = ["blue", "red", "green", "orange", "purple", "brown"]

    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            color=colors[i % len(colors)],
            lw=2,
            label=f"Clase {classes[i]} (AUC = {roc_auc[i]:.4f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Clasificador aleatorio",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curva ROC Multiclase (Macro-AUC = {macro_roc_auc:.4f})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"cross_validation_evaluation/{name}.png")
    # Default behaviour: do not show by default; caller can set show=True
    if show:
        plt.show()
    else:
        plt.close()

    roc_auc = macro_roc_auc

    return roc_auc


# Por último, unimos los resultados de ambas funciones en un mismo Dataframe que se guarda en un CSV.

# In[ ]:


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    name: str,
    show: Optional[bool] = False,
) -> pd.DataFrame:
    metrics_df = get_metrics(y_true, y_pred)
    roc_auc = get_roc_curve(y_true, y_prob, name, show=show)

    # Añadir AUC al DataFrame de métricas
    auc_df = pd.DataFrame({"metric": ["AUC"], "value": [roc_auc]})

    final_results = pd.concat([metrics_df, auc_df], ignore_index=True)
    final_results.to_csv(f"cross_validation_evaluation/{name}.csv", index=False)

    return final_results
