from train import KNN, SVM, RF, NB, train_model
from eval import get_metrics, get_roc_curve, evaluate_model
from results import create_f1_comparison_table, create_summary_table

methods = {
    "K" : KNN,
    "S" : SVM,
    "R" : RF,
    "N" : NB,
}

transformations = {
    "N" : "norm",
    "O" : "original",
    "S" : "std",
}

PCA = [0, 80, 95]


for method in methods.keys():
    for iteration in range(1, 6):
        for transform in transformations.keys():
            for value in PCA:
                y_test, y_pred, y_prob, model_name = train_model(method, iteration, transform, value)
                evaluate_model(y_test, y_pred, y_prob, model_name)

method_names = ['KNN', 'SVM', 'RF', 'NB']
for method_name in method_names:
    create_summary_table(method_name)          

create_f1_comparison_table()                