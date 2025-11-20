import nbimporter
import train
import eval

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
                y_test, y_pred, y_prob, model_name = train.train_model(method, iteration, transform, value)
                eval.evaluate_model(y_test, y_pred, y_prob, model_name)