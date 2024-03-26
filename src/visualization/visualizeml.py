import pathlib
import joblib
import sys
import yaml
import pandas as pd
import mlflow
from mlflow import sklearn
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def evaluate(model, X, y, split):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        X (pandas.DataFrame): Input DF.
        y (pandas.Series): Target column.
        split (str): Dataset name.
    """

    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]

    # Log metrics to MLflow
    fpr, tpr, thresholds = roc_curve(y, predictions)
    roc_auc = auc(fpr, tpr)
    mlflow.log_metric("roc_auc_" + split, roc_auc)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{split}.png")
    mlflow.log_artifact(f"roc_curve_{split}.png")

    plt.close()


def save_importance_plot(model, feature_names):
    """
    Save feature importance plot.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names.
    """
    fig, ax = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    ax.set_ylabel("Mean decrease in impurity")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=ax)

    mlflow.log_figure(fig, "feature_importance")


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    model_file = sys.argv[1]
    # Load the model.
    model = joblib.load(model_file)

    # Load the data.
    input_file = sys.argv[2]
    data_path = home_dir.joinpath(input_file)
    output_path = home_dir.joinpath('mlflow')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    TARGET = 'Class'
    train_features = pd.read_csv(data_path.joinpath('train.csv'))
    X_train = train_features.drop(TARGET, axis=1)
    y_train = train_features[TARGET]
    feature_names = X_train.columns.to_list()

    test_features = pd.read_csv(data_path.joinpath('test.csv'))
    X_test = test_features.drop(TARGET, axis=1)
    y_test = test_features[TARGET]

    # Set MLflow tracking URI
    # mlflow.set_tracking_uri("your_mlflow_tracking_uri")

    # Start MLflow run
    with mlflow.start_run():
        # Evaluate train and test datasets.
        evaluate(model, X_train, y_train, "train")
        evaluate(model, X_test, y_test, "test")

        # Dump feature importance plot.
        # save_importance_plot(model, feature_names)

        # Log model
            


if __name__ == "__main__":
    main()
