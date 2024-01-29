import pandas as pd

from sklearn.experimental import enable_halving_search_cv

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from datetime import datetime

import joblib

import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

cat = True  # if True, fraud type is used as target variable, if False, potential fraud is used as target variable

# sampling strategies for ROS and RUS for each class
ros_sampling_strategy = {1: 50000, 2: 50000, 3: 50000, 4: 50000, 5: 50000, 6: 50000}
rus_sampling_strategy = {0: 100000}


def initialize_data(cat):
    """
    Initialize the data for training and testing.

    Args:
        cat (bool): If True, fraud type is used as target variable, if False, potential fraud is used as target variable

    Returns:
        y_train (Series): Target variable for training
        X_train (DataFrame): Features for training
        y_test (Series): Target variable for testing
        X_test (DataFrame): Features for testing"""
    if cat:
        # replace POTENTIAL_FRAUD with the ID in fraud_id for each row in claims
        claims = pd.read_csv(
            "data/generated data/preprocessing/preprocessed.csv",
            engine="python",
            #usecols=["ID", "VENTILATION", "AVG_VENTILATION", "POTENTIAL_FRAUD"],
        )
        """fraud_id = pd.read_csv("data/generated data/preprocessing/ID_Fraud_Mapping.csv")
        claims["POTENTIAL_FRAUD"] = claims["ID"].map(
            fraud_id.set_index("ID")["FRAUD_ID"]
        )
        # set potential_fraud to 0 if it is not 1
        claims["POTENTIAL_FRAUD"] = claims["POTENTIAL_FRAUD"].apply(
            lambda x: 0 if x != 1 else x
        )"""
        y_train = claims["POTENTIAL_FRAUD"]  # target variable
        X_train = claims.drop("POTENTIAL_FRAUD", axis=1)  # features

        del claims  # delete claims to save memory

        validation = pd.read_csv(
            "data/generated data/preprocessing/validation.csv",
            engine="python",
            #usecols=["ID", "VENTILATION", "AVG_VENTILATION", "POTENTIAL_FRAUD"],
        )
        """validation["POTENTIAL_FRAUD"] = validation["ID"].map(
            fraud_id.set_index("ID")["FRAUD_ID"]
        )
        validation["POTENTIAL_FRAUD"] = validation["POTENTIAL_FRAUD"].apply(
            lambda x: 0 if x != 1 else x
        )"""
        y_test = validation["POTENTIAL_FRAUD"]  # target variable
        X_test = validation.drop("POTENTIAL_FRAUD", axis=1)  # features
        del fraud_id  # delete fraud_id to save memory
        del validation  # delete validation to save memory
    else:
        claims = pd.read_csv(
            "data/generated data/preprocessing/preprocessed.csv",
            engine="python",  # , usecols=["VENTILATION", "AVG_VENTILATION", "POTENTIAL_FRAUD"]
        )
        y_train = claims["POTENTIAL_FRAUD"]  # target variable
        X_train = claims.drop("POTENTIAL_FRAUD", axis=1)  # features
        del claims  # delete claims to save memory
        validation = pd.read_csv(
            "data/generated data/preprocessing/validation.csv", engine="python"
        )
        y_test = validation["POTENTIAL_FRAUD"]  # target variable
        X_test = validation.drop("POTENTIAL_FRAUD", axis=1)  # features
        del validation  # delete validation to save memory

    return y_train, X_train, y_test, X_test


y_train, X_train, y_test, X_test = initialize_data(cat)

feature_names = X_train.columns.tolist()

# set scoring metrics
scoring = {
    "f1": make_scorer(f1_score, average="macro"),
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="macro"),
    "recall": make_scorer(recall_score, average="macro"),
    "f1_weighted": make_scorer(f1_score, average="weighted"),
    "precision_weighted": make_scorer(precision_score, average="weighted"),
    "recall_weighted": make_scorer(recall_score, average="weighted"),
    "precision_class_0": make_scorer(precision_score, average=None, labels=[0]),
    "precision_class_1": make_scorer(precision_score, average=None, labels=[1]),
    "recall_class_0": make_scorer(recall_score, average=None, labels=[0]),
    "recall_class_1": make_scorer(recall_score, average=None, labels=[1]),
}


def knn():
    """KNN is not used in the final model, because it takes too long to train.

    It remains in the code for demonstration purposes. At some point, the code was not further adjusted to the new setup.
    """
    params_knn = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        # "selector__threshold": [0, 0.001, 0.01],
        "classifier__n_neighbors": [1, 3, 5, 7, 10],
        "classifier__p": [1, 2],
        # "classifier__leaf_size": [1, 5, 10, 15],
    }

    pipe_knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            ("smote", SMOTE()),
            ("classifier", KNeighborsClassifier()),
        ]
    )

    grid_knn = GridSearchCV(
        pipe_knn, param_grid=params_knn, scoring=scoring, refit="f1", cv=2, n_jobs=4
    ).fit(X_train, y_train)

    y_pred = grid_knn.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print_stats(grid_knn, cm)


def rf(halving=False):
    """Train a random forest classifier on the training data and evaluate it on the validation data.

    Several combinations of hyperparameters are tested, the best parameters found are now hardcoded in the function.
    The hyperparameters tested but not resulting in the best model are commented out.

    Args:
        halving (bool, optional): If True, halving grid search is used. Defaults to False.
    """
    params_rf = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        # "selector__threshold": [0, 0.001, 0.01],
        "classifier__n_estimators": [800],  # 700,  1000
        "classifier__criterion": ["log_loss"],  # other: "entropy", "gini", "log_loss"
        "classifier__min_samples_split": [3],  # 2, 4
        "classifier__max_depth": [None],
    }

    pipe_rf = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            # ("smote", SMOTE() ),
            ("oversampler", RandomOverSampler(sampling_strategy=ros_sampling_strategy)),
            (
                "undersampler",
                RandomUnderSampler(sampling_strategy=rus_sampling_strategy),
            ),
            ("classifier", RandomForestClassifier()),
        ]
    )
    if halving:
        grid_rf = HalvingGridSearchCV(
            pipe_rf,
            param_grid=params_rf,
            scoring=scoring["precision_class_1"],
            refit=True,
            cv=5,
            n_jobs=2,
        ).fit(X_train, y_train)
    else:
        grid_rf = GridSearchCV(
            pipe_rf,
            param_grid=params_rf,
            scoring=scoring,
            refit="precision_class_1",
            cv=5,
            n_jobs=4,
        ).fit(X_train, y_train)

    y_pred = grid_rf.best_estimator_.predict(X_test)

    print_stats(grid_rf, "rf")


def gb(halving=False):
    """Train a gradient boosting classifier on the training data and evaluate it on the validation data.

    Several combinations of hyperparameters are tested, the best parameters found are now hardcoded in the function.
    The hyperparameters tested but not resulting in the best model are commented out.

    Args:
        halving (bool, optional): If True, halving grid search is used. Defaults to False.
    """

    params_gb = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        # "selector__threshold": [0, 0.001, 0.01],
        "classifier__n_estimators": [1000],  # 200, 300, 500,
        "classifier__learning_rate": [0.7],  # 0.8, 0.9, 0.5,
        # "classifier__max_depth": [2, 3, 4], # the default proofed to work best
    }
    pipe_gb = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            # ("smote", SMOTE()),
            ("oversampler", RandomOverSampler(sampling_strategy=ros_sampling_strategy)),
            (
                "undersampler",
                RandomUnderSampler(sampling_strategy=rus_sampling_strategy),
            ),
            ("classifier", GradientBoostingClassifier()),
        ]
    )

    if halving:
        grid_gb = HalvingGridSearchCV(
            pipe_gb,
            param_grid=params_gb,
            scoring=scoring["precision_class_1"],
            refit=True,
            cv=5,
            n_jobs=4,
        ).fit(X_train, y_train)

    else:
        grid_gb = GridSearchCV(
            pipe_gb,
            param_grid=params_gb,
            scoring=scoring,
            refit="precision_class_1",
            cv=2,
            n_jobs=4,
        ).fit(X_train, y_train)

    print_stats(grid_gb, "gb")


def svm():
    """KNN is not used in the final model, because it takes too long to train.

    It remains in the code for demonstration purposes. At some point, the code was not further adjusted to the new setup.
    """
    params_svm = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        # "selector__threshold": [0, 0.001, 0.01],
        "classifier__C": [1],  # 0.1, 1, 10
        # "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
        # "classifier__degree": [2, 3, 4],
    }

    pipe_svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            ("smote", SMOTE()),
            ("classifier", SVC()),
        ]
    )
    grid_svm = GridSearchCV(
        pipe_svm, param_grid=params_svm, scoring=scoring, refit="f1", cv=2, n_jobs=4
    ).fit(X_train, y_train)
    y_pred = grid_svm.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print_stats_old(grid_svm, cm, "svm")


def svm_new():
    """Train an SVM classifier on the training data and evaluate it on the validation data to detect outliers for fraud in ventilation hours.

    Several combinations of hyperparameters are tested, the best parameters found are now hardcoded in the function.

    The results where not used in the thesis as the results were not good enough.
    """
    params_svm = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        # "selector__threshold": [0, 0.001, 0.01],
        "classifier__C": [0.1, 1, 10],
        # "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
        # "classifier__degree": [2, 3, 4],
    }

    pipe_svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            ("smote", SMOTE()),
            ("classifier", SVC()),
        ]
    )

    # X_train = X_train["VENTILATION"]
    # X_test = X_test["VENTILATION"]
    grid_svm = GridSearchCV(
        pipe_svm, param_grid=params_svm, scoring=scoring, refit="f1", cv=2, n_jobs=4
    ).fit(X_train, y_train)
    y_pred = grid_svm.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print_stats_old(grid_svm, cm, "svm")


def isoforest():
    """Train an isolation forest classifier on the training data and evaluate it on the validation data to detect outliers for fraud in ventilation hours.

    Several combinations of hyperparameters are tested, the best parameters found are now hardcoded in the function.

    The results where not used in the thesis as the results were not good enough.
    """

    params_isoforest = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        "classifier__n_estimators": [100, 200, 300],
        # "classifier__max_samples": [0.25, 0.5, 0.75],
        # "classifier__contamination": [0.01, 0.05, 0.1],
    }

    pipe_isoforest = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            ("classifier", IsolationForest()),
        ]
    )

    grid_isoforest = GridSearchCV(
        pipe_isoforest,
        param_grid=params_isoforest,
        scoring=scoring,
        refit="f1",
        cv=2,
        n_jobs=4,
    ).fit(X_train, y_train)
    y_pred = grid_isoforest.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print_stats_old(grid_isoforest, cm, "isoforest")


def mlp(halving=False):
    """Train a multi-layer perceptron classifier on the training data and evaluate it on the validation data.

    Several combinations of hyperparameters are tested, the best parameters found are now hardcoded in the function.
    The hyperparameters tested but not resulting in the best model are commented out.

    Args:
        halving (bool, optional): If True, halving grid search is used. Defaults to False.
    """
    params_mlp = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        # "selector__threshold": [0, 0.001, 0.01],
        "classifier__hidden_layer_sizes": [(800, 400)],  # (1600,800), failed
        "classifier__activation": ["relu"],  # "identity", "logistic", "tanh"
        "classifier__solver": ["adam"],  # "lbfgs", "sgd",
        # "classifier__alpha": [0.0001, 0.001, 0.01], # the default proofed to work best
        "classifier__learning_rate": ["constant"],  # "invscaling", "adaptive"
    }

    pipe_mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            ("smote", SMOTE()),
            ("classifier", MLPClassifier()),
        ]
    )

    if halving:
        grid_mlp = HalvingGridSearchCV(
            pipe_mlp,
            param_grid=params_mlp,
            scoring=scoring["precision_class_1"],
            refit=True,
            cv=5,
            n_jobs=4,
        ).fit(X_train, y_train)

    else:
        grid_mlp = GridSearchCV(
            pipe_mlp,
            param_grid=params_mlp,
            scoring=scoring,
            refit="precision_class_1",
            cv=5,
            n_jobs=4,
        ).fit(X_train, y_train)

    print_stats(grid_mlp, "mlp")


def lr(halving=False):
    """Train a logistic regression classifier on the training data and evaluate it on the validation data.

    Several combinations of hyperparameters are tested, the best parameters found are now hardcoded in the function.
    The hyperparameters tested but not resulting in the best model are commented out.

    Args:
        halving (bool, optional): If True, halving grid search is used. Defaults to False.
    """
    params_lr = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        # "selector__threshold": [0, 0.001, 0.01], # the default proofed to work best
        "classifier__penalty": ["l2"],  # "l1", "l2", "elasticnet"
        "classifier__C": [20],  # 10, 100
        "classifier__solver": ["newton-cg"],  # , "lbfgs", "liblinear", "sag", "saga"
        "classifier__max_iter": [300],  # 200, 400
    }

    pipe_lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            ("smote", SMOTE()),
            ("classifier", LogisticRegression()),
        ]
    )

    if halving:
        grid_lr = HalvingGridSearchCV(
            pipe_lr,
            param_grid=params_lr,
            scoring=scoring["precision_class_1"],
            refit=True,
            cv=5,
            n_jobs=4,
        ).fit(X_train, y_train)
    else:
        grid_lr = GridSearchCV(
            pipe_lr,
            param_grid=params_lr,
            scoring=scoring,
            refit="precision_class_1",
            cv=2,
            n_jobs=4,
        ).fit(X_train, y_train)

    print_stats(grid_lr, "lr")


def dt(halving=False):
    """Train a decision tree classifier on the training data and evaluate it on the validation data.

    Several combinations of hyperparameters are tested, the best parameters found are now hardcoded in the function.
    The hyperparameters tested but not resulting in the best model are commented out.

    Args:
        halving (bool, optional): If True, halving grid search is used. Defaults to False.
    """

    params_dt = {
        # "scaler": [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
        # "selector__threshold": [0, 0.001, 0.01], # the default proofed to work best
        # "classifier__criterion": ["gini", "entropy"], # the default proofed to work best
        "classifier__splitter": ["best"],  # "random"
        "classifier__max_depth": [20],  # None, 2, 3,4, 5, 6, , 50, 200
    }

    pipe_dt = Pipeline(
        [
            ("scaler", StandardScaler()),
            # ("selector", VarianceThreshold()),
            ("smote", SMOTE()),
            ("classifier", DecisionTreeClassifier()),
        ]
    )

    if halving:
        grid_dt = HalvingGridSearchCV(
            pipe_dt,
            param_grid=params_dt,
            scoring=scoring["precision_class_1"],
            refit=True,
            cv=5,
            n_jobs=4,
        ).fit(X_train, y_train)
    else:
        grid_dt = GridSearchCV(
            pipe_dt, param_grid=params_dt, scoring=scoring, refit="f1", cv=2, n_jobs=4
        ).fit(X_train, y_train)

    print_stats(grid_dt, "dt")


def print_stats(grid, name):
    """Prints the best parameters and the evaluation metrics of the best model.

    Prints:
        - Best parameters
        - Best F1 score
        - Confusion matrix
        - Best weighted F1 score
        - Best accuracy
        - Best weighted accuracy
        - Best precision
        - Best weighted precision
        - Best recall
        - Best weighted recall
        - Best precision for class 0
        - Best precision for class 1
        - Best recall for class 0
        - Best recall for class 1
        - Feature importances (if applicable)

    Saves the best model to data/models/ as a .pkl file. Adding "_without_ratio" to the name if ratio_sectio was not used in the model. Adding "_cat" to the name if fraud type was used as target variable.

    Args:
        grid (GridSearchCV): GridSearchCV object containing the best model.
        name (str): Name of the model.
    """

    y_pred = grid.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best F1 score: {grid.best_score_:.2f}")
    print(f"Confusion matrix: {cm}")
    print(f'Best weighted F1-score: {f1_score(y_test, y_pred, average="macro")}')
    print(f"Best accuracy: {accuracy_score(y_test, y_pred)}")
    # print(
    #    f'Best weighted accuracy: {grid_rf.cv_results_["mean_test_accuracy_weighted"][grid_rf.best_index_]:.2f}'
    # )
    print(f'Best precision: {precision_score(y_test, y_pred, average="macro")}')
    print(f'Best weighted precision: {recall_score(y_test, y_pred, average="macro")}')
    print(f'Best recall: {f1_score(y_test, y_pred, average="weighted")}')
    print(
        f'Best weighted recall: {precision_score(y_test, y_pred, average="weighted")}'
    )

    print(
        f"Best precision class 0: {precision_score(y_test, y_pred, average=None, labels=[0])}"
    )
    print(
        f"Best recall class 0: {recall_score(y_test, y_pred, average=None, labels=[0])}"
    )
    print(
        f"Best precision class 1: {precision_score(y_test, y_pred, average=None, labels=[1])}"
    )
    print(
        f"Best recall class 1: {recall_score(y_test, y_pred, average=None, labels=[1])}"
    )
    if cat:
        print(
            f"Best precision class 2: {precision_score(y_test, y_pred, average=None, labels=[2])}"
        )
        print(
            f"Best recall class 2: {recall_score(y_test, y_pred, average=None, labels=[2])}"
        )
        print(
            f"Best precision class 3: {precision_score(y_test, y_pred, average=None, labels=[3])}"
        )
        print(
            f"Best recall class 3: {recall_score(y_test, y_pred, average=None, labels=[3])}"
        )
        print(
            f"Best precision class 4: {precision_score(y_test, y_pred, average=None, labels=[4])}"
        )
        print(
            f"Best recall class 4: {recall_score(y_test, y_pred, average=None, labels=[4])}"
        )
        print(
            f"Best precision class 5: {precision_score(y_test, y_pred, average=None, labels=[5])}"
        )
        print(
            f"Best recall class 5: {recall_score(y_test, y_pred, average=None, labels=[5])}"
        )
        print(
            f"Best precision class 6: {precision_score(y_test, y_pred, average=None, labels=[6])}"
        )
        print(
            f"Best recall class 6: {recall_score(y_test, y_pred, average=None, labels=[6])}"
        )

        print(f"f1 score class 0: {f1_score(y_test, y_pred, average=None, labels=[0])}")

        print(f"f1 score class 1: {f1_score(y_test, y_pred, average=None, labels=[1])}")

        print(f"f1 score class 2: {f1_score(y_test, y_pred, average=None, labels=[2])}")

        print(f"f1 score class 3: {f1_score(y_test, y_pred, average=None, labels=[3])}")

        print(f"f1 score class 4: {f1_score(y_test, y_pred, average=None, labels=[4])}")

        print(f"f1 score class 5: {f1_score(y_test, y_pred, average=None, labels=[5])}")

        print(f"f1 score class 6: {f1_score(y_test, y_pred, average=None, labels=[6])}")

    if (
        "RATIO_SECTIO" not in feature_names
    ):  # if ratio_sectio is in the feature names, the model was trained with it
        name = name + "_without_ratio"
    if cat == True:
        name = name + "_cat"
    joblib.dump(grid.best_estimator_, "data/models/" + name + ".pkl")

    feature_importances = grid.best_estimator_.named_steps[
        "classifier"
    ].feature_importances_
    # sort the features by importance
    sorted_features = sorted(
        zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True
    )

    # print the sorted features
    for feature, importance in sorted_features:
        print(f"{feature}: {importance}")

    sns.heatmap(cm, annot=True)
    # plt.show()


def print_stats_old(grid, cm, name):
    """Deprecated! Remains only for not further developed models KNN and SVM. Prints the best parameters and the evaluation metrics of the best model.

    Prints:
        - Best parameters
        - Best F1 score
        - Confusion matrix
        - Best weighted F1 score
        - Best accuracy
        - Best weighted accuracy
        - Best precision
        - Best weighted precision
        - Best recall
        - Best weighted recall
        - Best precision for class 0
        - Best precision for class 1
        - Best recall for class 0
        - Best recall for class 1
        - Feature importances (if applicable)

    Saves the best model to data/models/ as a .pkl file. Adding "_without_ratio" to the name if ratio_sectio was not used in the model.

    Args:
        grid (GridSearchCV): GridSearchCV object containing the best model.
        cm (array): Confusion matrix.
        name (str): Name of the model.
    """

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best F1 score: {grid.best_score_:.2f}")
    print(f"Confusion matrix: {cm}")
    print(
        f'Best weighted F1-score: {grid.cv_results_["mean_test_f1_weighted"][grid.best_index_]:.2f}'
    )
    print(
        f'Best accuracy: {grid.cv_results_["mean_test_accuracy"][grid.best_index_]:.2f}'
    )
    # print(
    #    f'Best weighted accuracy: {grid_rf.cv_results_["mean_test_accuracy_weighted"][grid_rf.best_index_]:.2f}'
    # )
    print(
        f'Best precision: {grid.cv_results_["mean_test_precision"][grid.best_index_]:.2f}'
    )
    print(
        f'Best weighted precision: {grid.cv_results_["mean_test_precision_weighted"][grid.best_index_]:.2f}'
    )
    print(f'Best recall: {grid.cv_results_["mean_test_recall"][grid.best_index_]:.2f}')
    print(
        f'Best weighted recall: {grid.cv_results_["mean_test_recall_weighted"][grid.best_index_]:.2f}'
    )

    print(
        f'Best precision class 0: {grid.cv_results_["mean_test_precision_class_0"][grid.best_index_]:.2f}'
    )
    print(
        f'Best precision class 1: {grid.cv_results_["mean_test_precision_class_1"][grid.best_index_]:.2f}'
    )
    print(
        f'Best recall class 0: {grid.cv_results_["mean_test_recall_class_0"][grid.best_index_]:.2f}'
    )
    print(
        f'Best recall class 1: {grid.cv_results_["mean_test_recall_class_1"][grid.best_index_]:.2f}'
    )

    joblib.dump(grid.best_estimator_, "data/models/" + name + ".pkl")

    feature_importances = grid.best_estimator_.named_steps[
        "classifier"
    ].feature_importances_
    # sort the features by importance
    sorted_features = sorted(
        zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True
    )

    # print the sorted features
    for feature, importance in sorted_features:
        print(f"{feature}: {importance}")

    sns.heatmap(cm, annot=True)
    # plt.show()


if __name__ == "__main__":
    """Run all models and print their evaluation metrics.

    The try-except blocks are used to continue the execution of the script even if one model fails.
    Date and time are printed before and after each model to track the duration of the execution.
    """

    print(datetime.now())
    # knn() # taking too long, probably due to high dimensionality
    try:
        rf(True)
    except:
        print("RF failed")
    print(datetime.now())

    try:
        gb(True)
    except:
        print("GB failed")
    # svm() # taking too long, probably due to high dimensionality
    print(datetime.now())

    # svm_new() # test for SVM in outlier detection for ventilation hours
    # isoforest() # test for isolation forest in outlier detection for ventilation hours
    try:
        lr(True)
    except:
        print("LR failed")
    print(datetime.now())

    try:
        mlp(True)
    except:
        print("MLP failed")
    print(datetime.now())

    try:
        dt(True)
    except:
        print("DT failed")
    print(datetime.now())
