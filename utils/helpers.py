import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from typing import Dict, Any
from prettytable import PrettyTable
from dataclasses import dataclass, field
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import get_scorer
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report


def custom_model_specs(model, scores_cv):
    """
    Generates a formatted table displaying the model's hyperparameters and cross-validation scores.

    Args:
    - model: An instance of CustomModel containing the model's parameters.
    - scores_cv: A dictionary of cross-validation scores (metric: score).

    Returns:
    - A string representation of the formatted table.
    """

    # Initialize the PrettyTable with headers
    table = PrettyTable(
        float_format=".2",  # Float formatting (rounding to 2 decimal places)
        junction_char="•",  # Character used for joining rows
        horizontal_char="—",  # Character used for horizontal separators
    )
    table.title = f"{model.name} - Cross-Validation Scores"
    table.field_names = ["Hyperparameter", "Value"]
    table.align["Hyperparameter"] = "l"
    table.align["Value"] = "r"

    # Add model hyperparameters
    for key, value in model.params.items():
        table.add_row([f"{key}", value])

    # Add a divider between hyperparameters and scores
    table.add_divider()

    # Add cross-validation scores (formatted as percentages)
    table.add_row(["Scoring Metric", "Value (%)"], divider=True)
    for metric, score in scores_cv.items():
        formatted_score = (
            f"{score * 100:.2f}%"  # Convert to percentage with 2 decimal places
        )
        table.add_row([f"{metric}", formatted_score])

    return table.get_string()


def custom_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    *,
    cmap="Blues",
    subplots_kws=None,
    cm_kws=None,
    annot_kws=None,
):
    """
    Generates and displays a customized confusion matrix with a heatmap.

    Args:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - cmap: The color map to use for the heatmap (default is "Blues").
    - labels: List of labels to display on the x and y axes (default is None).
    - subplots_kws: Additional arguments to customize the plot (default is None).
    - cm_kws: Additional arguments for confusion matrix computation (default is None).
    - annot_kws: Additional arguments to customize annotation appearance (default is None).

    Returns:
    - None (Displays the plot).
    """

    # Set default values for None arguments
    subplots_kws = subplots_kws or {}
    cm_kws = cm_kws or {}
    annot_kws = annot_kws or {}

    # Create the figure for plotting with the given subplot options
    fig, ax = plt.subplots(**subplots_kws)

    # If 'labels' is provided, ensure it's a list (if it's a NumPy array or pandas Series)
    if labels is not None:
        labels = (
            labels.tolist() if isinstance(labels, (np.ndarray, pd.Series)) else labels
        )

    # Compute the confusion matrix and transpose it for correct orientation
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, **cm_kws).transpose()

    # Create the heatmap with the confusion matrix
    sns.heatmap(
        data=cm,
        annot=True,  # Annotate the cells with numeric values
        fmt=".1f",  # Format the annotation to 1 decimal place
        cmap=cmap,  # Use the specified colormap
        linewidths=3,  # Set the linewidth between cells
        annot_kws={"fontweight": "bold", **annot_kws},  # Bold text for annotations
        ax=ax,  # Plot on the specified axis
    )

    # Set labels for the axes
    ax.set_xlabel("GROUND TRUTH", labelpad=10)
    ax.set_ylabel("PREDICTIONS", labelpad=10)

    # Set the x and y tick labels, if 'labels' are provided
    if labels:
        ax.set_xticks(ax.get_xticks(), labels=labels)
        ax.set_yticks(ax.get_yticks(), labels=labels)

    # Adjust the spines to make the borders more visible
    ax.spines[["left", "top"]].set_position(("outward", 10))

    # Configure tick parameters for a better appearance
    ax.tick_params(direction="inout", length=0)

    # Move the x-axis to the top
    ax.tick_params(labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")

    # Invert the x-axis and y-axis to match the usual confusion matrix layout
    ax.invert_yaxis()
    ax.invert_xaxis()

    # Access and modify the color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(length=0)  # Remove ticks on the color bar

    # Display the plot
    plt.show()


def custom_GridSearchCV(
    model,
    param_grid,
    X_train,
    y_train,
    scoring=["f1", "accuracy", "recall", "precision"],
    n_splits=3,
    random_state=42,
):
    """
    Custom implementation of GridSearchCV with multiple scoring metrics and cross-validation.

    Parameters:
    - model: The machine learning model to be trained (e.g., a classifier).
    - param_grid: A dictionary of hyperparameters and their values to be tested.
    - X_train: The training feature set.
    - y_train: The corresponding labels for training data.
    - scoring: List of scoring metrics to evaluate the model. Default includes f1, accuracy, recall, and precision.
    - n_splits: Number of splits for Stratified K-Fold cross-validation (default is 3).
    - random_state: Random seed for reproducibility in StratifiedKFold (default is 42).

    Returns:
    - best_estimator: The best-performing model after cross-validation.
    - best_params: The hyperparameters of the best-performing model.
    - best_scores_cv: The corresponding cross-validation scores for each metric.
    """

    param_grid = ParameterGrid(param_grid)  # Generate all parameter combinations
    k_folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )  # Cross-validation strategy

    results = []

    # Iterate through all hyperparameter combinations
    for params in tqdm(param_grid):
        model_clone = clone(model).set_params(**params)  # Clone and apply parameters

        # Dictionary to store cross-validation scores for each metric
        scoring_metrics = {metric: [] for metric in scoring}

        # Perform Stratified K-Fold cross-validation
        for train_idx, val_idx in k_folds.split(X_train, y_train):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model_clone.fit(X_train_cv, y_train_cv)  # Train model on fold

            # Evaluate using each scoring metric
            for metric in scoring:
                scorer = get_scorer(metric)  # Get the scorer function
                score = scorer(model_clone, X_val_cv, y_val_cv)  # Compute the score
                scoring_metrics[metric].append(score)  # Store score

        # Compute the average score across all folds for each metric
        avg_scores = {
            metric: np.mean(scores) for metric, scores in scoring_metrics.items()
        }

        # Store results
        results.append(
            {"estimator": model_clone, "params": params, "scores": avg_scores}
        )

    # Sort results based on all scoring metrics (descending order)
    results.sort(
        key=lambda item: tuple(item["scores"][metric] for metric in scoring),
        reverse=True,
    )

    # Extract the best model, parameters, and scores
    best_estimator = results[0]["estimator"]
    best_params = results[0]["params"]
    best_scores_cv = results[0]["scores"]

    return best_estimator, best_params, best_scores_cv


@dataclass
class CustomModel:
    """
    A custom model class to store an estimator, its parameters, and its evaluation scores.

    Attributes:
    - name: The name of the model (e.g., "Model 1").
    - estimator: The trained machine learning estimator (e.g., a classifier or regressor).
    - params: Hyperparameters used to configure the model.
    - scores_train: Evaluation scores on the training set.
    - scores_test: Evaluation scores on the test set.
    """

    name: str
    estimator: BaseEstimator
    params: Dict[str, Any]
    scores_train: Dict[str, float] = field(default_factory=dict)
    scores_test: Dict[str, float] = field(default_factory=dict)

    def predict(self, X):
        """
        Make predictions using the stored estimator.

        Args:
        - X: Input data for predictions.

        Returns:
        - Predictions made by the estimator.
        """
        return self.estimator.predict(X)

    def eval_train(self, X_train, y_train):
        """
        Evaluate the model on the training data and store the scores.

        Args:
        - X_train: Training features.
        - y_train: Training labels.
        """
        y_pred = self.predict(X_train)

        # Generate classification report as a dictionary
        report_dict = classification_report(y_train, y_pred, output_dict=True)

        # Convert report to DataFrame and drop unnecessary columns/rows
        report_frame = (
            pd.DataFrame(report_dict)
            .T.drop(columns=["support"], errors="ignore")
            .drop(index=["accuracy", "macro avg"], errors="ignore")
        )

        self.scores_train = report_frame.to_dict()

    def eval_test(self, X_test, y_test):
        """
        Evaluate the model on the test data and store the scores.

        Args:
        - X_test: Test features.
        - y_test: Test labels.
        """
        y_pred = self.predict(X_test)

        # Generate classification report as a dictionary
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Convert report to DataFrame and drop unnecessary columns/rows
        report_frame = (
            pd.DataFrame(report_dict)
            .T.drop(columns=["support"], errors="ignore")
            .drop(index=["accuracy", "macro avg"], errors="ignore")
        )

        self.scores_test = report_frame.to_dict()
