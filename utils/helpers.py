import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, get_scorer
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from tqdm.notebook import tqdm


def custom_GridSearchCV(
    model,
    param_grid,
    X_train,
    y_train,
    scoring=["f1", "accuracy", "recall", "precision"],
    n_splits=3,
    random_state=42,
):
    # Grid of parameter combinations
    param_grid = ParameterGrid(param_grid)

    # Cross-validation folds
    k_folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    results = []

    # For each parameter combination
    for params in tqdm(param_grid):
        # Set the parameters for the current model configuration
        model = clone(model).set_params(**params)

        # Initialize the scoring metrics
        scoring_metrics = {key: [] for key in scoring}

        # For each fold:
        for train_index, val_index in k_folds.split(X_train, y_train):
            # Retrive the training fold
            X_train_cv, y_train_cv = (
                X_train.iloc[train_index],
                y_train.iloc[train_index],
            )
            # Retrive the validation fold
            X_val_cv, y_val_cv = X_train.iloc[val_index], y_train.iloc[val_index]

            # Fit the model
            model.fit(X_train_cv, y_train_cv)

            # Validate the model
            # For each scoring metric
            for scoring_method in scoring:
                # Get the corresponding socring function
                scorer = get_scorer(scoring_method)
                # Compute the score
                score = scorer(model, X_val_cv, y_val_cv)
                # Save the score
                scoring_metrics[scoring_method].append(score)

        # Compute mean score for each metric
        scoring_metrics = {
            key: np.mean(values) for key, values in scoring_metrics.items()
        }

        # Store the model along with the hyper-parameters and scoring metrics
        results.append(
            {"estimator": model, "params": params, "scores": scoring_metrics}
        )

    # Sort the results by all scoring metrics (in descending order)
    results = sorted(
        results,
        key=lambda item: tuple(item["scores"][metric] for metric in scoring),
        reverse=True,
    )

    return results


def custom_model_specs(title, result):
    table = PrettyTable(
        title=title,
        header_style="title",
        field_names=["Parameter", "Value"],
        float_format=".2",
        junction_char="•",
        horizontal_char="—",
    )

    # Print hyper-paramters
    for idx, (key, value) in enumerate(result["params"].items()):
        if idx == len(result["params"].items()) - 1:
            table.add_row([key, value], divider=True)
        else:
            table.add_row([key, value], divider=False)

    # Print scoring metrics
    table.add_row(["Scoring Metric", "Value (%)"], divider=True)
    for key, value in result["scores"].items():
        table.add_row([key, value], divider=False)

    return table.get_string()


def custom_confusion_matrix(y_true, y_pred, cmap="Blues", labels=None, normalize=None):
    # Create the figure
    fig, ax = plt.subplots()

    # Compute the confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalize).transpose()

    # Build the classication matrix
    sns.heatmap(
        data=cm,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        linewidths=3,
        annot_kws=dict(fontweight="bold"),
        ax=ax,
    )

    # Set x-axis and y-axis labels
    ax.set_xlabel("GROUND TRUTH", labelpad=10)
    ax.set_ylabel("PREDICTIONS", labelpad=10)

    # Set the x and y tick labels
    ax.set_xticks(ax.get_xticks(), labels)
    ax.set_yticks(ax.get_yticks(), labels)

    # Adjust the spines
    ax.spines[["left", "top"]].set_position(("outward", 10))

    # Tick params
    ax.tick_params(direction="inout", length=0)

    # Move the x-axis to the top
    ax.tick_params(labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")

    # Invert the x-axis and y-axis
    ax.invert_yaxis()
    ax.invert_xaxis()

    # Access the color bar
    cbar = ax.collections[0].colorbar

    # Modify tick apperance for color bar
    cbar.ax.tick_params(length=0)

    # Render the plot
    plt.show()


def custom_classification_report(y_true, y_pred, labels=None):
    # Generate the classification report as a dictionary and convert it to a DataFrame
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_frame = pd.DataFrame(report_dict).transpose().drop(columns="support")

    # If 'labels' is provided
    if labels is not None:
        # Ensure 'labels' is a list (if it's a NumPy array or pandas Series)
        labels = (
            labels.tolist() if isinstance(labels, (np.ndarray, pd.Series)) else labels
        )

        # Create the new index by combining 'labels' and the rest of the current index
        new_index = labels + report_frame.index[len(labels) :].to_list()

        # Assign the new index to the DataFrame
        report_frame.index = new_index

    return report_frame


from dataclasses import dataclass
from typing import Dict, Any
from sklearn.base import BaseEstimator


@dataclass
class CustomModel:
    estimator: BaseEstimator
    params: Dict[str, Any]
    scores: Dict[str, float]

    @classmethod
    def from_dict(cls, dict_):
        data = dict_
        return cls(**data)
