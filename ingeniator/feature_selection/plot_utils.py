"""Various plots to analyze predictions, feature importances, and model confidence."""
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging
from sklearn.calibration import CalibratedClassifierCV
import systemicmacrorisk.settings as settings
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel
import plotly_express as px
from typing import Any, Dict, List, Tuple, Optional
import pathlib
import copy


def save_fig(figure: object, file_path: pathlib.Path) -> None:
    """Generic wrapper to save go.Figure or plt.Figure objects to file_path.

    :param figure: Figure to save, either go.Figure or plt.Figure.
    :type figure: object
    :param file_path: Complete path to save file to, including file name and extension.
    :type file_path: pathlib.Path
    """
    if type(figure) == plt.Figure:
        save_plt_fig(figure, file_path)
    elif type(figure) == go.Figure:
        save_go_fig(figure, file_path)


def save_go_fig(fig: go.Figure, plot_path: pathlib.Path = None) -> None:
    """Save plotly graph objects figure to file.

    :param fig: Figure to save.
    :type fig: go.Figure
    :param plot_path: Plot path, excluding file name. If None, defaults to settings.PLOT_PATH. Defaults to None
    :type plot_path: Optional[str], optional
    """
    fig.write_image(plot_path)
    logging.info(f"Figure saved to: {plot_path}")


def save_plt_fig(fig: plt.Figure, plot_path: pathlib.Path = None) -> None:
    """Save matplotlib fig to file.

    :param fig: Figure to save.
    :type fig: plt.Figure
    :param plot_path: Plot path, excluding file name. If None, defaults to settings.PLOT_PATH. Defaults to None
    :type plot_path: Optional[str], optional
    """
    fig.savefig(plot_path, bbox_inches="tight")
    logging.info(f"Figure saved to: {plot_path}")


def get_prediction_plot(
    y_test: pd.Series,
    y_pred: pd.Series,
    fig_title: str,
    figsize: Optional[Tuple[int]] = None,
) -> plt.Figure:
    """Plot predictions and actual labels overlaid vs. date.

    :param y_test: True labels.
    :type y_test: pd.Series
    :param y_pred: Predicted labels.
    :type y_pred: pd.Series
    :param run_name: Name of run in azure mlflow.
    :type run_name: str
    :param figsize: Tuple in the form of width, height (inches). If None, default to (12,8)
    :return: Figure object.
    :rtype: plt.Figure
    """
    if figsize is None:
        figsize = (12, 8)
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=y_test.index, y=y_test, ax=ax)
    sns.lineplot(x=y_pred_series.index, y=y_pred_series, ax=ax)
    ax.set_title(fig_title)
    return fig


def get_overall_feature_importance(
    results: Dict[str, Any], min_occurrences: int = 2
) -> pd.DataFrame:
    """Get normal-based feature importance's sorted by Relative importance and mean rank of feature over all folds.

    :param results: Results dictionary containing keys 'Estimators' & 'X_train'. Estimators need to be prefit to X_train
    :type results: Dict[str, Any]
    :param min_occurrences: Minimum number of folds that a feature must occur in to be considered, defaults to 2
    :type min_occurrences: int, optional
    :return: Dataframe of features ranked by relative importance and containing other metrics to compare features with.
    :rtype: pd.DataFrame
    """
    importance_dfs = [
        get_importance_df(
            results["Estimators"][fold_index], results["X_train"][fold_index]
        )
        for fold_index in range(len(results["X_train"]))
    ]
    importance_dfs = get_feature_ranks(importance_dfs)
    importance_dfs_joined = pd.concat(importance_dfs)
    rename_dict = {
        ("Relative Importance", "count"): "Occurrence Count",
        ("Relative Importance", "mean"): "Relative Importance",
        ("Rank", "sum"): "Sum of rank percentile",
        ("Rank", "mean"): "Mean rank percentile",
    }
    feature_metrics = importance_dfs_joined.groupby("Feature").agg(
        {"Relative Importance": ["mean", "count"], "Rank": ["sum", "mean"]}
    )
    feature_metrics = feature_metrics.set_axis(
        [rename_dict[x, y] for x, y in feature_metrics.columns], axis=1
    )
    feature_metrics = feature_metrics.loc[
        feature_metrics["Occurrence Count"] >= min_occurrences
    ]
    feature_metrics = feature_metrics.reset_index()
    feature_metrics = feature_metrics.sort_values(
        by=["Relative Importance", "Mean rank percentile"], ascending=False
    )
    return feature_metrics


def get_combined_proba_plot(
    results: Dict[str, Any], fig_name: str, figsize: Optional[Tuple[int]] = None
) -> plt.Figure:
    """Get figure of probabilities over the test period.

    :param results:  Results dictionary containing keys 'Estimators' & 'y_test'. Estimators need to be prefit to X_train
    :type results: Dict[str, Any]
    :param fig_name: [description]
    :type fig_name: str
    :param figsize: Fig size to plot, defaults to (30,10).
    :return: [description]
    :rtype: plt.Figure
    """
    if figsize is None:
        figsize = (30, 10)
    class_probs = np.concatenate(
        [
            get_class_probs(results, fold_index)
            for fold_index in range(len(results["y_test"]))
        ]
    )
    y_test = pd.concat(results["y_test"])
    fig = get_proba_plot(y_test, class_probs, fig_name, figsize)
    return fig


def get_combined_pred(
    results: Dict[str, Any], fig_name: str, figsize: Optional[Tuple[int]] = None
) -> plt.figure:
    """Get combined prediction plot by concatenating all y_test periods.

    NOTE: No handling of gaps in dates, assumes continuous date periods in y_test folds.

    :param results: Results dictionary containing keys 'y_pred' & 'y_test'.
    :type results: Dict[str, Any]
    :param fig_name: Title of figure to use.
    :type fig_name: str
    :param figsize: Fig size to plot, defaults to (30,10).
    :return: Figure object of entire y_pred & y_test vs. Time.
    :rtype: plt.figure
    """
    if figsize is None:
        figsize = (30, 10)
    y_test = pd.concat(results["y_test"])
    y_pred = np.concatenate(results["y_pred"])
    fig = get_prediction_plot(y_test, y_pred, fig_name, figsize)
    return fig


def get_proba_plot(
    y_test: pd.Series,
    class_probs: np.array,
    fig_title: str,
    figsize: Optional[Tuple[int]] = None,
) -> plt.Figure:
    """Plot prediction probability (predict_proba) and actual labels overlaid vs. date.

    :param y_test: True labels.
    :type y_test: pd.Series
    :param class_probs: Output from estimator.predict_proba(X_test)
    :type class_probs: np.array
    :param run_name: Name of run in azure mlflow.
    :type run_name: str
    :param figsize: Fig size to use, defaults to (12,8).
    :return: Figure object.
    :rtype: plt.Figure
    """
    if figsize is None:
        figsize = (12, 8)
    rec_probs = class_probs[:, 1]
    rec_probs_series = pd.Series(rec_probs, index=y_test.index)
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=y_test.index, y=y_test, ax=ax)
    sns.lineplot(x=rec_probs_series.index, y=rec_probs_series, ax=ax)
    sns.lineplot(x=y_test.index, y=0.5, ax=ax)
    ax.lines[2].set_linestyle("--")
    ax.lines[2].set_color("red")
    ax.set_title(fig_title)
    return fig


def get_importance_df(estimator: BaseEstimator, X: pd.DataFrame) -> pd.DataFrame:
    """Get dataframe of relative feature importances.

    :param estimator: Estimator to get importanes from.
    :param X: Dataframe with features matching shape of X_train that estimator was fit to.
    :return: Dataframe of feature importances.
    :rtype: pd.DataFrame
    """
    sfm = SelectFromModel(estimator, threshold=-np.inf, prefit=True)
    important_features = X.iloc[:, sfm.get_support()].columns
    if hasattr(estimator, "coef_"):
        relative_importances = (
            np.abs(estimator.coef_[0][sfm.get_support(indices=True)])
            / np.abs(estimator.coef_[0]).sum()
        )
        # Normal-based criteron for selecting important features per Brank et al. (2002)
        # We are seeking features that maximize the margin between decision boundary and support vectors (data points).
    elif hasattr(estimator, "feature_importances_"):
        relative_importances = estimator.feature_importances_
    importances_df = pd.DataFrame(
        {"Feature": important_features, "Relative Importance": relative_importances}
    )
    return importances_df


def get_feature_ranks(importance_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Add 'Rank' feature to each df in importances_dfs based on relative importance rank per fold.

    :param importance_dfs: Dataframe as returned by get_importance_df.
    :type importance_dfs: pd.DataFrame
    :return: Importances df with Rank feature added.
    :rtype: pd.DataFrame
    """
    importance_dfs = copy.deepcopy(importance_dfs)
    for df in importance_dfs:
        df.sort_values(by="Relative Importance", ascending=True, inplace=True)
        df["Rank"] = (np.arange(len(df)) + 1) / df.shape[0]
    return importance_dfs


def plot_importance_df(
    fig_title: str,
    importances_df: pd.DataFrame,
    num_features: int = 10,
    figsize: Optional[Tuple[int]] = None,
) -> plt.Figure:
    """Plot importance dataframe in bar chart.

    :param name: Market name, eg. US
    :type name: str
    :param importance_df: Importance df from get_importance_df()
    :type importance_df: pd.DataFrame
    :num_features: Number of features to plot, defaults to 10.
    :param figsize: Tuple of figsize, in pixels in form (width, height). Defaults to 1200, 800.
    :return: Figure of importances.
    :rtype: plt.Figure
    """
    if figsize is None:
        figsize = (1200, 800)
    importances_df = importances_df.sort_values(
        by="Relative Importance", ascending=False
    )[:num_features]
    fig = px.bar(
        importances_df[::-1],
        x="Relative Importance",
        y="Feature",
        template=settings.PLOTLY_TEMPLATE,
        title=fig_title,
    )
    # NOTE: We reverse order of importances_df here to ensure top features are plotted at top of bar chart.
    fig.update_layout(
        title_x=0.5,
        margin_l=400,
        yaxis_tickfont_size=14,
        width=figsize[0],
        height=figsize[1],
        autosize=False,
    )
    return fig


def line_break_feature(feature_text: str) -> str:
    """Helper method for adding line breaks to features for readability

    :param feature_text: String to split into new lines each 50 chars
    :return: String split into multiple lines.
    """
    if len(feature_text) > 50:
        text_comp = feature_text.split(" ")
        char_counter = 0
        for i in range(len(text_comp)):
            comp = text_comp[i]
            char_counter += len(comp)
            if char_counter > 50:
                text_comp[i] = comp + "<br>"
                char_counter = 0
        feature_text = " ".join(text_comp)
    return feature_text


def get_class_probs(results: Dict[str, Any], fold_index: int) -> np.ndarray:
    """Return class probabilities. Fit's a CalibratedClassifierCV if estimator does not implement predict_proba().
    NOTE: Requires results dict with keys 'Estimators', 'X_train', 'X_test', and 'y_train'. Estimators should be prefit
    to X_train, y_train.

    :param results: Results dict, similar to that outputted by SklearnDojo objects.
    :type results: Dict[str, Any]
    :param fold_index: Fold index to access data elements from.
    :type fold_index: int
    :return: Class probabilities for X_test.
    :rtype: np.ndarray
    """
    base_estimator = results["Estimators"][fold_index]
    X_train = results["X_train"][fold_index]
    y_train = results["y_train"][fold_index]
    X_test = results["X_test"][fold_index]
    # TODO: Remove hasattr if we want to use calibrated classifiers for other models per
    # sep's comments.
    if not hasattr(base_estimator, "predict_proba"):
        estimator = CalibratedClassifierCV(base_estimator=base_estimator, cv="prefit")
        estimator.fit(X_train, y_train)
    else:
        estimator = base_estimator
    class_probs = estimator.predict_proba(X_test)
    return class_probs
