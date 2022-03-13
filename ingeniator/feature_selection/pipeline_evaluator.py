import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import logging
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import get_scorer
from itertools import product
import warnings
import re


class PipelineEvaluator(BaseEstimator):
    """PipelineEvaluator evaluates several pipelines and provides results on best
    performing candidate."""

    def __init__(
        self,
        pipelines: list,
        estimator: BaseEstimator,
        metric: str = "accuracy",
        test_size: float = 0.33,
        random_state: float = 42,  # TODO: Move to constants
        include_dummy_case: bool = True,
    ):
        self.metric = metric  # https://scikit-learn.org/stable/modules/model_evaluation.html
        self.pipelines = pipelines
        self.estimator = estimator
        self.scorer = get_scorer(metric)
        self.test_size = test_size
        self.random_state = random_state
        self.include_dummy_case = include_dummy_case
        self.logger = logging.getLogger(__file__)

    def _reset(self):
        """Private method to reset fit parameters."""
        if hasattr(self, "names_"):
            del self.scores_
            del self.names_
            del self.pipe_dict_

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._fit(X, y)

    def get_best_pipeline(self):
        if self.pipe_dict_ is None:
            raise NotFittedError("Not fit!")
        results = self.get_results().sort_values("Score", ascending=False)
        best_pipe = results.iloc[0]["Pipeline_name"]
        return self.pipe_dict_[best_pipe]

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        self._reset()
        self.scores_ = []
        self.names_ = []
        self.pipe_dict_ = {}

        if self.include_dummy_case:
            self._process_dummy_case(X, y)

        for pipe in self.pipelines:
            self._process_pipe(pipe, X, y)
        self.logger.info(f"Evaluation complete on {len(self.pipelines)} pipelines.")

    def _process_dummy_case(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy(deep=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        pipe_name = "No_pipeline"
        self.logger.info(f"Evaluating {pipe_name} pipeline...")
        self._score_pipeline(X_train, X_test, y_train, y_test, pipe_name)

    def _process_pipe(self, pipe: Pipeline, X: pd.DataFrame, y: pd.Series):
        X = X.copy(deep=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        pipe_name = self._get_pipeline_name(pipe)
        self.logger.info(f"Evaluating {pipe_name} pipeline...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                X_train = pipe.fit_transform(X_train, y_train)
                X_test = pipe.transform(X_test)
                self.pipe_dict_[pipe_name] = pipe
                self._score_pipeline(X_train, X_test, y_train, y_test, pipe_name)
            except Exception as e:
                self.logger.warning(f"Error in pipeline {pipe_name}. Skipping...")
                self.logger.debug(f"Full stack trace:\n{e}")

    def _get_pipeline_name(self, pipe: Pipeline) -> str:
        pipe_name = ""
        for name, transformer in pipe.steps:
            pipe_name = pipe_name + name + "-"
        pipe_name = pipe_name[:-1]
        return pipe_name

    def _score_pipeline(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, pipe_name: str,
    ):
        estimator = clone(self.estimator)
        estimator.fit(X_train, y_train)
        self.scores_.append(self.scorer(estimator, X_test, y_test))
        self.names_.append(pipe_name)

    def get_results(self) -> pd.DataFrame:
        check_is_fitted(self)
        df = pd.DataFrame({"Pipeline_name": self.names_, "Score": self.scores_})
        return df

    def get_fitted_pipes(self) -> dict:
        check_is_fitted(self)
        return self.pipe_dict_


# Following functions are intended for exhaustive pipeline searches
def get_type_name(obj):
    name = type(obj).__name__
    return name


def steps_builder(transforms: list) -> list:
    steps = [(get_type_name(x), x) for x in transforms]
    return steps


def clean_duplicate_step_names(steps) -> list:
    inventory = {}
    revised_steps = []
    for step in steps:
        if step in inventory:
            inventory[step] += 1
            step = (step[0] + str(inventory[step]), step[1])
        else:
            inventory[step] = 0
        revised_steps.append(step)
    return revised_steps


def get_pipes(scalers, filters, reducers):
    pipes = [
        Pipeline(clean_duplicate_step_names(x))
        for x in product(steps_builder(scalers), steps_builder(filters), steps_builder(reducers))
    ]
    return pipes


def get_default_pipelines(number_of_features, random_state=42):
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        MaxAbsScaler,
        Normalizer,
        PowerTransformer,
        RobustScaler,
    )
    from sklearn.feature_selection import (
        SelectKBest,
        SelectFpr,
        SelectFromModel,
        f_regression,
    )
    from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    alpha = 0.01
    threshold = 0.001
    n_components = number_of_features // 5
    k = number_of_features
    estimator = RandomForestRegressor(random_state=random_state)
    scalers = [
        None,
        StandardScaler(),
        MinMaxScaler(),
        MaxAbsScaler(),
        Normalizer(),
        PowerTransformer(),
        RobustScaler(),
    ]

    filters = [
        None,
        SelectKBest(k=k),
        SelectFpr(score_func=f_regression, alpha=alpha),
        #    RFE(clone(estimator), step=10),
        #    RFECV(clone(estimator), step=10),
        SelectFromModel(clone(estimator), threshold=threshold),
    ]

    reducers = [
        None,
        PCA(n_components=n_components, svd_solver="full", random_state=random_state),
        FastICA(n_components=n_components, random_state=random_state),
        IncrementalPCA(n_components=n_components),
        KernelPCA(n_components=n_components, random_state=random_state),
        # NMF(n_components=n_components, random_state=RANDOM_STATE)
    ]

    pipes = get_pipes(scalers, filters, reducers)
    return pipes


if __name__ == "__main__":
    from ingeniator.utils import toy_feature_selection_dataset
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score

    METRIC = "neg_mean_absolute_error"
    scorer = get_scorer(METRIC)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__file__)
    X, y = toy_feature_selection_dataset(
        classification_targets=False, num_samples=200, num_features=200, signal_features=10,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipes = get_default_pipelines(25)
    # rf = RandomForestClassifier(random_state=42)
    rf = RandomForestRegressor(random_state=42)
    evaluator = PipelineEvaluator(pipes, rf, test_size=0.2, metric=METRIC)
    evaluator.fit(X_train, y_train)
    results = evaluator.get_results()
    print(results.sort_values("Score", ascending=False))
    best_pipe = evaluator.get_best_pipeline()
    X_train = best_pipe.fit_transform(X_train, y_train)
    rf.fit(X_train, y_train)
    score = scorer(rf, best_pipe.transform(X_test), y_test)
    print(f"{METRIC} of final pipe: {score:.2f}")
