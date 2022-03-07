import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from typing import Optional
import logging
import warnings
from sklearn.metrics import get_scorer
import re

class RecursiveClusterElimination(object):

    # TODO: Convert to internally representing data as arrays and have a separate dict
    # for columns to allow for arbitrary sklearn transformers.

    # TODO: check re for neg in metric

    def __init__(
        self,
        metric: str,
        pipeline: Optional[Pipeline] = None,
        random_state: int = 42,
        cv_iterations: int = 20,
        extinction_factor: float = 0.1,
    ):
        self.metric = metric
        self.scorer = get_scorer(metric)
        self.pipeline = pipeline
        self.logger = logging.getLogger(__file__)
        self.logger.setLevel(logging.INFO)  # TODO: add debug mode
        self.random_state = random_state
        self.cv_iterations = cv_iterations
        self.extinction_factor = extinction_factor

    def fit(self, X, y, min_features: int = 50):
        X = X.copy()
        y = y.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        n_clusters = X.shape[1]
        # NOTE: We set n_clusters to number of features in initial state
        self.extinct_features = []
        results = {"features": [], "metric": [], "num_features": []}

        while (
            True
        ):  # We will recursively eliminate 10% of clusters until reaching min_features.
            ranks = self._get_feature_ranks(X_train, y_train, n_clusters)
            X_train = self._get_extinct_features(ranks, X_train)
            n_clusters = int(n_clusters * 0.9)
            result = self._test_model(X_train, X_test, y_train, y_test)
            results["num_features"].append(X_train.shape[1])
            results["features"].append(X_train.columns.to_list())
            results["metric"].append(result)
            if X_train.shape[1] <= min_features or n_clusters == 1:
                break
        self.results = results
        return results

    def get_best_features(self):
        if self.results is None:
            raise NotFittedError("Call .fit() first!")
        if re.search(r"^neg", self.metric):
            ascending = True
        else:
            ascending = False
        df = pd.DataFrame(self.results).sort_values("metric", ascending=ascending)
        return df.iloc[0].features

    def _get_train_val_split(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        X_train_val, X_val, y_train_val, y_val = train_test_split(
            X_train, y_train, test_size=0.3, random_state=np.random.randint(10000)
        )
        if self.pipeline is not None:
            X_train_val = self.pipeline.fit_transform(X_train_val)
            X_val = self.pipeline.transform(X_val)
        return X_train_val, X_val, y_train_val, y_val

    def _get_clusters(self, X_train_val: pd.DataFrame, n_clusters: int):
        if n_clusters is None:
            n_clusters = X_train_val.shape[0]
        k_means = KMeans(n_clusters=n_clusters, random_state=np.random.randint(10000))
        k_means.fit(X_train_val.T)
        clusters_df = pd.DataFrame(
            {"feature": X_train_val.columns, "cluster": k_means.labels_}
        )
        return clusters_df

    def _get_feature_ranks(self, X_train, y_train, n_clusters):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            feature_ranks = {k: [] for k in X_train.columns}
            estimator = LinearSVC(C=0.1, random_state=self.random_state)
            self.logger.info(f"Clustering features into {n_clusters} clusters...")
            for _ in range(self.cv_iterations):
                metrics_dict = {}
                features_dict = {}
                X_train_val, X_val, y_train_val, y_val = self._get_train_val_split(
                    X_train, y_train
                )
                clusters_df = self._get_clusters(X_train_val, n_clusters)
                for cluster in clusters_df["cluster"].unique():
                    features_dict[cluster] = (
                        clusters_df["feature"]
                        .loc[clusters_df["cluster"] == cluster]
                        .values
                    )
                    this_estimator = clone(estimator)
                    this_estimator.fit(X_train_val[features_dict[cluster]], y_train_val)
                    metrics_dict[cluster] = self.scorer(
                        this_estimator, X_val[features_dict[cluster]], y_val
                    )
                sorted_results = {
                    k: v
                    for k, v in sorted(
                        metrics_dict.items(), key=lambda x: x[1], reverse=True
                    )
                }
                ranked_clusters = dict(enumerate(sorted_results.keys()))
                for cluster, features in features_dict.items():
                    for feature in features:
                        rank = [k for k, v in ranked_clusters.items() if v == cluster][
                            0
                        ]
                        feature_ranks[feature].append(rank)
                ranks = pd.DataFrame(feature_ranks)
        return ranks

    def _get_extinct_features(self, feature_ranks, X_train):
        bottom_features = (
            feature_ranks.mean(axis=0)
            .sort_values(ascending=True)
            .iloc[
                int(-feature_ranks.shape[1] * (self.extinction_factor)) :  # noqa E203
            ]
            .index.to_list()
        )
        self.extinct_features += bottom_features
        return X_train.drop(columns=bottom_features)

    def _test_model(self, X_train, X_test, y_train, y_test):
        self.logger.info(f"Testing model on {X_train.shape[1]} features...")
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_test = X_test.drop(columns=self.extinct_features)
        if self.pipeline is not None:
            X_train = self.pipeline.fit_transform(X_train)
            X_test = self.pipeline.transform(X_test)
        estimator = LinearSVC(C=0.1, random_state=self.random_state)
        estimator.fit(X_train, y_train)
        this_metric = self.scorer(estimator, X_test, y_test)
        self.logger.info(f"{self.metric} score: {this_metric}")
        return this_metric


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    from ingeniator.utils import toy_feature_selection_dataset
    from ingeniator.feature_selection.sklearn_transformer_wrapper import (
        SklearnTransformerWrapper,
    )
    import pickle

    # TODO: Move to test suite
    logging.basicConfig()
    X, y = toy_feature_selection_dataset(
        classification_targets=True,
        num_samples=1000,
        num_features=200,
        signal_features=10,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipe = make_pipeline(SklearnTransformerWrapper(transformer=StandardScaler()))
    rce = RecursiveClusterElimination(metric=accuracy_score, pipeline=pipe)
    result = rce.rce_svc(X_train, y_train, min_features=25)
    with open("result.pkl", "wb") as handle:
        pickle.dump(result, handle)
