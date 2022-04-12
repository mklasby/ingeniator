import pandas as pd
import os
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils import resample
import logging
import datetime
from ingeniator.feature_selection.feature_selection_transformer import (
    FeatureSelectionTransformer,
)
from sklearn.feature_selection import SelectFromModel


class LOOCV_Wrapper(BaseEstimator):
    """
    TODO: Refactor to remove all extraneous feature selection methods
    TODO: remove X, y, label_col params from init.

    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: BaseEstimator,
        pipeline: Pipeline = None,
        perform_grid_search: bool = False,
        param_grid: dict = None,
        log_dir: str = None,
        log_file_name: str = None,
        copy: bool = True,
        label_col: str = None,
        balance_classes: bool = False,
        scoring: str = "accuracy",
        verbose: int = 2,
        single_label_upsample: str = None,
        n_samples: int = None,
        encode_labels: bool = False,
        cv: int = None,
        save_feature_importance: bool = False,
        random_state=42,
    ):

        self.X = X
        self.y = y
        self.estimator = estimator
        self.pipeline = pipeline
        self.perform_grid_search = perform_grid_search
        self.param_grid = param_grid
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.copy = copy
        self.label_col = label_col
        self.balance_classes = balance_classes
        self.scoring = scoring
        self.verbose = verbose
        self.n_samples = n_samples
        self.single_label_upsample = single_label_upsample
        self.cv = cv
        self.encode_labels = encode_labels
        self.save_feature_importance = save_feature_importance
        self.random_state = random_state
        self.__logger = logging.getLogger(__file__)

    def _reset(self):
        if hasattr(self, "pipe_"):
            del self.pipe_
        if hasattr(self, "results_df_"):
            del self.results_df_
        if hasattr(self, "y_pred_"):
            del self.y_pred_
            del self.y_true_
            del self.accuracy_scores_
        if hasattr(self, "importances_"):
            del self.importances_
        if hasattr(self, "best_params_"):
            del self.best_params_
        if hasattr(self, "le"):
            del self.le

    def _pipeline_transform(self, X_train, X_test, y_train=None) -> tuple:
        self.pipe_ = clone(self.pipeline)
        X_train = self.pipe_.fit_transform(X_train, y_train)
        X_test = self.pipe_.transform(X_test)
        return X_train, X_test

    def _get_best_grid_search_estimator(
        self, X_train: pd.DataFrame, y_train: pd.Series, estimator: BaseEstimator
    ) -> BaseEstimator:
        if self.cv is None:
            cv = y_train.groupby(y_train).count().min()
            if cv == 1:
                cv = 5
        else:
            cv = self.cv
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            verbose=self.verbose,
            cv=cv,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        self.best_params_.append(grid_search.best_params_)
        self.accuracy_scores_.append(grid_search.best_score_)
        return grid_search.best_estimator_

    def _upsample_label(
        self, df: pd.DataFrame, label: str, n_samples: int
    ) -> pd.DataFrame:
        label_subset = df[df[self.label_col] == label]
        if len(label_subset) == n_samples:
            return label_subset
        else:
            return resample(
                label_subset,
                replace=True,
                n_samples=n_samples,
                random_state=RANDOM_STATE,
            )

    def _get_balanced_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_samples: int = None,
        single_label: str = None,
    ) -> tuple:
        df = X_train.join(y_train)
        dfs = []
        if n_samples is None:
            n_samples = df.groupby(self.label_col).count().iloc[:, 0].max()
        if single_label is not None:
            dfs.append(df[df[self.label_col] != single_label])
            dfs.append(self._upsample_label(df, single_label, n_samples))
        else:
            labels = df[self.label_col].unique()
            for label in labels:
                dfs.append(self._upsample_label(df, label, n_samples))
        df_upsampled = pd.concat(dfs)
        return df_upsampled.drop(columns=[y_train.name]), df_upsampled[y_train.name]

    def _get_estimator_name(self, estimator: BaseEstimator):
        estimator_name = str(type(estimator)).split(".")[-1][:-2]
        return estimator_name

    def _save_combined_feature_importances(self):
        feature_importances_df = self.importances_.mean(axis=1).rename("Importance")
        save_path = os.path.join(
            self.log_dir, self.log_file_name + "_feature_importances.csv"
        )
        feature_importances_df.to_csv(save_path, index=True)
        self.__logger.info(f"Feature importances saved to: {save_path}")
        return

    def _check_features(self, X_train: pd.DataFrame, importance_filter):
        for feature in X_train.columns:
            if feature not in importance_filter.important_features_.values.tolist():
                raise ValueError(f"{feature} missing from X_train!")
        return "Features OK!"

    def _get_feature_importances(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        counter: int,
        estimator: BaseEstimator,
    ) -> pd.DataFrame:
        sfm = FeatureSelectionTransformer(
            transformer=SelectFromModel(estimator=estimator)
        )

        sfm.fit(X_train, y_train)
        feature_importances = sfm.get_feature_importances()
        self.importances_ = self.importances_.join(
            feature_importances, how="outer", rsuffix=f"_{counter}"
        )
        return X_train

    def fit(self, X=None, y=None):
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y
        self._fit()

    def _fit(self):
        self._reset()
        loo = LeaveOneOut()
        self.y_true_ = []
        self.y_pred_ = []
        self.importances_ = pd.DataFrame({"Feature": self.X.columns}).set_index(
            "Feature"
        )
        self.best_params_ = []
        self.accuracy_scores_ = []
        X = self.X.copy(deep=True)
        y = self.y.copy(deep=True)
        counter = 0
        est_name = self._get_estimator_name(self.estimator)
        self.__logger.info(f"Fitting LOOCV models for {est_name}")

        if self.encode_labels:
            self.le = LabelEncoder()
            y = pd.Series(
                self.le.fit_transform(y), name=self.y.name, index=self.y.index
            )

        for train_idx, test_idx in loo.split(X):
            counter += 1
            self.__logger.info(f"Fitting model {counter} of {len(X)}")
            X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]
            estimator = clone(self.estimator)

            if self.balance_classes:  # TODO: Use Imbalance learn instead here.
                self.__logger.info(f"Balancing classes for iteration {counter}...")
                X_train, y_train = self._get_balanced_train(
                    X_train,
                    y_train,
                    n_samples=self.n_samples,
                    single_label=self.single_label_upsample,
                )
                self.__logger.info(
                    f"Upsampled breakdown: {y_train.groupby(y_train).count()}"
                )

            if self.pipeline is not None:
                self.__logger.info(f"Fitting pipe to iteration {counter}...")
                self.__logger.info(f"X_train shape before pipeline: {X_train.shape}")
                X_train, X_test = self._pipeline_transform(X_train, X_test, y_train)
                self.__logger.info(f"X_train shape after pipeline: {X_train.shape}")

            if self.perform_grid_search:
                self.__logger.info(f"Performing grid search for iteration {counter}")
                start = datetime.datetime.now()
                estimator = self._get_best_grid_search_estimator(
                    X_train, y_train, estimator
                )
                end = datetime.datetime.now()
                self.__logger.info(
                    f"Grid search for iteration {counter} completed in: {end-start}..."
                )

            estimator.fit(X_train, y_train)
            self.y_pred_.append(estimator.predict(X_test)[0])
            self.y_true_.append(y_test.values[0])

            if self.save_feature_importance:
                self.__logger.info(
                    f"Calculating top features for iteration {counter}..."
                )
                X_train = self._get_feature_importances(
                    X_train, y_train, counter, estimator
                )

        self.__logger.info(f"LOOCV model fit complete for {est_name}")
        acc = accuracy_score(self.y_true_, self.y_pred_)
        self.__logger.info(f"Overall accuracy for {est_name}: {acc}")

        if self.perform_grid_search:
            grid_results = pd.DataFrame(
                {"Best_Params": self.best_params_, "Scores": self.accuracy_scores_}
            )
            fname = os.path.join(
                self.log_dir, self.log_file_name + "_grid_search" + ".csv"
            )
            grid_results.to_csv(fname)
            self.__logger.info(f"Saved grid search results to {fname}")

        if self.save_feature_importance:
            self._save_combined_feature_importances()

        if self.encode_labels:
            self.y_pred_ = self.le.inverse_transform(self.y_pred_)
            self.y_true_ = self.le.inverse_transform(self.y_true_)

        self.results_df_ = pd.DataFrame(
            {"y_true": self.y_true_, "y_pred": self.y_pred_}
        ).set_index(X.index)
        fname = os.path.join(self.log_dir, self.log_file_name + "_results.csv")
        self.results_df_.to_csv(fname)
        self.__logger.info(f"Saved results to {fname}")


if __name__ == "__main__":
    from ingeniator.utils import toy_feature_selection_dataset
    from sklearn.pipeline import make_pipeline
    from ingeniator.feature_selection.sklearn_transformer_wrapper import (
        SklearnTransformerWrapper,
    )
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler, PowerTransformer

    X, y = toy_feature_selection_dataset(classification_targets=True)
    estimator = LinearSVC(
        C=0.1,
        fit_intercept=True,
        max_iter=100000000,
        tol=1e-10,
        random_state=42,
        class_weight="balanced",
    )
    pipe = make_pipeline(
        SklearnTransformerWrapper(transformer=StandardScaler()),
        SklearnTransformerWrapper(transformer=PowerTransformer(standardize=False)),
        # FeatureSelectionTransformer(transformer=RFE(estimator=clone(estimator))),
        FeatureSelectionTransformer(
            transformer=SelectFromModel(estimator=clone(estimator))
        ),
    )
    loocv = LOOCV_Wrapper(
        X,
        y,
        estimator,
        pipeline=pipe,
        perform_grid_search=False,
        label_col="label",
        log_file_name=None,
        log_dir="logs",
        balance_classes=False,
        scoring="f1_score",
        verbose=2,
        n_samples=None,
        single_label_upsample=None,
        cv=None,
        select_features=True,
    )
    loocv.fit(X, y)
