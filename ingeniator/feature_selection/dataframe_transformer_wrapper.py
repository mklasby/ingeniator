"""Common parent for custom transformers intended to wrap pd.DataFrame objects.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Optional, List
from copy import deepcopy
import logging
import re


class DataFrameTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    :param copy: If True, copy all dataframes before transforing, defaults to True.
    :type copy: bool, optional
    :param ignore_features: List of features to ignore when calling .fit and
            .transform. Defaults to None.
    :type ignore_features: Optional[List[str]], optional
    :param retain_col_names: If True, columns order and names are guaranteed. Only
            suitable for transformers that do not rename columns. Defaults to True.
    :type retain_col_names: bool, optional
    :param append_cols: If true, append transformed columns to original df.
        Defaults to False
    :type append_cols: bool, optional
    :param column_suffix: If not None, adds suffix to transformed columns.
        Defaults to None
    :type column_suffix: Optional[str], optional
    :param keep_feature_order: If true, original column order is strictly maintained.
        Not suitable for transformers that change the shape of the df. Defaults to True.
    :type keep_feature_order: bool, optional
    :param regex_feature_selector: Regex match strings to select columns to apply
        transformer to. Appends any columns that do not match regex strings to
        self.ignore_features, defaults to None
    :type regex_feature_selector: Optional[List[str]], optional
    """

    def __init__(
        self,
        copy: bool = True,
        ignore_features: Optional[List[str]] = None,
        retain_col_names: bool = True,
        append_cols: bool = False,
        column_suffix: Optional[str] = None,
        keep_feature_order: bool = True,
        regex_feature_selector: Optional[List[str]] = None,
    ) -> DataFrameTransformerWrapper:
        """Constructor. Should be invoked from children to set these params."""

        self.copy = copy
        self.ignore_features = ignore_features
        self.retain_col_names = retain_col_names
        self.append_cols = append_cols
        self.column_suffix = column_suffix
        self.keep_feature_order = keep_feature_order
        self.regex_feature_selector = regex_feature_selector
        self.logger = logging.getLogger(__file__)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> DataFrameTransformerWrapper:
        """Public method to fit transformer to X.

        :param X: A dataframe to fit to.
        :type X: pd.DataFrame
        :param y: Ignored for most transformers, except feature selection and similar.
        :type y: Optional[pd.Series]
        :return: This transformer, fitted to X.
        :rtype: DataFrameTransformerWrapper
        """
        self._validate_params()
        self._reset()
        X = self._prepare_for_fit(X)
        self._fit(X, y)  # Must be implemented in child classes
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        copy: Optional[bool] = None,
    ) -> DataFrameTransformerWrapper:
        """Public method to transform X.

        :param X: A dataframe to transform.
        :type X: pd.DataFrame
        :param y: A target series, defaults to None.
        :type y: Optional[pd.Series]
        :param copy: Override for self.copy, defaults to None
        :type copy: bool, optional
        :return: Transformed dataframe.
        :rtype: DataFrameTransformerWrapper
        """
        logging.debug(f"X shape before transform: {X.shape}")
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        if copy:
            X = X.copy(deep=True)
        X = self._prepare_for_transform(X)
        X_transformed = self._transform(X, y)  # Must be implemented in child classes
        X_transformed = self._prepare_for_return(X, X_transformed)
        self.logger.debug(f"X shape after transform: {X_transformed.shape}")
        return X_transformed

    def _validate_params(self) -> None:
        self.ignore_features = deepcopy(self.ignore_features)
        return

    def _get_ignored_features(self, X: pd.DataFrame) -> List[str]:
        """Private method to determine columns to remove from X based on argument.

        :param X: Dataframe to get columns from.
        :type X: pd.DataFrame
        :return: List of features to drop
        :rtype: List[str]
        """
        ignored_feature_list = X[[feature for feature in X.columns if feature in self.ignore_features]].columns
        return ignored_feature_list

    def _prepare_for_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """Private method to prepare X for fitting.

        :param X: Dataframe to prepare for fitting.
        :type X: pd.DataFrame
        :return: Dataframe prepared for fitting.
        :rtype: pd.DataFrame
        """
        if self.keep_feature_order:
            self.original_col_order_ = X.columns

        # TODO: Create helper to deal with regex + ignore features list seperately
        if self.regex_feature_selector is not None:
            self._find_regex_matches(X)

        if self.ignore_features is not None:
            X = X.copy()
            ignored_feature_list = self._get_ignored_features(X)
            X = X.drop(columns=ignored_feature_list)
        self.columns_before_transform_ = X.columns
        return X

    def _find_regex_matches(self, X: pd.DataFrame) -> None:
        """Finds columns that match self.regex_feature_selector strings.

        Any column that does not match is added to ignore_features.

        :param X: Dataframe to find matches in.
        :type X: pd.DataFrame
        """
        if self.ignore_features is None:
            self.ignore_features = []
        regex_matches = []
        for regex in self.regex_feature_selector:
            regex_matches += [feature for feature in X.columns if re.match(regex, feature)]
        self.ignore_features += [feature for feature in X.columns if feature not in regex_matches]
        return

    def _prepare_for_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Private method to prepare X for transforming.

        :param X: Dataframe to prepare for transforming.
        :type X: pd.DataFrame
        :return: Dataframe prepared for transforming.
        :rtype: pd.DataFrame
        """
        if self.ignore_features is not None:
            X = X.copy()
            ignored_feature_list = self._get_ignored_features(X)
            self.ignored_features_ = X[ignored_feature_list]
            X = X.drop(columns=ignored_feature_list)
        self.columns_before_transform_ = X.columns
        self.index_before_transform_ = X.index
        return X

    def _prepare_for_return(self, X: pd.DataFrame, X_transformed: np.array) -> pd.DataFrame:
        """Private method to prepare X for returning.

        Resolves ignored columns and column order.

        :param X: Dataframe to prepare for transforming.
        :type X: pd.DataFrame
        :param X_transformed: Transformed np.array
        :type X_transformed: np.array
        :return: Dataframe to prepare for transforming.
        :rtype: pd.DataFrame
        """
        X_transformed = self._create_dataframe(X_transformed)
        if self.ignore_features is not None:
            X_transformed = pd.merge(X_transformed, self.ignored_features_, right_index=True, left_index=True)
        if self.keep_feature_order:
            if not self.retain_col_names:
                raise Exception("retain_col_names must be True to keep_feature_order.")
            X_transformed = X_transformed[self.original_col_order_]

        if self.column_suffix is not None:
            X_transformed = self._apply_column_suffix(X_transformed)

        if self.append_cols:
            if self.column_suffix is None:
                self.column_suffix = "_transformed"
                # We must suffix we if append but no suffix was declared
                X_transformed = self._apply_column_suffix(X_transformed)
            X_transformed = pd.merge(X, X_transformed, left_index=True, right_index=True)

        return X_transformed

    def _reset(self):
        """Resets state of fit transformer."""
        if hasattr(self, "ignored_features_"):
            del self.ignored_features_
        if hasattr(self, "original_col_order_"):
            del self.original_col_order_
        if hasattr(self, "columns_before_transform_"):
            del self.columns_before_transform_
            del self.index_before_transform_

    def _create_dataframe(self, X_transformed: np.array) -> pd.DataFrame:
        """Generates dataframes from transformed np.array by resolving column options.

        :param X: Dataframe immediately prior to transforming
            (after _prepare_for_transform)
        :type X: pd.DataFrame
        :param X_transformed: Transformed np.array
        :type X_transformed: np.array
        :return: Dataframe of np.array with column names restored.
        :rtype: pd.DataFrame
        """
        if isinstance(X_transformed, pd.DataFrame):
            return X_transformed  # If we dealt with conversion in child class, we skip.
        columns = None
        index = self.index_before_transform_
        if self.retain_col_names:
            columns = self.columns_before_transform_
        X_transformed = pd.DataFrame(X_transformed, index=index, columns=columns)
        return X_transformed

    def _apply_column_suffix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies suffix to columns in X, excpet any ignored_features.

        :param X: Dataframe to suffix columns of.
        :type X: pd.DataFrame
        :return: Dataframe with suffix on column names that are not ignored.
        :rtype: pd.DataFrame
        """
        if self.ignore_features is None:
            col_map = {col: f"{col}{self.column_suffix}" for col in X.columns}
        else:
            col_map = {col: f"{col}{self.column_suffix}" for col in X.columns if col not in self.ignore_features}
        X = X.rename(columns=col_map)
        return X
