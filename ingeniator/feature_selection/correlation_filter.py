"""Transformer to drop correlated columns.

Typical usage example:

    cf = CorrelationFilter(threshold=0.8, columns_to_ignore=['label_col'])
    cf.fit(X_train)
    X_test = cf.transform(X_test)

    TODO: Refactor to use new parent class.
"""
from __future__ import annotations  # noqa: E501
from ingeniator.feature_selection.dataframe_transformer_wrapper import (
    DataFrameTransformerWrapper,
)
import numpy as np
import pandas as pd
import logging
from typing import Optional, List


class CorrelationFilter(DataFrameTransformerWrapper):
    """Custom transformer to remove columns that are correlated. Only one columns is dropped for each correlated pair.

    :param threshold: Threshold at which a column is considered correlated.
        (eg., 0.9 means drop all columns with >= 90% correlation), defaults to 0.9.
    :type threshold: float, optional
    :param columns_to_ignore: List of columns names to ignore. These columns will not be considered when
        calculating correlations nor when dropping columns. Defaults to None.
    :type columns_to_ignore: list, optional
    :param copy: A bool indicating if the dataframes passed to transform should be copied before being transformed.
        Defaults to True.
    :type copy: bool, optional
    """

    def __init__(
        self,
        threshold: float = 0.9,
        ignore_features: Optional[List] = None,
        copy: bool = True,
        regex_feature_selector: Optional[List] = None,
    ):
        """CF Constructor"""
        super().__init__(
            copy=copy,
            ignore_features=ignore_features,
            retain_col_names=False,
            keep_feature_order=False,
            regex_feature_selector=regex_feature_selector,
        )

        self.threshold = threshold

    def _reset(self):
        """Private method to reset fit parameters."""
        super()._reset()
        if hasattr(self, "to_drop_"):
            del self.to_drop_

    def _fit(self, X: pd.DataFrame, y=None):
        """Private method to implement fit."""
        self.to_drop_ = []
        corrdata = X.copy(deep=True)
        corr_matrix = corrdata.corr().abs()
        upper_diagonal = pd.DataFrame(
            np.triu(corr_matrix, k=1),
            columns=corr_matrix.columns,
            index=corr_matrix.index,
        )
        self.to_drop_ = [
            column
            for column in upper_diagonal.columns
            if any(upper_diagonal[column] >= self.threshold)
        ]
        logging.info(f"Found {len(self.to_drop_)} correlated features to drop")

    def _transform(self, X: pd.DataFrame, y=None, copy: bool = None) -> pd.DataFrame:
        """Transforms X by dropping one of each correlated column pair.

        :param X: A dataframe to transform.
        :type X: pd.DataFrame
        :param y: Ignored. Left as a parameter to maintain compatibility with existing fit_transform() interfaces.
            Defaults to None.
        :param copy: Optional parameter to use instead of self.copy. Defaults to None.
        :type copy: bool, optional
        :return: Transformed copy of X.
        :rtype: pd.DataFrame
        :raises: NotFittedError: If the transformer has not been fit.
        """
        X = X.drop(columns=self.to_drop_)
        logging.debug(
            f"Dropped {self.to_drop_}\n total of {len(self.to_drop_)} Features"
        )
        return X
