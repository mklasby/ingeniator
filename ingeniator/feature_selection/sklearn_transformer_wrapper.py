"""TODO
"""
from __future__ import annotations  # noqa: E501
from ingeniator.feature_selection.dataframe_transformer_wrapper import (
    DataFrameTransformerWrapper,
)
import pandas as pd
from typing import List, Optional
from sklearn.base import TransformerMixin, clone


class SklearnTransformerWrapper(DataFrameTransformerWrapper):
    """Class to wrap any sklearn transformer that preserves column names.

    :param transformer: [description]
    :type transformer: TransformerMixin
    :param copy: [description], defaults to True
    :type copy: bool, optional
    :param ignore_features: [description], defaults to None
    :type ignore_features: Optional[List[str]], optional
    :param maintain_feature_order: [description], defaults to True
    :type maintain_feature_order: bool, optional
    :return: [description]
    :rtype: SklearnTransformerWrapper
    """

    def __init__(
        self,
        transformer: TransformerMixin,
        copy: bool = True,
        ignore_features: Optional[List[str]] = None,
        retain_col_names: bool = True,
        append_cols: bool = False,
        column_suffix: Optional[str] = None,
        keep_feature_order: bool = True,
    ) -> SklearnTransformerWrapper:
        """TODO"""
        super().__init__(
            copy=copy,
            ignore_features=ignore_features,
            retain_col_names=retain_col_names,
            append_cols=append_cols,
            column_suffix=column_suffix,
            keep_feature_order=keep_feature_order,
        )
        self.transformer = transformer

    def _reset(self):
        """Reset state of fit transformer."""
        super()._reset()  # Must be called as we are overriding.
        if hasattr(self, "fit_transformer_"):
            del self.fit_transformer_

    def _fit(self, X: pd.DataFrame, y: None) -> SklearnTransformerWrapper:
        """Fit to X, y using self.transformer."""
        self.fit_transformer_ = clone(self.transformer)
        # Maintains original transformer in the event of refitting.
        self.fit_transformer_.fit(X, y)
        return self

    def _transform(self, X: pd.DataFrame, y: None) -> pd.DataFrame:
        """Transform X using self.transformer"""
        X = self.fit_transformer_.transform(X)
        return X
