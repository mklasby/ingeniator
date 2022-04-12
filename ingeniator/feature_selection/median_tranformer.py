"""Convenience wrapper for .median() so we can add it in pipeline.
"""
from __future__ import annotations  # noqa: E501
from ingeniator.feature_selection.dataframe_transformer_wrapper import DataFrameTransformerWrapper
import numpy as np
import pandas as pd
from typing import List, Optional


class MedianTransformer(DataFrameTransformerWrapper):
    # TODO: Docstring

    def __init__(
        self,
        copy: bool = True,
        ignore_features: Optional[List[str]] = None,
        column_suffix: Optional[str] = None,
    ) -> MedianTransformer:
        super().__init__(
            copy=copy,
            ignore_features=ignore_features,
            retain_col_names=False,
            keep_feature_order=False,
            column_suffix=column_suffix,
        )

    def _reset(self):
        """Reset state of fit transformer."""
        super()._reset()
        if hasattr(self, "fitted_"):
            del self.fitted_

    def _fit(self, X: pd.DataFrame, y: None) -> MedianTransformer:
        """No fit required for this transformer."""
        self.fitted_ = True
        return self

    def _transform(self, X: pd.DataFrame, y: None) -> np.array:
        """Transforms all columns in X to a single vector of row-wise median values"""
        X = X.median(axis=1).rename("median")
        return X
