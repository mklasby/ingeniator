"""TODO Move to wrapper module?
"""
from __future__ import annotations  # noqa: E501
from ingeniator.feature_selection.dataframe_transformer_wrapper import (
    DataFrameTransformerWrapper,
)
import pandas as pd
from typing import List, Optional, Union
import numpy as np
from sklearn.base import TransformerMixin, clone
import logging
from sklearn.feature_selection._base import _get_feature_importances
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectFromModel


class FeatureSelectionTransformer(DataFrameTransformerWrapper):
    """Class to wrap any sklearn feature selection transformer that preserves column
    names.

    :param transformer: Object implementing transformermixin interface to wrap.
    :type transformer: TransformerMixin
    :param copy: If true, copy feature matrix before transforming, defaults to True
    :type copy: bool, optional
    :param ignore_features: Features to exclude from consideration of importance,
        defaults to None
    :type ignore_features: Optional[List[str]], optional
    :param column_suffix: If not None, adds suffix to transformed columns.
        Defaults to None
    :type column_suffix: Optional[str], optional
    """

    def __init__(
        self,
        transformer: TransformerMixin,
        copy: bool = True,
        ignore_features: Optional[List[str]] = None,
        column_suffix: Optional[str] = None,
    ) -> FeatureSelectionTransformer:
        """Constructor for class"""
        super().__init__(
            copy=copy,
            ignore_features=ignore_features,
            retain_col_names=True,
            append_cols=False,
            column_suffix=column_suffix,
            keep_feature_order=False,
        )
        self.transformer = transformer
        self.logger = logging.getLogger(__file__)

    def _reset(self):
        """Reset state of fit transformer."""
        super()._reset()  # Must be called as we are overriding.
        if hasattr(self, "fit_transformer_"):
            del self.fit_transformer_

    def _fit(self, X: pd.DataFrame, y: None) -> FeatureSelectionTransformer:
        """Fit to X, y using self.transformer."""
        self.fit_transformer_ = clone(self.transformer)
        # Maintains original transformer in the event of refitting.
        self.fit_transformer_.fit(X, y)
        return self

    def _transform(self, X: pd.DataFrame, y: None) -> pd.DataFrame:
        """Transform X using self.transformer"""
        self.logger.info(
            f"X_shape in feature_selection_transformer is: {X.shape} features..."
        )
        X_trans = self.fit_transformer_.transform(X)
        X_trans = self._convert_to_dataframe(X_trans, X)
        return X_trans

    def _convert_to_dataframe(
        self, X_trans: np.ndarray, X: pd.DataFrame
    ) -> pd.DataFrame:
        """We convert to dataframe here instead of in parent since we need to mask
        columns using get_support.

        :param X_trans: Df in progress of transform.
        :type X_trans: np.ndarray
        :param X: Original Df before being transformed (but after preparing for
            transform in parent!)
        :type X: pd.DataFrame
        :return: Dataframe with column names restored.
        :rtype: pd.DataFrame
        """
        columns = X.loc[:, self.fit_transformer_.get_support()].columns
        X_trans = pd.DataFrame(X_trans, columns=columns, index=X.index)
        return X_trans

    def get_feature_importances(
        self,
        getter: Union[str, callable] = "auto",
        transform_fuction: str = "norm",
        norm_order: int = 1,
    ) -> pd.DataFrame:
        """
        TODO: Implement fixes for SelectKBest and RFE
        """
        check_is_fitted(self)
        if not isinstance(self.fit_transformer_, SelectFromModel):
            raise NotImplementedError(
                "Only SelectFromModel is currently supported for "
                "get_feature_importances."
            )
        feature_importances = _get_feature_importances(
            self.fit_transformer_.estimator_,
            getter=getter,
            transform_func=transform_fuction,
        )
        return pd.DataFrame(
            {
                "Feature": self.columns_before_transform_,
                "Importance": feature_importances,
            }
        ).set_index("Feature")
