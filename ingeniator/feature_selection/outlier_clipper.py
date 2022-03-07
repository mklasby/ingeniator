"""Transformer to clip outliers.

Typical usage example:
    oc = OutlierClipper(lower_limit=0.05, upper_limit=0.95, regexs=['Fee', 'pay'],
        copy=True, case=True)
    oc.fit(X_train)
    X_test = oc.transform(X_test)
"""
from ingeniator.feature_selection.dataframe_transformer_wrapper import DataFrameTransformerWrapper
import numpy as np
import pandas as pd
from typing import Optional, List
import logging


class OutlierClipper(DataFrameTransformerWrapper):
    """Customer transformer to clip outliers to lower and upper limits.

    #TODO: Update init for new params

    Can target specific columns using regexs parameter or apply to all columns.

    Attributes:
        lower_limit: Lower quantile to clip to, eg. clip values <= 0.01 (1%) Defaults to
            0.01
        upper_limit: Upper quantile to clip to, eg., clip values >= 0.99 (99%) Defaults
            to 0.99
        regexs: List of strings to use to filter column names. If None, all columns
            considered valid targets for clipping.
        copy: A bool indicating if the dataframes passed to transform should be copied
            before being transformed.
    """

    def __init__(
        self,
        lower_limit: float = 0.01,
        upper_limit: float = 0.99,
        copy: bool = True,
        ignore_features: Optional[List[str]] = None,
        retain_col_names: bool = True,
        append_cols: bool = False,
        column_suffix: Optional[str] = None,
        keep_feature_order: bool = True,
        regex_feature_selector: Optional[List[str]] = None,
    ):
        """Init a new OutlierClipper object"""
        super().__init__(
            copy=copy,
            ignore_features=ignore_features,
            retain_col_names=retain_col_names,
            append_cols=append_cols,
            column_suffix=column_suffix,
            keep_feature_order=keep_feature_order,
        )
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.logger = logging.getLogger(__file__)

    def _reset(self):
        """Private method to reset fit parameters."""
        super()._reset()  # Must be called as we are overriding.
        if hasattr(self, "quantiles_"):
            del self.quantiles_
            del self.to_clip_

    def _fit(self, X: pd.DataFrame, y=None):
        """Private method to implement fit.
        Args:
            X: A dataframe to fit to.
            y: Ignored. Defaults to None.
        """
        self.quantiles_ = {}
        self.to_clip_ = X.columns
        self.logger.info(f"Found {len(self.to_clip_)} features to clip outliers for...")
        for feature in self.to_clip_:
            self.quantiles_[feature] = X[feature].quantile([self.lower_limit, self.upper_limit]).values

    def _transform(self, X: pd.DataFrame, y=None, copy: bool = None):
        """Transforms X by clipping outliers based on fit dataframe.

        Args:
            X:
                A dataframe to transform
            y:
                Ignored. Left as a parameter to maintain compatibility with existing
                fit_transform() interfaces. Defaults to None.
            copy:
                Optional parameter to use instead of self.copy. Defaults to None.

        Returns:
            Transformed dataframe with outliers clipped based on values in fit() dataframe.

        Raises:
            NotFittedError: If the transformer has not been fit.
        """
        for feature in self.to_clip_:
            X[feature] = np.clip(X[feature], self.quantiles_[feature][0], self.quantiles_[feature][1])
        return X

if __name__ == "__main__":
    from sklearn import datasets
    X,y = datasets.load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    oc = OutlierClipper()
    oc.fit(X)
    oc.fit_transform(X)
    oc.transform(X)
    print(X)