"""Custom transformer to wrap PCA decomposition.

Typical usage example:
    pca_transformer = PCATransformer(copy=True)
    X_train = pca_transformer.fit_transform(X_train)
    X_test = pca_transformer.transform(X_test)
"""

from systemicmacrorisk.data_inspection.dataframe_transformer_wrapper import (
    DataFrameTransformerWrapper,
)
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from typing import Optional, List


class PCATransformer(DataFrameTransformerWrapper):
    """Transformer wrapper for PCA. Reduces dimensions of dataframes using PCA.

    Option to append pca dimensions to original dataframe instead of dropping and ignore certain columns is included.

    :param random_state: Random state to be used in PCA. Defaults to 42
    :type random_state: int, optional
    :param copy: A bool indicating if the dataframes passed to transform should be copied before being transformed
        Defaults to True
    :type copy: bool, optional
    :param ignore_features: Features to ignore in transformation, defaults to None
    :type ignore_features: list, optional
    :param pca_kwargs: Keyword arguments to pass to PCA. If None, defaults from sklearn are used. Defaults to None
    :type pca_kwargs: dict, optional
    :param regex_feature_selector: List of regex strings to select columns from. Any column not matching at
        least one regex is added to ignore_features. Defaults to None.
    :type regex_feature_selector: Optional[List[str]]
    :param fill_na: If True, fill all NaN values in targetted columns with 0.
    :type fill_na: bool, optional.
    """

    def __init__(
        self,
        random_state: int = 42,
        copy: bool = True,
        ignore_features: list = None,
        pca_kwargs: dict = None,
        column_suffix: str = "_pca",
        regex_feature_selector: Optional[List[str]] = None,
        fill_na_with_zero: bool = True,
    ):
        super().__init__(
            copy=copy,
            ignore_features=ignore_features,
            column_suffix=column_suffix,
            retain_col_names=False,
            keep_feature_order=False,
            regex_feature_selector=regex_feature_selector,
        )
        self.random_state = random_state
        self.pca_kwargs = pca_kwargs
        self.fill_na_with_zero = fill_na_with_zero

    def _reset(self):
        """Private method to reset fit parameters."""
        super()._reset()
        if hasattr(self, "pca_"):
            del self.pca_

    def _fit(self, X: pd.DataFrame, y=None):
        """Private method to implement fit."""
        X = X.copy()
        X = self._fill_na(X)
        if self.pca_kwargs is not None:
            self.pca_ = PCA(**self.pca_kwargs, random_state=self.random_state)
        else:
            self.pca_ = PCA(random_state=self.random_state)
        self.pca_.fit(X)

    def _transform(self, X: pd.DataFrame, y=None) -> np.array:
        """Transforms X by projecting it into principal components.

        :param X: A dataframe to transform.
        :type X: pd.DataFrame
        :param y: Ignored. Left as a parameter to maintain compatibility with existing fit_transform() interfaces.
            Defaults to None.
        :return: Transformed dataframe with principal components as features or joined to original df.
        :rtype: pd.DataFrame
        :raises: NotFittedError if the transformer has not been fit.
        """
        X = self._fill_na(X)
        return self.pca_.transform(X)

    def _fill_na(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill na in X with 0.

        :param X: Dataframe to fill NaNs in.
        :type X: pd.DataFrame
        :return: Dataframe with NaNs imputed to 0.
        :rtype: pd.DataFrame
        """
        if self.fill_na_with_zero:
            # By this stage, only a subset of valid columns have been selected
            X = X.fillna(0, axis=1)
        return X
