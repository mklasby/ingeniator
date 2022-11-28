import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


def p_value_wrapper(estimator: BaseEstimator) -> None:
    _unwrapped_fit = estimator.fi

    def calculate_p(estimator, X, y):
        # check_is_fitted(estimator=estimator)
        # sse = np.sum((estimator.predict(X) - y) ** 2, axis=0) / float(
        #     X.shape[0] - X.shape[1]
        # )
        # se = np.array(
        #     [
        #         np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
        #         for i in range(sse.shape[0])
        #     ]
        # )
        raise NotImplementedError()
        # t = self.coef_ / se
        # self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))

    def wrapped_fit(X, y, **fit_params):
        _unwrapped_fit(X, y, **fit_params)
        estimator.p_ = calculate_p(estimator, X, y)
        

