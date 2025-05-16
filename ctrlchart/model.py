import numpy as np
import pandas as pd

from ctrlchart.tools import ols_regression, nmbe, rmse, cv_rmse, umod


class LinearRegression:
    """
    Ordinary Least Squares linear regression with intercept handling and model
    diagnostics.

    This class implements a simple linear regression estimator using the closed-form OLS
    solution. It supports models with or without an intercept and provides built-in
    diagnostics such as R², RMSE, NMBE, CV(RMSE), and model projection uncertainty
    (`umod`) based on the regression residuals and predictor variance.

    Parameters
    ----------
    intercept : bool, default=False
        If True, an intercept column of ones is added to the feature matrix.

    Attributes
    ----------
    beta_hat_ : pd.Series
        Estimated regression coefficients.

    stand_err_ : pd.Series
        Standard errors of the estimated coefficients.

    var_cov_ : pd.DataFrame
        Variance-covariance matrix of the coefficients.

    r2_ : float
        Coefficient of determination for the training data.

    ddof_ : int
        Degrees of freedom used in the residual-based metrics
        (equals number of predictors).

    training_metrics_ : pd.DataFrame
        Diagnostic metrics computed on the training dataset including R², RMSE, NMBE,
        CV(RMSE), and projection uncertainty (umod).

    target_label_ : str
        Column name of the target variable used during fitting.

    is_fitted_ : bool
        Whether the model has been fitted.

    Methods
    -------
    fit(X, y)
        Estimate model parameters using the training data.

    predict(X)
        Predict target values for new data using the fitted model.

    Example
    -------
    >>> import pandas as pd
    >>> from ctrlchart.model import LinearRegression
    >>> X = pd.Series([1, 2, 3, 4, 5], name="x")
    >>> y = pd.DataFrame({"y": [2, 4, 6, 8, 10]})
    >>> model = LinearRegression(intercept=True)
    >>> model.fit(X, y)
    >>> preds = model.predict(X)
    >>> print(model.training_metrics_)
                  metrics
    R2              1.000
    RMSE            0.000
    NMBE            0.000
    CV(RMSE)        0.000
    umod            0.000
    """

    def __init__(self, intercept: bool = False):
        self.intercept = intercept

    def _add_intercept(self, x):
        if isinstance(x, pd.Series):
            x = x.to_frame()
        intercept = pd.Series(np.ones(x.shape[0]), name="intercept", index=x.index)
        return pd.concat([intercept, x], axis=1)

    def fit(self, X, y):
        x_fit = self._add_intercept(X) if self.intercept else X
        self.beta_hat_, self.stand_err_, self.var_cov_, self.r2_ = ols_regression(
            x_fit, y
        )

        training_predictions = x_fit @ self.beta_hat_

        self.ddof_ = x_fit.shape[1]
        self.training_metrics_ = pd.DataFrame(
            {
                "R2": self.r2_,
                "RMSE": rmse(training_predictions, y, self.ddof_),
                "NMBE": nmbe(training_predictions, y),
                "CV(RMSE)": cv_rmse(training_predictions, y, self.ddof_),
                "umod": umod(
                    training_predictions.squeeze(),
                    y.squeeze(),
                    x_pred=X.squeeze(),
                    ddof=self.ddof_,
                ),
            },
            index=["metrics"],
        ).T
        self.target_label_ = y.columns[0]
        self.is_fitted_ = True

    def predict(self, X):
        if not hasattr(self, "is_fitted_"):
            raise ValueError("LinearRegression is not fitted")
        x = self._add_intercept(X) if self.intercept else X
        result = x @ self.beta_hat_
        result.columns = [self.target_label_]
        return result
