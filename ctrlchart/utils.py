import pandas as pd
import numpy as np


def check_and_return_dt_index_df(X: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if not (isinstance(X, pd.Series) or isinstance(X, pd.DataFrame)):
        raise ValueError(
            f"Invalid X data, was expected an instance of pandas Dataframe "
            f"or Pandas Series. Got {type(X)}"
        )

    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("Index is not a pandas DateTimeIndex")

    return X.to_frame() if isinstance(X, pd.Series) else X


def find_closest_value(df, pom, delta_1):
    x_vals = df.columns.values
    y_vals = df.index.values

    closest_x = x_vals[np.argmin(np.abs(x_vals - pom))]
    closest_y = y_vals[np.argmin(np.abs(y_vals - delta_1))]

    return df.loc[closest_y, closest_x]


def ols_regression(x: np.ndarray, y: np.ndarray):
    """
    Performs an Ordinary Least Squares (OLS) regression using matrix algebra.

    This function estimates the linear relationship between independent variables
    (`x`) and a dependent variable (`y`) using the closed-form solution to OLS.
    It returns the estimated coefficients, standard errors, variance-covariance
    matrix, and coefficient of determination (R²).

    Parameters
    ----------
    x : np.ndarray
        2D array of shape (n_samples, n_features) representing the independent
        variables. To include an intercept (affine regression), a column of ones
        must be added manually.

    y : np.ndarray
        1D or 2D array of shape (n_samples,) or (n_samples, 1) representing the
        dependent variable values.

    Returns
    -------
    beta_hat : np.ndarray
        Estimated regression coefficients of shape (n_features, 1).

    stand_err : np.ndarray
        Standard error of the estimated coefficients, shape (n_features,).

    var_cov : np.ndarray
        Variance-covariance matrix of shape (n_features, n_features).

    R2 : float
        Coefficient of determination (R²), representing the proportion of
        variance in `y` explained by `x`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.column_stack((np.ones(5), np.arange(5)))
    >>> y = np.array([1, 2, 1.3, 3.75, 2.25])
    >>> beta_hat, stand_err, var_cov, R2 = ols_regression(x, y)

    Notes
    -----
    - The OLS estimator is given by:
        beta_hat = (XᵀX)⁻¹ Xᵀy
    - Residual variance is estimated using degrees of freedom = n - k,
      where n is number of samples and k is number of predictors.
    - R² is calculated as:
        R² = 1 - SSR / SST
      where SSR is the residual sum of squares, and SST is the total sum of squares.

    Raises
    ------
    ValueError
        If the number of samples in `x` and `y` do not match.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")

    # Ensure y is a column vector
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # beta_hat = (XᵀX)⁻¹ Xᵀy
    xtx_inv = np.linalg.inv(x.T @ x)
    beta_hat = xtx_inv @ x.T @ y

    # Residuals and residual sum of squares
    residuals = y - x @ beta_hat
    SSR = (residuals.T @ residuals)[0, 0]

    # Degrees of freedom
    dof = x.shape[0] - x.shape[1]

    # Residual variance
    sig2 = SSR / dof

    # Variance-covariance matrix and standard errors
    var_cov = sig2 * xtx_inv
    stand_err = np.sqrt(np.diag(var_cov))

    # Total sum of squares
    y_mean = y.mean()
    SStot = ((y - y_mean).T @ (y - y_mean))[0, 0]

    R2 = 1 - SSR / SStot

    return beta_hat, stand_err, var_cov, R2


def model_projection_uncertainty(
    y_pred, y_true, x_pred, x_train=None, ddof: int = 1
) -> float:
    """
    This function calculates the propagated uncertainty when applying a linear
    regression model to new input values. It accounts for the model's residual error
    (RMSE), the number of training points, and the dispersion of the independent
    variable. The result is a global scalar uncertainty value representing the
    combined confidence interval over all prediction points.

    Parameters
    ----------
    y_pred : pd.Series
        Estimated target values corresponding to `x_pred`.

    y_true : pd.Series
        Actual target values corresponding to `x_pred`.

    x_pred : pd.Series
        Independent variable values used for making predictions.
        Must be aligned with `y_pred`.

    x_train : pd.Series, optional
        Independent variable values used during model training.
        If not provided, `x_pred` is used (useful when the model is fit and
        evaluated on the same dataset).

    ddof : int, default=2
        Degrees of freedom used in the RMSE calculation.
        This is subtracted from the number of samples to compute the error denominator.

    Returns
    -------
    float
        Scalar uncertainty on the model projections.

    Examples
    --------
    >>> import pandas as pd
    >>> from ctrlchart.utils import model_projection_uncertainty
    >>> x_train = pd.Series([1, 2, 3, 4, 5])
    >>> x_pred = pd.Series([2, 3, 4])
    >>> y_true = pd.Series([2, 2.5, 3])
    >>> y_pred = pd.Series([2.1, 2.6, 2.9])
    >>> model_projection_uncertainty(y_pred, y_true, x_pred, x_train)
    0.157...

    Notes
    -----
    - The formula used is based on uncertainty propagation for linear regression:
        u_mod = sqrt( sum_k [ s * sqrt(1 + 1/n + ((x_k - x̄)^2 / sum_i((x_i - x̄)^2)) ]^2 )
      where:
        - s is the RMSE of the model (based on `y_pred`, `y_true`)
        - x̄ is the mean of `x_train`
        - n is the number of training points
    - If `x_train` is None, it is assumed the model was trained and evaluated on the same data.
    - Designed for uncertainty analysis in fields such as energy forecasting and regression modeling.

    Raises
    ------
    ValueError
        If `y_pred` and `y_true` are not the same length, or sample size is <= ddof.
    """

    if len(y_pred) != len(y_true):
        raise ValueError("`y_pred` and `y_true` must be of the same length.")

    if len(y_true) <= ddof:
        raise ValueError(
            "Number of samples must be greater than degrees of freedom (`ddof`)."
        )

    if x_train is None:
        x_train = x_pred

    x_bar = x_train.mean()
    n = x_train.shape[0]

    s = np.sqrt(((y_pred - y_true) ** 2).sum() / (len(y_true) - ddof))
    num = (x_pred - x_bar) ** 2
    den = ((x_train - x_bar) ** 2).sum()

    return s * np.sqrt(1 + 1 / n + num / den)


def umod(
    y_pred,
    y_true,
    x_pred,
    x_train=None,
    ddof=1,
) -> float:
    individual_uncertainties = model_projection_uncertainty(
        y_pred, y_true, x_pred, x_train, ddof
    )
    return np.sqrt((individual_uncertainties**2).sum())


def nmbe(y_pred: pd.Series, y_true: pd.Series) -> float:
    """Normalized Mean Biased Error

    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :return:
    Normalized Mean biased error as float
    """
    return (
        np.sum(y_pred.to_numpy() - y_true.to_numpy()) / np.sum(y_true.to_numpy()) * 100
    )


def rmse(y_pred: pd.Series, y_true: pd.Series, ddof: int = 1) -> float:
    """Compute the Root Mean Squared Error (RMSE) between predicted and true values.

    The square root of the average squared differences between predicted
    and actual values. This implementation includes an optional adjustment for the
    number of model parameters, which is useful when calculating an unbiased estimator
    of the variance and to comply with french regulation FD X30-148.

    Parameters
    ----------
    y_pred : pd.Series
        Estimated target values. Must be a 1D pandas Series with the same length as
        `y_true`.

    y_true : pd.Series
        Ground truth (correct) target values. Must be a 1D pandas Series with the same
        length as `y_pred`.

    ddof : int, default=1
        Number of model parameters used in the prediction. This is subtracted from the
        number of samples in the denominator to adjust for degrees of freedom, similar
        to the unbiased variance estimator in statistics.

    Returns
    -------
    float
        Root mean squared error between `y_pred` and `y_true`.

    Examples
    --------
    >>> import pandas as pd
    >>> from tide.metrics import rmse
    >>> y_true = pd.Series([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = pd.Series([2.5, 0.0, 2.0, 8.0])
    >>> rmse(y_pred, y_true)
    0.6123724356957945

    Notes
    -----
    - The RMSE is calculated as:
        sqrt( sum((y_pred - y_true)^2) / (n_samples - ddof) )
    - `ddof` should reflect the number of model parameters fitted to the data.
      For standard RMSE, set `n_parameters=0`.
    - This function assumes that `y_pred` and `y_true` are aligned (i.e., same index).
      If using pandas Series with different indices, align them explicitly or convert
      to NumPy arrays before passing.
    - If `ddof >= len(y_true)`, a ValueError is raised.
    """

    n = y_true.shape[0]
    if n <= ddof:
        raise ValueError(
            "Number of samples must be greater than number of model parameters."
        )

    err = y_pred.to_numpy() - y_true.to_numpy()
    return np.sqrt(np.sum(err**2) / (n - ddof))


def cv_rmse(y_pred: pd.Series, y_true: pd.Series, ddof: int = 1) -> float:
    """
    Compute the Coefficient of Variation of the Root Mean Squared Error (CV(RMSE)).

    CV(RMSE) is a normalized version of RMSE, typically expressed as a percentage.
    It represents the RMSE as a proportion of the mean of the true values.

    Parameters
    ----------
    y_pred : pd.Series
        Estimated target values. Must be a 1D pandas Series with the same length as
        `y_true`.

    y_true : pd.Series
        Ground truth (correct) target values. Must be a 1D pandas Series with the
        same length as `y_pred`.

    ddof : int, default=1
        Degrees of freedom adjustment for the RMSE calculation. This value is subtracted
        from the number of samples when calculating the mean squared error denominator.
        For standard RMSE (not adjusted), set `ddof=0`.
        It is meant to comply with french regulation FD X30-148.

    Returns
    -------
    float
        Coefficient of variation of the RMSE, expressed as a percentage.

    Examples
    --------
    >>> import pandas as pd
    >>> from ctrlchart.utils import cv_rmse
    >>> y_true = pd.Series([100, 102, 98, 101])
    >>> y_pred = pd.Series([98, 100, 95, 99])
    >>> cv_rmse(y_pred, y_true)
    2.777...

    Notes
    -----
    - Expressing RMSE as a percentage of the mean allows for scale-independent
      performance comparison across different datasets or target magnitudes.
    - If the mean of `y_true` is 0, a ZeroDivisionError will be raised.
    - `y_pred` and `y_true` must be aligned and of equal length.
    - This implementation is commonly used in energy modeling
    (e.g., ASHRAE Guideline 14).

    Raises
    ------
    ValueError
        If number of samples is less than or equal to `ddof`.

    ZeroDivisionError
        If the mean of `y_true` is zero.
    """

    mean_true = np.mean(y_true)
    if mean_true == 0:
        raise ZeroDivisionError("Mean of y_true is zero, CV(RMSE) is undefined.")

    return (1 / mean_true) * rmse(y_pred, y_true, ddof) * 100
