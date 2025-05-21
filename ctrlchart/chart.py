import numpy as np
import pandas as pd

from ctrlchart.model import LinearRegression
from ctrlchart.utils import check_and_return_dt_index_df, find_closest_value

POM = [100, 370, 500, 1000]
DELTA_1 = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

LAMBDA = np.array(
    [
        [0.07, 0.06, 0.05, 0.04],
        [0.12, 0.1, 0.09, 0.07],
        [0.19, 0.15, 0.15, 0.13],
        [0.33, 0.26, 0.24, 0.22],
        [0.52, 0.4, 0.37, 0.35],
        [0.66, 0.54, 0.52, 0.46],
        [0.81, 0.7, 0.7, 0.66],
    ]
)

L = np.array(
    [
        [2.01, 2.55, 2.62, 2.82],
        [2.21, 2.7, 2.79, 2.97],
        [2.35, 2.8, 2.91, 3.11],
        [2.47, 2.9, 2.99, 3.2],
        [2.54, 2.96, 3.05, 3.25],
        [2.56, 2.98, 3.07, 3.27],
        [2.57, 2.99, 3.09, 3.29],
    ]
)

POM1 = np.array(
    [
        [17.3, 26.5, 28.7, 34.3],
        [10.3, 14.7, 15.8, 18.4],
        [7, 9.6, 10.2, 11.7],
        [3.9, 5.2, 5.5, 6.1],
        [2.6, 3.3, 3.5, 3.9],
        [1.89, 2.38, 2.5, 2.76],
        [1.45, 1.78, 1.86, 2.06],
    ]
)

LAMBDA_DF = pd.DataFrame(LAMBDA, DELTA_1, POM)
L_DF = pd.DataFrame(L, DELTA_1, POM)
POM1_DF = pd.DataFrame(POM1, DELTA_1, POM)


class EWMAChart:
    """
    An exponentially weighted moving average (EWMA) control chart for Heating and
    Cooling system performance monitoring.

    This chart is designed to track the deviation of Heating/Cooling system performance
    from expected behavior.
    It uses a fitted linear regression model to predict target values and compute
    control limits based on the exponentially weighted moving average (EWMA) of
    residuals.

    Implementation is based on the FBE Document : FBE MPEB - Fiche Carte de contrôles
    fiche opérationelle

    Model uncertainties and error metrics are computed according to FD X30-148

    Limits calculation is based on the NF X 06-031-3 standard using pre-tabulated
    values for λ, L, and POM1.

    Parameters
    ----------
    m0 : float
        Target or nominal value representing in-control system performance.

    m1 : float
        Upper performance threshold used for delta_1 calculation.

    m1p : float
        Lower performance threshold used for delta_1 calculation.

    pom_0 : int, default=370
        Number of samples (Average Operational Period) before detecting a false
        positive.
        FBE recommand values around 370-500. It is used for table lookup of λ and L.
        Must be one of [100, 370, 500, 1000] as per the standard.

    intercept : bool, default=False
        Whether to fit the linear regression model with an intercept.

    Attributes
    ----------
    model_ : LinearRegression
        The fitted regression model used to estimate expected system performance.

    s0_ : float
        Estimated standard deviation of the combined model and measurement error.

    delta1_ : float
        Normalized deviation used to determine λ and L from the standard tables.

    lambda_ewma_ : float
        Smoothing factor for the EWMA, selected from standard tables.

    l_ : float
        Control limit multiplier used to compute upper and lower control limits.

    pom_1_ : float
        Sample size before detecting a deviation.

    zi : pd.Series
        EWMA series of residuals from the predictive model.

    is_fitted_ : bool
        Indicates whether the chart has been fitted with reference data.

    Methods
    -------
    fit(X, y)
        Fit the control chart on reference input and output data.

    predict(X, y)
        Apply the EWMA chart to new data and compute control limits.

    reset()
        Reset the EWMA series to the initial condition.

    Examples
    --------
    >>> chart = EWMAChart(m0=0, m1=3000, m1p=-3000, pom_0=370)
    >>> chart.fit(data["DJU__C"], data["Consumption__kWh"])
    >>> result = chart.predict(data["DJU__C"], data["Consumption__kWh"])
    >>> result.head()
                        zi          Lcs          Lci
    2010-09-01  -76.347976  1259.944907 -1259.944907
    2010-10-01    2.631510  1653.602692 -1653.602692
    2010-11-01  277.742846  1887.608676 -1887.608676

    Notes
    -----
    - The model is sensitive to the selection of `m0`, `m1`, and `m1p`, which define
        the expected performance band of the HVAC system.
    - Measurement uncertainty (`u_mes`) is currently fixed to 0 but can be adapted if
        needed.
    - Assumes input data is time-indexed with no duplicate timestamps.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the EWMA values (`zi`), upper control limits (`Lcs`),
        and lower control limits (`Lci`) for the input timestamps.
    """

    def __init__(
        self,
        m0: float,
        m1: float,
        m1p: float,
        pom_0: int = 370,
        intercept: bool = False,
    ):
        self.m0 = m0
        self.m1 = m1
        self.m1p = m1p
        self.u_mes = 0
        self.pom_0 = pom_0
        self.lambda_ewma_ = 0.54
        self.l_ = 2.98
        self.intercept = intercept
        self.reset()

    def __repr__(self):
        ewma_par_dict = {
            "m0": self.m0,
            "m1": self.m1,
            "m1p": self.m1p,
            "pom_0": self.pom_0,
        }
        repr_string = ""
        if not hasattr(self, "is_fitted_"):
            repr_string += "Chart is not fitted yet\n"
            repr_string += f"{ewma_par_dict}"

        else:
            ewma_computed_par = {
                "delta_1": self.delta1_,
                "lambda": self.lambda_ewma_,
                "L": self.l_,
                "pom_1": self.pom_1_,
            }
            repr_string += "EWMA parameters \n"
            repr_string += f"{ewma_par_dict} \n\n"
            repr_string += "Model properties \n"
            repr_string += f"{self.model_.training_metrics_} \n\n"
            repr_string += "EWMA computed properties \n"
            repr_string += f"{ewma_computed_par}"

        return repr_string

    def reset(self):
        self.zi = pd.Series([self.m0], index=[pd.Timestamp("1970-01-01")])

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):
        X = check_and_return_dt_index_df(X)
        y = check_and_return_dt_index_df(y)

        # Fit model and get its u_mod
        self.model_ = LinearRegression(self.intercept)
        self.model_.fit(X, y)
        umod = self.model_.training_metrics_.loc["umod", "metrics"]

        # From u_mod and u_mes compute s0 and delta1
        self.s0_ = np.sqrt(umod**2 + self.u_mes**2) / np.sqrt(self.model_.ddof_)
        self.delta1_ = min(
            (self.m1 - self.m0) / self.s0_, (self.m0 - self.m1p) / self.s0_
        )

        # Given POM, delta_1, and considering n = 1 (one observation at a time)
        # find lambda, L and POM1 according to NF X 06-031-3 Table 2
        self.lambda_ewma_ = find_closest_value(LAMBDA_DF, self.pom_0, self.delta1_)
        self.l_ = find_closest_value(L_DF, self.pom_0, self.delta1_)
        self.pom_1_ = find_closest_value(POM1_DF, self.pom_0, self.delta1_)

        self.is_fitted_ = True

    def predict(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):
        X = check_and_return_dt_index_df(X)
        y = check_and_return_dt_index_df(y)
        if not hasattr(self, "is_fitted_"):
            raise ValueError(
                "Chart is not fitted yet, use fit() method on reference data"
            )
        x = (self.model_.predict(X) - y).squeeze()
        self.zi = pd.concat([self.zi, x])
        self.zi = self.zi.loc[~self.zi.index.duplicated(keep="last")]
        self.zi.sort_index(inplace=True)
        zi_ewm_start = self.zi.index.get_loc(X.index[0]) - 1
        zi_ewm_stop = self.zi.index.get_loc(X.index[-1])
        new_zi_m1 = self.zi.iloc[zi_ewm_start : zi_ewm_stop + 1]
        self.zi.iloc[zi_ewm_start : zi_ewm_stop + 1] = new_zi_m1.ewm(
            alpha=self.lambda_ewma_, adjust=False
        ).mean()

        uncertain_term = []
        for t_stamp in X.index:
            i = self.zi.index.get_loc(t_stamp)
            uncertain_term.append(
                self.l_
                * self.s0_
                * np.sqrt(self.lambda_ewma_ / (2 - self.lambda_ewma_))
                * np.sqrt(1 - (1 - self.lambda_ewma_) ** (2 * i))
            )
        uncertain_term = np.array(uncertain_term)

        return pd.DataFrame(
            {
                "zi": self.zi.loc[X.index],
                "Lcs": self.m0 + uncertain_term,
                "Lci": self.m0 - uncertain_term,
            },
            index=X.index,
        )
