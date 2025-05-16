import numpy as np
import pandas as pd

from ctrlchart.model import LinearRegression


class EWMAChart:
    def __init__(self, m0: float, m1: float, m1p: float, intercept: bool = False):
        self.m0 = m0
        self.m1 = m1
        self.m1p = m1p
        self.u_mes = 0
        self.lambda_ewma = 0.4
        self.intercept = intercept

    def fit(self, X, y):
        self.model_ = LinearRegression(self.intercept)
        self.model_.fit(X, y)
        umod = self.model_.training_metrics_.loc["umod", "metrics"]
        self.s0_ = np.sqrt(umod**2 + self.u_mes**2) / np.sqrt(self.model_.ddof_)
        self.delta1_ = min(
            (self.m1 - self.m0) / self.s0_, (self.m0 - self.m1p) / self.s0_
        )

    def predict(self, X, y):
        x = (self.model_.predict(X) - y).squeeze()
        x_with_m0 = pd.concat([pd.Series([self.m0], index=[-1]), x])
        zi = x_with_m0.ewm(alpha=self.lambda_ewma, adjust=False).mean().iloc[1:]

        return None

    #
    # def transform(self, ):
    #
    #     # Paramètres CUSUM
    #     s0 =
    #     m0 = 0
    #     delta1 = min((m1 - m0) / s0, (m0 - m1_prime) / s0)
    #     k = (delta1 * np.sqrt(1)) / 2
    #     b = k * POM1 + (1 / (2 * k))
    #     h = b - 1.166
    #
    #     # CUSUM sur résidus test
    #     z = (resid_test - m0) / s0
    #     S_pos = [0] * (len(z) + 1)
    #     S_neg = [0] * (len(z) + 1)
    #     alerts = []
    #
    #     for i in range(len(z)):
    #         S_pos[i + 1] = max(0, S_pos[i] + z.iloc[i] - k)
    #         S_neg[i + 1] = min(0, S_neg[i] + z.iloc[i] + k)
    #         if S_pos[i + 1] >= h or S_neg[i + 1] <= -h:
    #             alerts.append(z.index[i])
    #
    #     # Construction DataFrame
    #     df_test = pd.DataFrame(index=test_y.index)
    #     df_test['CUSUM+'] = pd.Series(S_pos[1:], index=test_y.index)
    #     df_test['CUSUM-'] = pd.Series(S_neg[1:], index=test_y.index)
    #     df_test['alerte'] = df_test.index.isin(alerts)
    #
    #     if return_all:
    #         df_ref = pd.DataFrame(index=ref_y.index)
    #         df_ref['CUSUM+'] = np.nan
    #         df_ref['CUSUM-'] = np.nan
    #         df_ref['alerte'] = False
    #         return pd.concat([df_ref, df_test])
    #
    #     return df_test
