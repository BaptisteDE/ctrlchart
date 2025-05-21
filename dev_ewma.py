import numpy as np
import pandas as pd

import statsmodels.api as sm
from pathlib import Path
from ctrlchart.tools import model_projection_uncertainty, umod, nmbe, cv_rmse, rmse
from ctrlchart.model import LinearRegression
from ctrlchart.chart import EWMAChart

import seaborn as sns
import matplotlib.pyplot as plt

# pio.renderers.default = "browser"


test_data = pd.read_csv(
    Path(r"C:\Users\bdurandestebe\PycharmProjects\ctrlchart\tests\resources")
    / "stat_model_data.csv",
    parse_dates=True,
    index_col=0,
)


# %%
ewma_chart = EWMAChart(0, 3, -3, intercept=True)
ewma_chart.fit(test_data[["DJU__C"]], test_data[["Consumption__kWh"]])

# %%
ewma_chart.predict(test_data[["DJU__C"]], test_data[["Consumption__kWh"]]).plot()
plt.show()

# %%
ewma_chart.predict(
    test_data.loc[:"2010-12-01", ["DJU__C"]],
    test_data.loc[:"2010-12-01":, ["Consumption__kWh"]]
)

#%%
ewma_chart.predict(
    test_data.loc["2010-12-01":, ["DJU__C"]],
    test_data.loc["2010-12-01":, ["Consumption__kWh"]]
).plot()
plt.show()


#%%
ewma_chart.predict(
    test_data.loc["2011-02":, ["DJU__C"]],
    test_data.loc["2011-02":, ["Consumption__kWh"]]
)
# %%
line_reg = LinearRegression(intercept=True)
line_reg.fit(test_data[["DJU__C"]], test_data[["Consumption__kWh"]])

# %%
line_reg.predict(test_data[["DJU__C"]], test_data[["Consumption__kWh"]])


# %%
plt.figure()
plt.scatter(test_data[["DJU__C"]], line_reg.predict(test_data[["DJU__C"]]))
plt.scatter(test_data[["DJU__C"]], test_data[["Consumption__kWh"]])
plt.show()

# %%
x = test_data.to_numpy()
inter = np.ones((x.shape[0], 1))
np.concatenate([x, inter], axis=1)

# %%
X = sm.add_constant(test_data[["DJU__C"]])
y = test_data["Consumption__kWh"]

mod_dt = sm.OLS(y, X)
res = mod_dt.fit()
print(res.summary())

preds = res.predict(X)


# %%
print(f"RMSE = {rmse(test_data["Consumption__kWh"], preds, ddof=2)}")
print(
    f"proj_uncertainty = {
        model_projection_uncertainty(
            test_data["Consumption__kWh"],
            preds,
            x_pred=test_data["DJU__C"],
            ddof=2
        )
    }"
)

print(
    f"u_mod = {
        umod(
            test_data["Consumption__kWh"],
            preds,
            x_pred=test_data["DJU__C"],
            ddof=2
        )
    }"
)
# %%

print(f"CV(RMSE) = {cv_rmse(test_data["Consumption__kWh"], preds, ddof=2)}")
print(f"NMBE = {nmbe(test_data["Consumption__kWh"], preds)}")
print(f"RMSE = {rmse(test_data["Consumption__kWh"], preds, ddof=2)}")
