import numpy as np
import pandas as pd

import statsmodels.api as sm
from pathlib import Path
from ctrlchart.utils import model_projection_uncertainty, umod, nmbe, cv_rmse, rmse
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
ewma_chart = EWMAChart(0, 3000, -3000, intercept=True)
ewma_chart.fit(test_data["DJU__C"], test_data["Consumption__kWh"])

# %%
ewma_chart.predict(test_data["DJU__C"], test_data["Consumption__kWh"]).plot()
plt.show()
