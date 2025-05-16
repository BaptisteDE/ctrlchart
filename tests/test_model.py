import pandas as pd

from pathlib import Path

from ctrlchart.model import LinearRegression

RESOURCES_PATH = Path(__file__).parent / "resources"

linear_data = pd.read_csv(
    Path(RESOURCES_PATH / "stat_model_data.csv"),
    parse_dates=True,
    index_col=0,
)


class TestModel:
    def test_linear_regression(self):
        linear_model = LinearRegression(intercept=True)
        linear_model.fit(linear_data[["DJU__C"]], linear_data[["Consumption__kWh"]])

        assert linear_model.training_metrics_.to_dict() == {
            "metrics": {
                "R2": 0.9435077765840214,
                "RMSE": 1224.6913198780155,
                "NMBE": -2.0582080664112855e-15,
                "CV(RMSE)": 18.47670322523144,
                "umod": 4242.455179234623,
            }
        }

        prediction = linear_model.predict(linear_data[["DJU__C"]])

        assert prediction.to_dict() == {
            "Consumption__kWh": {
                pd.Timestamp("2010-09-01 00:00:00"): 1344.0134941163983,
                pd.Timestamp("2010-10-01 00:00:00"): 4475.181929693233,
                pd.Timestamp("2010-11-01 00:00:00"): 9224.707084781689,
                pd.Timestamp("2010-12-01 00:00:00"): 15627.77062719724,
                pd.Timestamp("2011-01-01 00:00:00"): 11546.697160602713,
                pd.Timestamp("2011-02-01 00:00:00"): 7606.350365270067,
                pd.Timestamp("2011-03-01 00:00:00"): 8485.892060656819,
                pd.Timestamp("2011-04-01 00:00:00"): 3806.7302411993014,
                pd.Timestamp("2011-05-01 00:00:00"): 3138.2785527053707,
                pd.Timestamp("2011-06-01 00:00:00"): 1027.3784837771677,
            }
        }
