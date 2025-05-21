import pandas as pd

from pathlib import Path

from ctrlchart.chart import EWMAChart

RESOURCES_PATH = Path(__file__).parent / "resources"

linear_data = pd.read_csv(
    Path(RESOURCES_PATH / "stat_model_data.csv"),
    parse_dates=True,
    index_col=0,
)


class TestChart:
    def test_ewma_chart(self):
        ewma_chart = EWMAChart(0, 3000, -3000, intercept=True)
        ewma_chart.fit(linear_data["DJU__C"], linear_data["Consumption__kWh"])
        res = ewma_chart.predict(linear_data["DJU__C"], linear_data["Consumption__kWh"])

        ref = {
            "zi": {
                pd.Timestamp("2010-09-01 00:00:00"): -76.34797588254025,
                pd.Timestamp("2010-10-01 00:00:00"): 2.6315099538256845,
                pd.Timestamp("2010-11-01 00:00:00"): 277.7428461780051,
                pd.Timestamp("2010-12-01 00:00:00"): 281.64701333089033,
                pd.Timestamp("2011-01-01 00:00:00"): 144.4045354216637,
                pd.Timestamp("2011-02-01 00:00:00"): -224.3035901010757,
                pd.Timestamp("2011-03-01 00:00:00"): -257.1242424873915,
                pd.Timestamp("2011-04-01 00:00:00"): -122.44606993438755,
                pd.Timestamp("2011-05-01 00:00:00"): 75.36262346157616,
                pd.Timestamp("2011-06-01 00:00:00"): -15.234997491085096,
            },
            "Lcs": {
                pd.Timestamp("2010-09-01 00:00:00"): 1259.9449069690525,
                pd.Timestamp("2010-10-01 00:00:00"): 1653.602691975545,
                pd.Timestamp("2010-11-01 00:00:00"): 1887.6086762220712,
                pd.Timestamp("2010-12-01 00:00:00"): 2040.0433145790778,
                pd.Timestamp("2011-01-01 00:00:00"): 2143.4422904918574,
                pd.Timestamp("2011-02-01 00:00:00"): 2215.146795282559,
                pd.Timestamp("2011-03-01 00:00:00"): 2265.5415667467473,
                pd.Timestamp("2011-04-01 00:00:00"): 2301.265184707568,
                pd.Timestamp("2011-05-01 00:00:00"): 2326.734227726294,
                pd.Timestamp("2011-06-01 00:00:00"): 2344.9634877485937,
            },
            "Lci": {
                pd.Timestamp("2010-09-01 00:00:00"): -1259.9449069690525,
                pd.Timestamp("2010-10-01 00:00:00"): -1653.602691975545,
                pd.Timestamp("2010-11-01 00:00:00"): -1887.6086762220712,
                pd.Timestamp("2010-12-01 00:00:00"): -2040.0433145790778,
                pd.Timestamp("2011-01-01 00:00:00"): -2143.4422904918574,
                pd.Timestamp("2011-02-01 00:00:00"): -2215.146795282559,
                pd.Timestamp("2011-03-01 00:00:00"): -2265.5415667467473,
                pd.Timestamp("2011-04-01 00:00:00"): -2301.265184707568,
                pd.Timestamp("2011-05-01 00:00:00"): -2326.734227726294,
                pd.Timestamp("2011-06-01 00:00:00"): -2344.9634877485937,
            },
        }

        pd.testing.assert_frame_equal(res, pd.DataFrame(ref))
