import json

import numpy as np
import pandas as pd
import pytest

from vip2 import vix30


def payload(rows):
    return {"d": json.dumps({"TrendData": rows})}


def test_taiwan_official_dates_and_values_are_preserved():
    # The official service returns YYYYMMDD local dates inside an encoded object.
    data = payload([{"Time": "20261022", "Price": 30.31},
                    {"Time": "20261020", "Price": 27.22},
                    {"Time": "20261021", "Price": 28.40},
                    {"Time": "20261023", "Price": 99.0}])
    result = vix30.parse_taiwan_vix(data, pd.Timestamp("2026-10-22"))
    assert result.index.tolist() == list(pd.date_range("2026-10-20", periods=3))
    assert result.tolist() == [27.22, 28.40, 30.31]
    assert result.index.tz is None


def test_cboe_uses_close_and_excludes_outside_window():
    csv = "DATE,OPEN,HIGH,LOW,CLOSE\n10/21/2021,90,99,80,10\n10/22/2021,90,99,80,15.43\n10/22/2026,90,99,80,19.03\n10/23/2026,90,99,80,19.45\n"
    result = vix30.parse_cboe_vix(csv, pd.Timestamp("2026-10-22"))
    assert result.index.strftime("%Y-%m-%d").tolist() == ["2021-10-22", "2026-10-22"]
    assert result.tolist() == [15.43, 19.03]


def test_missing_trading_date_is_not_invented():
    result = vix30.parse_taiwan_vix(payload([
        {"Time": "20261020", "Price": 27.22},
        {"Time": "20261022", "Price": 30.31},
    ]), pd.Timestamp("2026-10-22"))
    assert pd.Timestamp("2026-10-21") not in result.index


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-1], [0]])
def test_rejects_invalid_official_values(values):
    with pytest.raises(ValueError, match="invalid"):
        vix30.clean_official_vix(["20261022"], values, "%Y%m%d", pd.Timestamp("2026-10-22"))


def test_rejects_duplicate_dates():
    with pytest.raises(ValueError, match="duplicate"):
        vix30.clean_official_vix(["20261022", "20261022"], [30, 31], "%Y%m%d", pd.Timestamp("2026-10-22"))


@pytest.mark.parametrize("data", [{"d": "{}"}, {"d": "[]"}, {}, {"d": {"TrendData": [{"Time": "20261022"}]}}])
def test_rejects_taiwan_schema_changes(data):
    with pytest.raises(ValueError, match="Missing"):
        vix30.parse_taiwan_vix(data, pd.Timestamp("2026-10-22"))


def test_rejects_cboe_schema_changes():
    with pytest.raises(ValueError, match="Missing"):
        vix30.parse_cboe_vix("Date,Price\n2026-10-22,19.03\n", pd.Timestamp("2026-10-22"))


def test_taiwan_request_covers_five_years_on_local_price_dates(monkeypatch):
    requests = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return payload([{"Time": "20261022", "Price": 30.31}])
    def post(url, **kwargs):
        requests.append((url, kwargs))
        return Response()
    monkeypatch.setattr(vix30.requests, "post", post)
    result = vix30.download_vix30("TAIEX", pd.Timestamp("2026-10-22"))
    assert result.iloc[0] == 30.31
    assert requests[0][0] == vix30.SOURCES["TAIEX"]["url"]
    assert requests[0][1]["json"] == {"syid": "TAIWANVIX", "flag": "MS",
                                        "startDate": "2021/10/22", "endDate": "2026/10/22"}
