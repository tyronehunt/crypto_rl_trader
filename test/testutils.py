import pytest
from prepare_trade_data import get_data


def test_get_data():
    data = get_data('../data/btc_ohlc_1d.csv')
    assert len(data) > 0


if __name__ == "__main__":
    test_get_data()