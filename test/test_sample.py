import pytest
import numpy as np
import stg.sample
from datetime import datetime

def test_transaction_amount_positive():
    # all transaction amount must be positive
    amount = stg.sample.transaction_amount(size=10)
    amount_fraud = stg.sample.transaction_amount(size=10, fraud=True)

    assert (amount > 0).all()
    assert (amount_fraud > 0).all()

def test_transaction_amount_fraud_mean_larger():
    # avg amount for fraud in large sample should be larger 
    amount = stg.sample.transaction_amount(size=100000)
    amount_fraud = stg.sample.transaction_amount(size=100000, fraud=True)

    assert np.mean(amount) < np.mean(amount_fraud)

def test_transaction_date_dates():
    td = stg.sample.transaction_date(
        transaction_rate=0.001,
        date_from=datetime(2022,1,1),
        date_to=datetime(2022,1,31)
    )

    assert len(td) == len(set(td)) # all dates should be unique
    assert max(td) <= datetime(2022,1,31) # last date should be earlier than date_to

def test_features_correct_shape():
    # check all the features generated have expected shapes
    f = stg.sample.features(10, 10, fraud=False)
    f_fraud = stg.sample.features(10, 10, fraud=True)

    assert f.shape == (10,10)
    assert f_fraud.shape == (10,10)

def test_days_until_fraud_detected_correct_shape():
    # check all days until fraud detected have correct shape
    d = stg.sample.days_until_fraud_detected(10)

    assert len(d) == 10

def test_days_until_fraud_detected_positive():
    # check all days until fruad detected have positive days 
    d = stg.sample.days_until_fraud_detected(1000)

    assert (d > 0).all()