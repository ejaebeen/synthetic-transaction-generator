import pytest
import numpy as np
import stg.simulator
from datetime import datetime
import pandas as pd
from stg.utils import read_params

params = read_params()

def test_simulate():
    t_date_test, t_amount_test, t_features_test, t_fraud_detected_days_test = stg.simulator.simulate(
        date_from=datetime(2022,1,1),
        date_to=datetime(2022,1,2),
        transaction_rate=0.01,
        n_features=2,
        fraud=False
    )
    t_date_test_fraud, t_amount_test_fraud, t_features_test_fraud, t_fraud_detected_days_test_fraud = stg.simulator.simulate(
        date_from=datetime(2022,1,1),
        date_to=datetime(2022,1,2),
        transaction_rate=0.001,
        n_features=2,
        fraud=True
    )

    # test for normal: all length must be same and fraud detected days must be null
    assert len(t_date_test) == len(t_amount_test) 
    assert len(t_date_test) == len(t_features_test)
    assert t_fraud_detected_days_test is None

    # test for fraud: all length must be the same
    assert len(t_date_test_fraud) == len(t_amount_test_fraud)
    assert len(t_date_test_fraud) == len(t_features_test_fraud)
    assert len(t_date_test_fraud) == len(t_fraud_detected_days_test_fraud)

def test_simulator_simulate_transaction():
    # test for simulate_transaction in simulator for normal transaction
    simulator = stg.simulator.Simulator(
        date_from=datetime(2022,1,1),
        date_to=datetime(2022,1,2),
        transaction_rate=0.01,
        fraud_rate=0.001,
        n_features=2
    )
    df_normal = simulator.simulate_transaction(fraud=False)
    
    assert ~(df_normal[params["column_name"]['fraud']].all()) # all fraud label should be false
    assert df_normal[params["column_name"]['fraud_identified_date']].isna().all() # all fraud identified must be null
    assert df_normal[params["column_name"]['transaction_amount']].notna().all() # all should have transaction amount
    assert df_normal[params["column_name"]['transaction_date']].notna().all() # all should have transaction date
    assert df_normal.shape[1] == 6 # expected shape
    assert pd.api.types.is_datetime64_ns_dtype(df_normal[params["column_name"]['transaction_date']]) # dtype for transaction date must be datetime

def test_simulaor_simulate_transaction_fraud():
    simulator = stg.simulator.Simulator(
        date_from=datetime(2022,1,1),
        date_to=datetime(2022,1,2),
        transaction_rate=0.01,
        fraud_rate=0.001,
        n_features=2
    )
    df_fraud = simulator.simulate_transaction(fraud=True)
    
    assert df_fraud[params["column_name"]['fraud']].all()
    assert df_fraud[params["column_name"]['fraud_identified_date']].notna().all()
    assert df_fraud[params["column_name"]['transaction_amount']].notna().all()
    assert df_fraud[params["column_name"]['transaction_date']].notna().all()
    assert (df_fraud[params["column_name"]['transaction_date']] < df_fraud[params["column_name"]['fraud_identified_date']]).all()
    assert df_fraud.shape[1] == 6
    assert pd.api.types.is_datetime64_ns_dtype(df_fraud[params["column_name"]['transaction_date']]) # dtype for transaction date must be datetime
    assert pd.api.types.is_datetime64_ns_dtype(df_fraud[params["column_name"]['fraud_identified_date']]) # dtype for fraud identified date must be datetime

def test_simulator_simulate_all_transaction():
    simulator = stg.simulator.Simulator(
        date_from=datetime(2022,1,1),
        date_to=datetime(2022,1,2),
        transaction_rate=0.01,
        fraud_rate=0.001,
        n_features=2
    )
    df = simulator.simulate_all_transaction()

    assert df[params["column_name"]['fraud']].notna().all()
    assert df[params["column_name"]['transaction_amount']].notna().all()
    assert df[params["column_name"]['transaction_date']].notna().all()
    assert df.loc[df[params["column_name"]['fraud']], params["column_name"]['fraud_identified_date']].notna().all()
    assert df.loc[~df[params["column_name"]['fraud']], params["column_name"]['fraud_identified_date']].isna().all()
    assert df.shape[1] == 6
    assert pd.api.types.is_datetime64_ns_dtype(df[params["column_name"]['transaction_date']]) # dtype for transaction date must be datetime
    assert pd.api.types.is_datetime64_ns_dtype(df[params["column_name"]['fraud_identified_date']]) # dtype for fraud identified date must be datetime
