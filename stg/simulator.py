import stg.sample
from stg.utils import read_params
from datetime import datetime
import pandas as pd

params = read_params()

def simulate(
        date_from: datetime, 
        date_to: datetime,
        transaction_rate: float,
        n_features: int,
        fraud: bool = False,
        random_state: int = 12321
    ):
    """
    generate all the features required for the data. 
    It will create the following:
        transaction date, transaction amount, features, 
        number of days until fraud detected (only for fraud, null if not fraud) 
    """
    # transaction date
    t_date = stg.sample.transaction_date(
        transaction_rate=transaction_rate,
        date_from=date_from,
        date_to=date_to,
        random_state=random_state
    )
    # transaction amount
    t_amount = stg.sample.transaction_amount(
        size=len(t_date),
        fraud=fraud,
        random_state=random_state
    )
    # transaction features
    t_features = stg.sample.features(
        size=len(t_date),
        n_features=n_features,
        fraud=fraud,
        random_state=random_state,
    )
    # fraud identified date
    if fraud:
        t_fraud_detected_days = stg.sample.days_until_fraud_detected(
            size=len(t_date)
        )
    else:
        t_fraud_detected_days = None
    
    return t_date, t_amount, t_features, t_fraud_detected_days

def compile(t_date, t_amount, t_features, t_fraud_detected_days):
    """
    compile all the features into one dataframe
    """
    df_t_date = pd.Series(t_date, name=params["column_name"]['transaction_date'])
    df_t_amount = pd.Series(t_amount, name=params["column_name"]['transaction_amount'])
    df_t_features = pd.DataFrame(t_features)
    df = pd.concat([df_t_date, df_t_amount, df_t_features], axis=1)
    if t_fraud_detected_days is None:
        df[params["column_name"]['fraud']] = False
        df[params["column_name"]['fraud_identified_date']] = None
    else:
        df[params["column_name"]['fraud']] = True
        df[
            params["column_name"]['fraud_identified_date']
        ] = df[params["column_name"]['transaction_date']]  + pd.to_timedelta(t_fraud_detected_days, unit='D')
    
    return df


class Simulator:
    """
    simulator class for generating transaction dataset
    """
    def __init__(
        self, 
        date_from: datetime, 
        date_to: datetime,
        transaction_rate: float,
        fraud_rate: float,
        n_features: int,
        random_state: int = 12321
    ):
        assert date_from < date_to

        self.date_from = date_from
        self.date_to = date_to
        self.transaction_rate = transaction_rate
        self.fraud_rate = fraud_rate
        self.n_features = n_features
        self.random_state = random_state
    
    def simulate_transaction(self, fraud: bool):
        """
        simulate one type of transaction (fraud or no fraud)
        """
        if fraud:
            t_date, t_amount, t_features, t_fraud_detected_days = simulate(
                self.date_from, 
                self.date_to,
                self.fraud_rate,
                self.n_features,
                fraud,
                self.random_state
            )
        else:
            t_date, t_amount, t_features, t_fraud_detected_days = simulate(
                self.date_from, 
                self.date_to,
                self.transaction_rate,
                self.n_features,
                fraud,
                self.random_state
            )
        df = compile(t_date, t_amount, t_features, t_fraud_detected_days)

        return df
    
    def simulate_all_transaction(self):
        """
        simulate all transactions that contain both fraud and non-fraud
        """
        df_normal = self.simulate_transaction(fraud=False)
        df_fraud = self.simulate_transaction(fraud=True)
        df = pd.concat([df_normal, df_fraud])

        df = df.sort_values(params["column_name"]['transaction_date'])

        return df
