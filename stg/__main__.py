import stg.utils
from stg.simulator import Simulator 
from datetime import datetime
import click
import os

# obtain parameters for data simulation
params = stg.utils.read_params()

DATE_FROM = datetime.strptime(params["transaction_from"], "%Y-%m-%d")
DATE_TO = datetime.strptime(params["transaction_to"], "%Y-%m-%d")
FEATURE_COUNT = params["feature_count"]
TRANSACTION_RATE = params["transaction_rate_per_second"]
FRAUD_RATE = TRANSACTION_RATE*params["fraud_rate"]
RANDOM_STATE = params["random_state"]

# create simulator
transaction_data_simulator = Simulator(
    date_from=DATE_FROM, 
    date_to=DATE_TO,
    transaction_rate=TRANSACTION_RATE,
    fraud_rate=FRAUD_RATE,
    n_features=FEATURE_COUNT,
    random_state=RANDOM_STATE
)

# simulate the transaction data
df = transaction_data_simulator.simulate_all_transaction()

# save transaction data
df.to_csv("sim_data_{}_{}.csv".format(params["transaction_from"], params["transaction_to"]))