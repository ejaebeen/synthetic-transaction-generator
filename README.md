# Synthetic Transaction Generator

This is a mini-package to generate synthetic transaction dataset that contains both normal and fraudulent transaction. It generates the following:
- transaction date (between start and end date)
- transaction amount
- features (number of features configurable)
- fraud label
- fraud identified date

## How to run 

simply run `python -m stg`. It will save the file name `sim_data_{transaction_from}_{transaction_to}.csv`

You can also do the unit tests by running `python -m pytest`  

You can configure the simulator by going `params.yaml`. 

## how does it work?

It begins by sampling the transaction date between transaction_from and transaction_to. Once we have the list of the transaction dates, it simulates the transaction amount, features (and fraud identified date if it is fraudulent transaction)

## details of the model

NORMAL TRANSACTION
- transaction date: modelled as homogeneous poisson process with rate `transaction_rate`
- transaction amount: sampled from X~Gamma(alpha=2, beta=1)
- features: X~N(0, 1) + noise N(0, 0.1) (multivariate normal distribution)

FRAUDULENT TRANSACTION
- transaction date: modelled as homogeneous poisson process with rate `fraud_rate` multiplied by `transaction_rate`
- transaction amount: sampled from X~Gamma(alpha=2, beta=1) with prob 0.8, X~Gamma(alpha=15, beta=1) with prob 0.2 
    - some transaction assumed to have higher value.
- features:  X~N(mu, 1.5) + noise N(0, 1.0) where mu for each dim equals 0 with prob 0.3 and 1 with prob 0.7
- days until fraud detected:  sampled from X~exp((1/30)*ln10)
    - (1/30)*ln10 is the value that makes P(X<30) = 0.9

## TODO

- logger for outputting the information
- include params for file saving destination
- add more unit tests
- ability to choose different distribution (probably need to break down the sample.py into smaller modules)
