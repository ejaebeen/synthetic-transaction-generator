from datetime import datetime, timedelta
import numpy as np
from scipy import stats


def transaction_date(
        transaction_rate: float, 
        date_from: datetime, 
        date_to: datetime,
        random_state: int = 12321
    ):
    """
    compute the transaction time. Transaction modelled as homogeneous poisson process with rate transaction rate per second
    time between each transaction have exponential distribution
    """
    assert date_from < date_to

    # work out expected value of transaction count to determine how many rvs need to be sampled
    seconds_between = (date_to-date_from).total_seconds()
    exp_transaction_count = seconds_between*transaction_rate
    
    # sample and only keep until cumulative sum equals time between date_from and date_to
    t = stats.expon.rvs(scale=1/transaction_rate, size=int(exp_transaction_count*5), random_state=random_state)
    t_cumsum = np.cumsum(t)
    t_cumsum = t_cumsum[t_cumsum <= seconds_between]
    transaction_time = [date_from + timedelta(seconds=x) for x in t_cumsum]
    
    return transaction_time

def transaction_amount(
        size: int, 
        fraud: bool = False, 
        random_state: int = 12321    
    ):
    """
    sample transaction amount 
        normal transaction: X~Gamma(alpha=2, beta=1)
        fraud transaction: X~Gamma(alpha=2, beta=1) with prob 0.8, X~Gamma(alpha=15, beta=1) with prob 0.2
    """
    if fraud:
        sample_uniform = stats.uniform.rvs(size=size, random_state=random_state)
        u = (sample_uniform < 0.8).astype(int)
        v = (sample_uniform >= 0.8).astype(int)
        sample = u*stats.gamma.rvs(a=2, size=size, random_state=random_state) + v*stats.gamma.rvs(a=15, size=size, random_state=random_state)
    else:
        sample = stats.gamma.rvs(a=2, size=size, random_state=random_state)
    
    return sample

def features(
        size: int, 
        n_features: int, 
        fraud: bool = False, 
        random_state: int = 12321
    ):
    """
    sample transaction features 
        normal transaction: X~N(0, 1) + noise N(0, 0.1)
        fraud transaction: X~N(mu, 1.5) + noise N(0, 1.0) where mu for each dim equals 0 with prob 0.3 and 1 with prob 0.7
    """
    if fraud:
        sample_uniform = stats.uniform.rvs(size=n_features, random_state=random_state)
        mean = (sample_uniform > 0.3).astype(int)
        sample = stats.multivariate_normal.rvs(mean=mean, cov=2.5, size=size, random_state=random_state)
    else:
        sample = stats.multivariate_normal.rvs(mean=[0]*n_features, cov=1.1, size=size, random_state=random_state)
    
    return sample


def days_until_fraud_detected(
        size: int, 
        random_state: int = 12321
    ):
    """
    sample number of days until fraud detected modelled as X~exp((1/30)*ln10)
    """
    l = (1/30)*np.log(10)
    sample = stats.expon.rvs(scale=1/l, size=size, random_state=random_state)
    
    return sample