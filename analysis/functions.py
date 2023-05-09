import numpy as np
import statsmodels.api as sm
from pyitlib import discrete_random_variable as drv


def corr_df_to_distribution(df):
    """
    Function for transforming symmetric dataframe to list of values (only values above (below) the main diagonal)
    :param df: symmetric dataframe
    :return: list of values
    """
    corr = []
    vals = df.values

    symm = (vals == vals.T).all()
    for i, row in enumerate(vals):
        if symm:
            corr.extend(row[i + 1 :])
        else:
            corr.extend(list(row[:i]) + list(row[i + 1 :]))

    return corr


def active_df_to_dict(df):
    """
    Function for transforming dataframe of active states to dict
    :param df: DataFrame of active states
    :return: dict with indexes of active states
    """
    d = {}
    for col in df:
        sig = df[col]
        active = sig[sig == True]

        idx = active.reset_index()[["index"]].diff()
        idx = idx[idx["index"] > 1]

        d[col] = np.array_split(np.array(active.index.tolist()), idx.index.tolist())
    return d


def crosscorr(signal1, signal2, lag=100):
    """
    Function for computing cross-correlation
    :param signal1: signal 1
    :param signal2: signal 2
    :param lag: lag radius
    :return: maximum correlation value
    """
    corr = list(sm.tsa.stattools.ccf(signal1, signal2)[: lag + 1]) + list(
        sm.tsa.stattools.ccf(signal2, signal1)[1 : lag + 1]
    )
    return max(corr)


def transfer_entropy(x, y, k=1):
    x = x[:-k]
    z = y[:-k]
    y = y[k:]

    return float(drv.information_mutual_conditional(y, x, z))
