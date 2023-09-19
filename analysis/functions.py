import numpy as np
import itertools
from statannot import add_stat_annotation
import matplotlib.pyplot as plt
import seaborn as sns
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
    """
    Function for computing transfer entropy
    :param x: signal 1
    :param y: signal 2
    :param k: lag
    :return: transfer entropy
    """
    x = x[:-k]
    z = y[:-k]
    y = y[k:]

    return float(drv.information_mutual_conditional(y, x, z))


def stat_test(data, x, y, test, text_format="simple", kind="bar"):
    """
    Function for statistical testing
    :param data: DataFrame with data
    :param x: feature 1
    :param y: feature 2
    :param test: statistical test
        ["t-test_ind","t-test_welch","t-test_paired","Mann-Whitney","Mann-Whitney-gt","Mann-Whitney-ls","Levene","Wilcoxon","Kruskal"]
    :param text_format: format of pvalue annotation
        ["star", "simple", "full"]
    :param kind: kind of plot
        ["bar", "box"]
    """
    plt.figure(figsize=(7, 6))
    if kind == "box":
        ax = sns.boxplot(data=data, x=x, y=y)
    else:
        ax = sns.barplot(data=data, x=x, y=y)
    test_results = add_stat_annotation(ax, data=data, x=x, y=y,
                                       box_pairs=itertools.combinations(data[x].unique().tolist(), 2),
                                       test=test, text_format=text_format,
                                       loc='outside', verbose=0)

    plt.show()
