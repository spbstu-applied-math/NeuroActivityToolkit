import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.notebook import tqdm

from analysis.active_state import ActiveStateAnalyzer
from analysis.functions import active_df_to_dict, corr_df_to_distribution


class Data:
    def __init__(self, path_to_data, sessions, verbose=False):
        """
        Initialising function
        :param path_to_data: path to data folder
        :param sessions: dict with information about sessions
                        {session 1:
                            {'path': path to session folder,
                             'mouse': mouse id,
                             'condition': condition of session,
                             'fps': fps of session},
                         session 2: ...
                         }

        :param verbose: progressbar
        """

        self.sessions = sessions.keys()
        self.data = None
        self.disable_verbose = not verbose
        self.params = sessions
        self.data_reduced = None
        self.path_to_data = path_to_data

        self.models = {}
        for date in tqdm(self.sessions, disable=self.disable_verbose):
            session_path = self.params[date]["path"]
            ma = ActiveStateAnalyzer(
                f"{self.path_to_data}/{session_path}/minian/", self.params[date]["fps"]
            )
            ma.active_state_df = pd.read_excel(
                f"{self.path_to_data}/{session_path}/results/active_states_spike.xlsx",
                index_col=0,
            ).astype(bool)
            ma.active_state = active_df_to_dict(ma.active_state_df)

            ma.smooth_signals = ma.signals.rolling(
                window=10, center=True, min_periods=0
            ).mean()
            ma.smooth_diff = ma.smooth_signals.diff()[1:].reset_index(drop=True)

            self.models[date] = ma

    def _get_burst_rate_data(self):
        """
        Function for collecting burst rate data
        :return: burst rate data
        """
        df_br = pd.DataFrame()
        for date in tqdm(
            self.sessions,
            disable=self.disable_verbose,
            desc="Step 1/6: Burst rate computing...",
        ):
            df_ptr = pd.DataFrame()
            df_ptr["burst_rate"] = self.models[date].burst_rate()["activations per min"]
            df_ptr["model"] = date
            df_br = pd.concat([df_br, df_ptr])

        df_br = df_br.reset_index(drop=True)

        return df_br

    def _get_nsp_data(self):
        """
        Function for collecting network spike peak data
        :return: network spike peak data
        """
        df_nsp = pd.DataFrame()
        for date in tqdm(
            self.sessions,
            disable=self.disable_verbose,
            desc="Step 2/6: Network spike peak computing...",
        ):
            df_ptr = pd.DataFrame()
            df_ptr["network_spike_peak"] = (
                self.models[date].network_spike_peak(1).T["peak"]
            )
            df_ptr["model"] = date
            df_nsp = pd.concat([df_nsp, df_ptr])

        df_nsp = df_nsp.reset_index(drop=True)

        return df_nsp

    def _get_nsr_data(self):
        """
        Function for collecting network spike rate data
        :return: network spike rate data
        """
        df_nsr = pd.DataFrame()
        for date in tqdm(
            self.sessions,
            disable=self.disable_verbose,
            desc="Step 3/6: Network spike rate computing...",
        ):
            df_ptr = pd.DataFrame()
            df_ptr["network_spike_rate"] = (
                self.models[date].network_spike_rate(1).T["spike rate"]
            )
            df_ptr["model"] = date
            df_nsr = pd.concat([df_nsr, df_ptr])

        df_nsr = df_nsr.reset_index(drop=True)

        return df_nsr

    def _get_corr_data(self, method="signal"):
        """
        Function for collecting correlation data
        :param method: method of correlation
        :return: DataFrame with correlation distribution
        """
        corr_dict = {}
        for date in self.sessions:
            corr_dict[date] = self.models[date].get_correlation(method).fillna(0)

        return corr_dict

    def _get_nd_data(
        self,
        df_corr,
        method="signal",
        thrs=None,
    ):
        """
        Function for collecting 'network degree' data
        Network degree - share of strong network connections
        :param df_corr: DataFrame with correlation distribution
        :param method: method of correlation
        :param thrs: list of thresholds for strong correlation between neurons
        :return: network degree data
        """

        if thrs is None:
            thrs = [0.1, 0.2, 0.3, 0.4, 0.5]

        df_network_degree = pd.DataFrame()
        for date in self.sessions:
            df_ptr = pd.DataFrame()
            corr = np.array(df_corr[df_corr["model"] == date][f"correlation_{method}"])
            for thr in thrs:
                df_ptr[f"network_degree_{method}_thr_{thr}"] = [
                    (corr > thr).sum() / len(corr)
                ]
            df_ptr["model"] = date
            df_network_degree = pd.concat([df_network_degree, df_ptr])

        df_network_degree = df_network_degree.reset_index(drop=True)
        df_network_degree = df_network_degree.fillna(0)
        df_network_degree = df_network_degree.set_index("model")

        return df_network_degree

    def _get_conn_data(self, df_corr, method="signal", q=0.9):
        """
        Function for collecting 'connectivity' data
        Connectivity - share of strong connections for each neuron
        :param df_corr: DataFrame with correlation distribution
        :param method: method of correlation
        :param q: threshold for strong correlation between neurons
        :return: connectivity data
        """
        df_conn = pd.DataFrame()

        total_distr = []
        for x in df_corr:
            total_distr.extend(corr_df_to_distribution(df_corr[x]))
        thr = np.quantile(total_distr, q=q)

        for date in self.sessions:
            df_ptr = pd.DataFrame()
            corr_df = df_corr[date]
            df_ptr[f"connectivity_{method}"] = ((corr_df > thr).sum() - 1) / len(
                corr_df
            )
            df_ptr["model"] = date
            df_conn = pd.concat([df_conn, df_ptr])

        df_conn = df_conn.reset_index(drop=True)

        return df_conn

    def get_data(self, transfer_entropy=False):
        """
        Function for collecting all data
        :param transfer_entropy: bool: compute transfer entropy or not (time-consuming)
        """

        df_br = self._get_burst_rate_data()
        df_nsp = self._get_nsp_data()
        df_nsr = self._get_nsr_data()

        agg_functions = ["mean", "std", q95, q5, iqr]

        nsr = df_nsr.groupby("model").agg(agg_functions)
        nsp = df_nsp.groupby("model").agg(agg_functions)
        br = df_br.groupby("model").agg(agg_functions)

        corr_types = ["signal", "diff", "active", "active_acc"]

        if transfer_entropy:
            corr_types += ["transfer_entropy"]

        df_corr = {}
        corr_distr = {}
        corrs = {}
        for corr in tqdm(
            corr_types,
            disable=self.disable_verbose,
            desc="Step 4/6: Correlation computing...",
        ):
            corr_tmp = self._get_corr_data(corr)
            df_corr[corr] = corr_tmp

            corr_distr[corr] = pd.concat(
                [
                    pd.DataFrame(
                        {
                            f"correlation_{corr}": corr_df_to_distribution(corr_tmp[x]),
                            "model": x,
                        }
                    )
                    for x in corr_tmp
                ]
            ).reset_index(drop=True)

            corrs[corr] = corr_distr[corr].groupby("model").agg(agg_functions)

        df_network_degree = {}
        for corr in tqdm(
            corr_types,
            disable=self.disable_verbose,
            desc="Step 5/6: Network degree computing...",
        ):
            df_network_degree[corr] = self._get_nd_data(corr_distr[corr], method=corr)

        df_conn = {}
        for corr in tqdm(
            corr_types,
            disable=self.disable_verbose,
            desc="Step 6/6: Connectivity computing...",
        ):
            df_conn[corr] = self._get_conn_data(df_corr[corr], corr)
            df_conn[corr] = df_conn[corr].groupby("model").agg(agg_functions)

        data = nsp.join(nsr)
        data = data.join(br)

        for corr in corr_types:
            data = data.join(df_conn[corr])

        for corr in corr_types:
            data = data.join(corrs[corr])

        data = data.T.set_index(data.columns.map("_".join)).T

        data = data.reset_index()

        for corr in corr_types:
            data = data.merge(df_network_degree[corr].reset_index(), on="model")

        data = data.set_index("model")

        data = data[data.columns[data.apply(lambda x: len(x.unique()) > 1)]]

        self.data = data

    def drop_strong_corr(self, thr=0.9):
        """
        Function for drop strong correlation statistics
        :param thr: threshold for dropping
        """
        while (self.data.corr() > thr).sum().sum() + (
            self.data.corr() < -thr
        ).sum().sum() > len(self.data):
            strong_corr = (self.data.corr() > thr).sum() + (
                self.data.corr() < -thr
            ).sum()
            val = strong_corr.max()
            if val <= 1:
                break
            col = strong_corr.idxmax()
            self.data = self.data.drop(columns=[col])

            if len(self.data.columns) <= 2:
                continue

    def data_reduction(
        self, model=PCA(n_components=2, random_state=42), scaler=StandardScaler()
    ):
        """
        Function for data reduction
        :param model: reduction model using 'fit_transform' method
        :param scaler: scaling model using 'fit_transform' method
        :return: reduced data
        """
        sessions = self.data.index

        data = self.data.copy()

        if scaler:
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        data_reduced = model.fit_transform(data)

        data_reduced = pd.DataFrame(data_reduced, columns=["x", "y"])

        data_reduced["session"] = sessions
        data_reduced["mouse"] = [self.params[session]["mouse"] for session in sessions]
        data_reduced["condition"] = [
            self.params[session]["condition"] for session in sessions
        ]

        self.data_reduced = data_reduced
        return data_reduced, model

    def show_result(self, mouse, condition_order=None):
        """
        Function for plotting result after data reduction
        :param mouse: mouse for plotting
        :param condition_order: order of conditions in time
        """
        if self.data_reduced is None:
            data = self.data_reduction()[0]
        else:
            data = self.data_reduced.copy()

        if condition_order is None:
            condition_order = data["condition"].unique()

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", ["red", "yellow", "green"], len(condition_order)
        )
        palette = [cmap(i) for i in range(len(condition_order))]
        palette = [palette[-1]] + palette[:-1]

        palette_dict = {}
        for cond, color in zip(condition_order, palette):
            palette_dict[cond] = color

        fig = plt.figure(figsize=(9, 8))

        data_mouse = data[data["mouse"] == mouse]
        plt.title(f"Mouse {mouse}", fontsize=18)

        sns.scatterplot(
            data=data_mouse,
            x="x",
            y="y",
            hue="condition",
            hue_order=condition_order,
            palette=palette_dict,
            s=120,
            zorder=2,
        )

        conditions = data_mouse["condition"].value_counts()
        centers = data_mouse.groupby("condition").mean()

        for cond in conditions[conditions > 1].index:
            sns.scatterplot(
                x=[centers.loc[cond]["x"]],
                y=[centers.loc[cond]["y"]],
                label=f"{cond} mean",
                color=palette_dict[cond],
                marker="s",
                s=120,
                zorder=2,
            )

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        d = max(xmax - xmin, ymax - ymin)
        for start_cond, end_cond in zip(condition_order[:-1], condition_order[1:]):
            plt.arrow(
                centers.loc[start_cond]["x"],
                centers.loc[start_cond]["y"],
                centers.loc[end_cond]["x"] - centers.loc[start_cond]["x"],
                centers.loc[end_cond]["y"] - centers.loc[start_cond]["y"],
                width=d * 0.0045,
                length_includes_head=True,
                color=palette_dict[end_cond],
                zorder=1,
            )

        formatter = mpl.ticker.StrMethodFormatter("{x:.1f}")

        fig.axes[0].xaxis.set_major_formatter(formatter)
        fig.axes[0].yaxis.set_major_formatter(formatter)

        plt.tick_params(axis="both", labelsize=14)

        plt.show()

    def get_stats_deviation(self, condition="all"):
        """
        Function for computing deviation of statistics
        :param condition: {'all' or specific condition} condition for plotting
        :return: pd.Series with deviation of statistics
        """
        if condition == "all":
            df = self.data.copy()
        else:
            df = self.data[
                [self.params[x]["condition"] == condition for x in self.data.index]
            ]

        df = pd.DataFrame(
            MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index
        )

        df["mouse"] = [self.params[session]["mouse"] for session in df.index]

        if condition == "all":
            df["condition"] = [
                self.params[session]["condition"] for session in df.index
            ]

            df = df.groupby(["mouse", "condition"]).mean()
            feat_mad = df.reset_index().groupby("mouse").mad().mean().sort_values()
        else:
            df = df.groupby("mouse").mean()
            feat_mad = df.mad().sort_values()

        return feat_mad

    def show_stats_deviation(self, condition="all", topn=8):
        """
        Function for plotting deviation of statistics
        :param condition: {'all' or specific condition} condition for plotting
        :param topn: number of statistics for plotting
        """
        feat_mad = self.get_stats_deviation(condition)

        plt.figure(figsize=(7, 6))
        plt.barh(feat_mad[:topn].index, feat_mad[:topn])
        plt.barh(feat_mad[-topn:].index, feat_mad[-topn:])
        plt.show()

    def show_stat(self, stat, condition="all", conditions_order=None):
        """
        Function for plotting bars with information about statistic
        :param stat: statistic for plotting
        :param condition: {'all' or specific condition} condition for plotting
        :param conditions_order: order of conditions in time (used only if condition=='all')
        """

        if condition == "all":
            df = self.data[[stat]]
        else:
            df = self.data[
                [self.params[x]["condition"] == condition for x in self.data.index]
            ][[stat]]

        df = pd.DataFrame(
            MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index
        )

        df["mouse"] = [self.params[session]["mouse"] for session in df.index]

        if condition == "all":
            df["condition"] = [
                self.params[session]["condition"] for session in df.index
            ]

            df = df.groupby(["mouse", "condition"]).mean().reset_index()

            fig, ax = plt.subplots(1, 2, figsize=(18, 8))

            ax[0].set_title(stat)

            sns.barplot(data=df, y=stat, x="mouse", ax=ax[0])

            h_order = None
            if conditions_order:
                h_order = get_order(list(conditions_order.values()))
            sns.barplot(
                data=df, hue="condition", y=stat, x="mouse", ax=ax[1], hue_order=h_order
            )

            plt.legend(loc="upper right")
        else:
            df = df.groupby("mouse").mean().reset_index()
            sns.barplot(
                data=df,
                y=stat,
                x="mouse",
            )

        plt.show()

    def get_stat_list(self):
        """
        Function for getting list of statistics
        :return: list of statistics
        """
        return self.data.columns.tolist()

    def save_results(self, path):
        """
        Function for saving all data
        :param path: path to target folder
        """
        self.data.to_excel(path + "/all_data.xlsx")
        self.data_reduced.to_excel(path + "/reduced_data.xlsx")

    def save_stats_deviation(self, path):
        """
        Function for saving all data
        :param path: path to target folder
        """
        conditions = set(
            ["all"] + [self.params[session]["condition"] for session in self.params]
        )

        features_mad = []
        for condition in conditions:
            cond_div = self.get_stats_deviation(condition)
            cond_div = cond_div.rename(condition)
            features_mad.append(cond_div)

        features_mad = pd.concat(features_mad, axis=1)

        features_mad.to_excel(path + "/stats_deviation.xlsx")


def iqr(x):
    """
    Function for computing interquartile range
    :param x: pd.Series with data
    :return: interquartile range
    """
    return x.quantile(0.75) - x.quantile(0.25)


def q95(x):
    """
    Function for computing 95-quantile
    :param x: pd.Series with data
    :return: 95-quantile
    """
    return x.quantile(0.95)


def q5(x):
    """
    Function for computing 5-quantile
    :param x: pd.Series with data
    :return: 5-quantile
    """
    return x.quantile(0.05)


def get_order(orders):
    """
    Function for creating general condition order from several
    :param orders: list of several condition orders
    :return: list of general condition order
    """
    prev = {}
    for x in orders:
        for i, y in enumerate(x):
            ptr = prev.get(y, [])
            ptr.extend(x[:i])
            prev[y] = ptr

    prev_all = {}
    for key, val in prev.items():
        ptr = []
        for y in val:
            ptr.append(y)
            ptr.extend(prev[y])
        prev_all[key] = set(ptr)

    order = []
    for key, val in prev_all.items():
        order.append([val, key])

    order.sort(key=lambda x: len(x[0]))

    return [x[1] for x in order]
