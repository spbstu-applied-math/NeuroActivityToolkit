import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kstest, mannwhitneyu, ks_2samp, wilcoxon
from tqdm.notebook import tqdm

from analysis.functions import corr_df_to_distribution, active_df_to_dict
from analysis.minian import MinianAnalysis

sns.set(color_codes=True)


class StatTests:
    """
    Class for comparing and analysing different sessions.
    Based on results of MinianAnalysis
    """

    def __init__(self, path_to_data, sessions, verbose=True):
        """
        Initialization function
        :param path_to_data: path to directory with sessions folders
        :param sessions: dict with information about sessions
                        {session 1:
                            {'path': path to session folder,
                             'fps': fps of session},
                         session 2: ...
                         }
        :param verbose: visualization of interim results and progress
        """
        self.base_correlations = [
            "signal",
            "diff",
            "active",
            "active_acc",
        ]

        self.verbose = verbose

        self.corr_dfs = {}
        self.corr_distr = {}
        self.signals = {}
        self.params = sessions
        self.path_to_data = path_to_data

        for date in self.params:
            session_path = self.params[date]["path"]
            ma = MinianAnalysis(
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

            self.signals[date] = ma

            distr_dict = {}
            df_dict = {}
            for t in self.base_correlations:
                corr_df = ma.get_correlation(t)

                df_dict[t] = corr_df.copy()
                distr_dict[t] = corr_df_to_distribution(corr_df)

                corr_df = ma.get_correlation(t, position=True)

                df_dict[t + "_position"] = corr_df.copy()
                distr_dict[t + "_position"] = corr_df_to_distribution(corr_df)

            self.corr_dfs[date] = df_dict

            self.corr_distr[date] = distr_dict

        br = pd.DataFrame()
        nsr = pd.DataFrame()
        nsp = pd.DataFrame()
        nsd = pd.DataFrame()

        for date in tqdm(self.signals, disable=(not self.verbose)):
            br = br.append(
                self.signals[date].burst_rate()["activations per min"].rename(date)
            )
            nsr = nsr.append(
                self.signals[date].network_spike_rate(1).T["spike rate"].rename(date)
            )
            nsp = nsp.append(
                self.signals[date].network_spike_peak(1).T["peak"].rename(date)
            )
            nsd = nsd.append(
                self.signals[date]
                .network_spike_duration(np.arange(3, 63, 3), verbose=False)[
                    "Network spike duration"
                ]
                .rename(date)
            )

        self.br = br.T
        self.nsr = nsr.T
        self.nsp = nsp.T
        self.nsd = nsd.T

        self.all_corr = pd.DataFrame()

        for date in self.params:
            for t in self.base_correlations:
                self.all_corr = self.all_corr.append(
                    pd.DataFrame(
                        {"values": self.corr_distr[date][t], "date": date, "type": t}
                    )
                )
                self.all_corr = self.all_corr.append(
                    pd.DataFrame(
                        {
                            "values": self.corr_distr[date][t + "_position"],
                            "date": date,
                            "type": t,
                        }
                    )
                )

    def show_correlation_distribution(self, method="kde", position=False):
        """
        Function for plotting correlation distribution of sessions
        :param method: type of chart
                    * 'kde'. KDE represents the data using a continuous probability density curve. (Default)
                    * 'box'. Box-and-whisker plot
                    * 'hist'. Histograms
        :param position: consideration of spatial position
        """
        if position:
            corr_types = [x + "_position" for x in self.base_correlations]
        else:
            corr_types = self.base_correlations

        if method == "box":
            plt.figure(figsize=(12, 10))
            sns.boxplot(
                data=self.all_corr[self.all_corr["type"].isin(corr_types)],
                y="values",
                hue="date",
                x="type",
            )
            plt.title("Correlation distribution", fontsize=17)
            plt.xlabel("type", fontsize=16)
            plt.ylabel("Correlation coefficient", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=16)
            plt.show()

        elif method == "hist":
            fig, axs = plt.subplots(2, 2, figsize=(18, 15))

            for t, ax in zip(corr_types, axs.flatten()):
                df = pd.DataFrame()

                for date in self.params:
                    label = f"{len(self.corr_dfs[date][t])} neurons"
                    df = df.append(
                        pd.DataFrame(self.corr_distr[date][t]).T.rename(
                            index=lambda x: label
                        )
                    )

                sns.histplot(df.T, stat="percent", bins=20, ax=ax)

                ax.set_title(f"Correlation distribution {t}", fontsize=17)
                ax.set_xlabel("Correlation coefficient", fontsize=16)
                ax.set_ylabel("Density", fontsize=16)
                ax.tick_params(axis="both", labelsize=14)
            plt.show()

        else:
            fig, axs = plt.subplots(2, 2, figsize=(18, 15))

            for t, ax in zip(corr_types, axs.flatten()):
                for date in self.params:
                    sns.kdeplot(
                        self.corr_distr[date][t],
                        label=f"{len(self.corr_dfs[date][t])} neurons",
                        hue_norm=True,
                        linewidth=3,
                        ax=ax,
                    )

                ax.set_title(f"Correlation distribution {t}", fontsize=17)
                ax.set_xlabel("Correlation coefficient", fontsize=16)
                ax.set_ylabel("Density", fontsize=16)
                ax.tick_params(axis="both", labelsize=14)
                ax.legend(fontsize=16)
            plt.show()

    def __distr_test(self, data):
        """
        Function for testing the identity of distributions
        :param data: dict of DataFrames. dict - different types of date. DataFrames columns - different sessions
        :return: results of testing
        """
        dist_tets = pd.DataFrame(columns=[True, False])

        n = (len(self.params) ** 2 - len(self.params)) / 2

        alpha = 0.05 / n

        for test, title in zip(
            [mannwhitneyu, ks_2samp], ["Mann–Whitney U test", "Kolmogorov–Smirnov test"]
        ):
            test_res = {}
            for t in data:
                df = data[t]
                lst = []
                dates = list(self.params.keys())
                for i in range(len(dates) - 1):
                    for j in range(i + 1, len(dates)):
                        p = test(df[dates[i]].dropna(), df[dates[j]].dropna())[1]
                        lst.append(p)
                test_res[t] = (pd.Series(lst) > alpha).value_counts()

            test_res = (
                pd.DataFrame(test_res).T.reset_index().rename(columns={"index": "type"})
            )
            test_res["test"] = title
            dist_tets = dist_tets.append(test_res)

        dist_tets = (
            dist_tets.groupby(["test", "type"])
            .agg({True: "first", False: "first"})
            .fillna(0)
            .astype(int)
        )

        return dist_tets

    def __norm_test(self, data):
        """
        Function for testing normality of distributions
        :param data: dict of DataFrames. dict - different types of date. DataFrames columns - different sessions
        :return: results of testing
        """
        test_df = pd.DataFrame(columns=[True, False])

        alpha = 0.05 / len(self.params)

        test_res = {}
        for t in data:
            df = data[t]
            lst = []
            for i in self.params:
                lst.append(kstest(df[i], "norm")[1])
                # lst.append(normaltest(df[dates[i]])[1])

            test_res[t] = (pd.Series(lst) > alpha).value_counts()

        test_df = test_df.append(pd.DataFrame(test_res).T).fillna(0)

        return test_df

    def __get_test_data(self, data_type):
        """
        Function for preparing data for tests
        :param data_type: type of tested data
                * 'stat'. Distribution of statistics
                * 'corr'. Distribution of correlation
        :return: prepared data
        """
        if data_type == "corr":
            data = {}
            for t in self.base_correlations:
                type_df = pd.DataFrame()
                for date in self.params:
                    type_df = type_df.append(
                        pd.Series(self.corr_distr[date][t], name=date)
                    )
                data[t] = type_df.T

                type_df = pd.DataFrame()
                for date in self.params:
                    type_df = type_df.append(
                        pd.Series(self.corr_distr[date][t + "_position"], name=date)
                    )
                data[t + "_position"] = type_df.T

            return data
        else:
            data = {}
            for df, name in zip(
                [self.br, self.nsr, self.nsp, self.nsd],
                [
                    "Burst rate",
                    "Network spike rate",
                    "Network spike peak",
                    "Network spike duration",
                ],
            ):
                data[name] = df
            return data

    def get_test(self, test_type="distr", data_type="stat"):
        """
        Function to perform the test
        :param test_type: type of test
                * 'distr'. Testing the identity of distributions
                * 'norm'. Testing normality of distributions
        :param data_type: type of tested data
                * 'stat'. Distribution of statistics
                * 'corr'. Distribution of correlation
        :return:
        """
        data = self.__get_test_data(data_type)
        if test_type == "norm":
            return self.__norm_test(data)
        else:
            return self.__distr_test(data)

    def show_stats_distribution(self):
        """
        Function for plotting statistics distribution of sessions
        """
        fig, axs = plt.subplots(2, 2, figsize=(18, 15))
        for df, name, ax in zip(
            [self.br, self.nsr, self.nsp, self.nsd],
            [
                "Burst rate",
                "Network spike rate",
                "Network spike peak",
                "Network spike duration",
            ],
            axs.flatten(),
        ):
            for day in df:
                sns.kdeplot(
                    df[day],
                    label=f"{len(self.signals[day].active_state)} neurons",
                    hue_norm=True,
                    linewidth=3,
                    ax=ax,
                )

            ax.set_title(f"{name} distribution", fontsize=17)
            ax.set_xlabel(f"{name} value", fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(fontsize=16)

        plt.show()

    def show_network_degree(self, interval=(0, 1), step=0.01, position=False):
        """
        Function for plotting network degree
        :param interval: range of x-axis
        :param step: step on x-axis
        :param position: consideration of spatial position
        """
        degree_of_network = self.get_network_degree(interval, step, position)

        fig, axs = plt.subplots(2, 2, figsize=(18, 15))

        if position:
            corr_types = [x + "_position" for x in self.base_correlations]
        else:
            corr_types = self.base_correlations

        for corr_type, ax in zip(corr_types, axs.flatten()):
            ax.set_xlabel("Threshold", fontsize=16)
            ax.set_ylabel("Correlation coefficient value", fontsize=16)
            ax.set_title(f"Network degree ({corr_type})", fontsize=18)
            ax.tick_params(axis="both", labelsize=14)
            thrs = np.arange(interval[0], interval[1], step)
            for date in self.params:
                ax.plot(
                    thrs,
                    degree_of_network[corr_type][date],
                    lw=2.5,
                    label=f"{date} - {len(self.corr_dfs[date][corr_type])} neurons",
                )

            ax.legend(fontsize=15)
        plt.show()

    def get_network_degree(self, interval=(0, 1), step=0.01, position=False):
        """
        Function for computing network degree
        :param interval: range of x-axis
        :param step: step on x-axis
        :param position: consideration of spatial position
        :return: dict with network degree
        """
        degree_of_network = {}

        if position:
            corr_types = [x + "_position" for x in self.base_correlations]
        else:
            corr_types = self.base_correlations

        for corr_type in corr_types:
            values = {}
            thrs = np.arange(interval[0], interval[1], step)
            for date in self.params:
                corr = np.array(self.corr_distr[date][corr_type])
                values[date] = [(corr > thr).sum() / len(corr) for thr in thrs]

            degree_of_network[corr_type] = values
        return degree_of_network

    def show_distribution_of_connectivity(self, q=0.9, position=False):
        """
        Function for plotting distribution of connectivity
        :param q: quantile level for all values of this type, the value of which will be the threshold value
        :param position: consideration of spatial position
        """
        degree_distribution = self.get_distribution_of_connectivity(q, position)
        fig, axs = plt.subplots(2, 2, figsize=(18, 15))

        if position:
            corr_types = [x + "_position" for x in self.base_correlations]
        else:
            corr_types = self.base_correlations

        for corr_type, ax in zip(corr_types, axs.flatten()):
            ax.set_xlabel("Percent of correlated neurons", fontsize=16)
            ax.set_ylabel("Percent of neurons", fontsize=16)
            ax.set_title(f"Distribution of connectivity ({corr_type})", fontsize=18)
            ax.tick_params(axis="both", labelsize=14)
            for date in self.params:
                degree = degree_distribution[corr_type][date]
                num_of_neurons = len(self.corr_dfs[date][corr_type])
                sns.lineplot(
                    x=degree.index / num_of_neurons,
                    y=degree,
                    lw=2.2,
                    label=f"{date} - {num_of_neurons} neurons",
                    ax=ax,
                )

            ax.legend(fontsize=15)
        plt.show()

    def get_distribution_of_connectivity(self, q=0.9, position=False):
        """
        Function for computing distribution of connectivity
        :param q: quantile level for all values of this type, the value of which will be the threshold value
        :param position: consideration of spatial position
        :return: share of coactive neurons for each other
        """
        if position:
            corr_types = [x + "_position" for x in self.base_correlations]
        else:
            corr_types = self.base_correlations

        degree_distribution = {}
        for corr_type in corr_types:
            total_distr = []
            for date in self.params:
                total_distr += self.corr_distr[date][corr_type]

            thr = np.quantile(total_distr, q=q)

            degrees = {}
            for date in self.params:
                corr_df = self.corr_dfs[date][corr_type]
                degree = ((corr_df > thr).sum() - 1).value_counts().sort_index() / len(
                    corr_df
                )
                degrees[date] = degree

            degree_distribution[corr_type] = degrees
        return degree_distribution

    def __wilcoxon_test(self, data, alpha=0.005):
        """
        Wilcoxon test
        :param data: dict of DataFrames. dict - different types of date. DataFrames columns - different sessions
        :param alpha: p_value threshold
        """
        dist_tets = pd.DataFrame(columns=[True, False])

        n = (len(self.params) ** 2 - len(self.params)) / 2

        alpha = alpha / n

        test_res = {}
        for t in data:
            lst = []
            dates = list(self.params.keys())
            for i in range(len(dates) - 1):
                for j in range(i + 1, len(dates)):
                    p = wilcoxon(data[t][dates[i]], data[t][dates[j]])[1]
                    lst.append(p)
            test_res[t] = (pd.Series(lst) > alpha).value_counts()

        test_res = (
            pd.DataFrame(test_res).T.reset_index().rename(columns={"index": "type"})
        )
        dist_tets = dist_tets.append(test_res)
        dist_tets = (
            dist_tets.groupby(["type"])
            .agg({True: "first", False: "first"})
            .fillna(0)
            .astype(int)
        )
        return dist_tets

    def get_nd_test(self, interval=(0, 1), step=0.01, position=False):
        """
        Function for testing network degree identity
        :param interval: range of x-axis
        :param step: step on x-axis
        :param position: consideration of spatial position
        """
        data = self.get_network_degree(interval=interval, step=step, position=position)
        return self.__wilcoxon_test(data)

    def get_connectivity_distr_test(self, q=0.9, position=False):
        """
        Function for testing network degree distribution identity
        :param q: quantile level for all values of this type, the value of which will be the threshold value
        :param position: consideration of spatial position
        """
        data = self.get_distribution_of_connectivity(q=q, position=position)
        return self.__distr_test(data)
