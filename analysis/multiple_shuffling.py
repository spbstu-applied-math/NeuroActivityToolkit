from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.shuffling import ShuffleAnalysis
from analysis.functions import corr_df_to_distribution

sns.set(color_codes=True)


class MultipleShuffler:
    """
    Class for multiple generation of shuffled signals and their analysis
    """

    def __init__(
        self,
        path_to_data,
        sessions,
        num_of_shuffles=10,
        shuffle_fractions=None,
        correlation_type="active",
        verbose=True,
    ):
        """
        Initialization function
        :param path_to_data: path to directory with sessions folders
        :param sessions: dict with information about sessions
                        {session 1:
                            {'path': path to session folder,
                             'fps': fps of session},
                         session 2: ...
                         }
        :param num_of_shuffles: number of shuffles per case
        :param shuffle_fractions: fraction of shuffled neurons
        :param correlation_type: type of correlation for analysis
                * active
                * active_acc
        :param verbose: visualization of interim results and progress
        """

        if shuffle_fractions is None:
            shuffle_fractions = [0.25, 0.5, 0.75, 1.0]

        self.path_to_data = path_to_data
        self.dates = sessions
        self.num_of_shuffles = num_of_shuffles
        self.shuffle_fractions = shuffle_fractions
        self.correlation_type = correlation_type
        self.verbose = verbose

        self.models = self._generate_models()

        self.stat_df = self._create_stats()

        self.corr_df = self._create_corrs()

    def _generate_models(self):
        """
        Function for generating shuffled models
        """
        models = {
            0: {
                0: ShuffleAnalysis(
                    self.path_to_data,
                    self.dates,
                    shuffle_fraction=0,
                    verbose=False,
                )
            }
        }

        for shuffle_fraction in tqdm(
            self.shuffle_fractions,
            disable=(not self.verbose),
            desc="Generating models...",
        ):
            ptr = {}
            for i in range(self.num_of_shuffles):
                ptr[i] = ShuffleAnalysis(
                    self.path_to_data,
                    self.dates,
                    shuffle_fraction=shuffle_fraction,
                    verbose=False,
                )
            models[shuffle_fraction] = ptr
        return models

    def _create_stats(self):
        """
        Function for computing statistics of generated models
        """
        stat_df = pd.DataFrame()
        for shuffle_fraction in tqdm(
            self.models, disable=(not self.verbose), desc="Computing statistics..."
        ):
            for i in self.models[shuffle_fraction]:
                model = self.models[shuffle_fraction][i]
                for date in model.shuffled_data:
                    ptr_df = pd.DataFrame()

                    ptr_df["network spike rate"] = (
                        model.shuffled_data[date]
                        .network_spike_rate(1)
                        .T["spike rate"]
                        .tolist()
                    )
                    ptr_df["network spike peak"] = (
                        model.shuffled_data[date]
                        .network_spike_peak(1)
                        .T["peak"]
                        .tolist()
                    )
                    ptr_df["date"] = date
                    ptr_df["shuffle_fraction"] = shuffle_fraction
                    ptr_df["attempt"] = i

                    stat_df = stat_df.append(ptr_df)

        return stat_df.reset_index(drop=True)

    def _create_corrs(self):
        """
        Function for computing correlations of generated models
        """
        corr_df = pd.DataFrame()

        for shuffle_fraction in tqdm(
            self.models, disable=(not self.verbose), desc="Computing correlations..."
        ):
            for i in self.models[shuffle_fraction]:
                model = self.models[shuffle_fraction][i]

                for date in model.shuffled_data:
                    for position in [False, True]:
                        ptr_df = pd.DataFrame()

                        ptr_df["corr"] = corr_df_to_distribution(
                            model.shuffled_data[date].get_correlation(
                                self.correlation_type, position=position
                            )
                        )
                        ptr_df["date"] = date
                        ptr_df["position"] = position
                        ptr_df["attempt"] = i
                        ptr_df["shuffle_fraction"] = shuffle_fraction

                        corr_df = corr_df.append(ptr_df)

        corr_df["corr"] = corr_df["corr"].fillna(0)

        return corr_df.reset_index(drop=True)

    def show_day_mean_correlation_range(self, position=False):
        """
        Function for plotting average correlation range by sessions
        :param position: consideration of spatial position
        """
        ptr = (
            self.corr_df[self.corr_df["position"] == position]
            .groupby(["shuffle_fraction", "date", "attempt"])
            .agg({"corr": np.ptp})
            .groupby(["date", "shuffle_fraction"])
            .agg({"corr": np.mean})
            .reset_index()
        )

        plt.figure(figsize=(15, 8))
        sns.barplot(data=ptr, x="date", y="corr", hue="shuffle_fraction")
        plt.xlabel("Session", fontsize=16)
        plt.ylabel("Range of correlation values", fontsize=16)
        plt.title(f"Mean correlation range ({self.correlation_type})", fontsize=18)
        plt.tick_params(axis="both", labelsize=14)
        plt.legend(
            title="Shuffle ratio", fontsize=14, bbox_to_anchor=(1, 1)
        )  # , loc= 'lower right', shadow = True)
        plt.show()

    def show_position_mean_correlation_range(self):
        """
        Function for plotting average correlation range by with/without position consideration
        """
        ptr = (
            self.corr_df.groupby(["position", "shuffle_fraction", "date", "attempt"])
            .agg({"corr": np.ptp})
            .groupby(["position", "shuffle_fraction"])
            .agg({"corr": np.mean})
            .reset_index()
        )

        plt.figure(figsize=(15, 8))
        sns.barplot(data=ptr, x="position", y="corr", hue="shuffle_fraction")
        plt.xlabel("Session", fontsize=16)
        plt.ylabel("Range of correlation values", fontsize=16)
        plt.title(f"Mean correlation range ({self.correlation_type})", fontsize=18)
        plt.tick_params(axis="both", labelsize=14)
        plt.legend(
            title="Shuffle ratio", fontsize=14, bbox_to_anchor=(1, 1)
        )  # , loc= 'lower right', shadow = True)
        plt.show()

    def show_mean_statistic_peak(
        self, daily=False, statistic_type="network spike peak"
    ):
        """
        Function for plotting average peak of statistic
        :param daily: daily average or only average by  shuffle fraction
        :param statistic_type: type of statistic
                * network spike peak (default)
                * network spike rate
        """
        plt.figure(figsize=(15, 8))

        if daily:
            ptr = (
                self.stat_df.groupby(["shuffle_fraction", "date", "attempt"])
                .agg({f"{statistic_type}": np.max})
                .groupby(["date", "shuffle_fraction"])
                .agg({f"{statistic_type}": np.mean})
                .reset_index()
            )
            sns.barplot(
                data=ptr, x="date", y=f"{statistic_type}", hue="shuffle_fraction"
            )
            plt.xlabel("Session", fontsize=16)
            plt.legend(
                title="Shuffle ratio", fontsize=14, bbox_to_anchor=(1, 1)
            )  # , loc= 'lower right', shadow = True)
        else:
            ptr = (
                self.stat_df.groupby(["shuffle_fraction", "date", "attempt"])
                .agg({f"{statistic_type}": np.max})
                .groupby(["shuffle_fraction"])
                .agg({f"{statistic_type}": np.mean})
                .reset_index()
            )
            sns.barplot(data=ptr, x="shuffle_fraction", y=f"{statistic_type}")
            plt.xlabel("Shuffle fraction", fontsize=16)

        plt.ylabel(f"{statistic_type} peak", fontsize=16)
        plt.title(f"Mean {statistic_type} peaks", fontsize=18)
        plt.tick_params(axis="both", labelsize=14)
        plt.show()

    def save_results(self, path):
        """
        Function for saving connectivity
        :param path: path to target folder
        """
        self.stat_df.to_excel(path + "/ms_stats.xlsx")

        agg_stat = (
            self.stat_df.groupby(["date", "shuffle_fraction", "attempt"])
            .agg(["mean", "max", "std"])
            .reset_index()
        )
        agg_stat = (
            agg_stat.groupby(["date", "shuffle_fraction"])
            .agg("mean")
            .drop(columns=["attempt"])
        )

        agg_stat.to_excel(path + "/ms_stats_aggregated.xlsx")

        agg_corr = (
            self.corr_df.groupby(["position", "date", "shuffle_fraction", "attempt"])
            .agg(["mean", "std", np.ptp])
            .reset_index()
            .groupby(["position", "date", "shuffle_fraction"])
            .agg("mean")
            .drop(columns=["attempt"])
        )

        agg_corr.to_excel(path + "/ms_corr_aggregated.xlsx")
