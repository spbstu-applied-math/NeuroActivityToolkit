import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from analysis.functions import corr_df_to_distribution, active_df_to_dict
from analysis.active_state import ActiveStateAnalyzer

sns.set(color_codes=True)


class ShuffleAnalysis:
    """
    Class for randomly shuffling neuron activity
    """

    def __init__(self, path_to_data, sessions, shuffle_fraction=1.0, verbose=True):
        """
        Initialization function
        :param path_to_data: path to directory with sessions folders
        :param sessions: dict with information about sessions
                        {session 1:
                            {'path': path to session folder,
                             'fps': fps of session},
                         session 2: ...
                         }
        :param shuffle_fraction: fraction of shuffled neurons
        :param verbose: visualization of interim results and progress
        """

        self.dates = sessions
        self.path_to_data = path_to_data
        self.verbose = verbose

        self.original_data = {}
        self.shuffled_data = {}

        for date in tqdm(self.dates, disable=(not self.verbose)):
            session_path = self.dates[date]["path"]
            ma_o = ActiveStateAnalyzer(
                f"{self.path_to_data}/{session_path}/minian/", self.dates[date]["fps"]
            )
            ma_o.active_state_df = pd.read_excel(
                f"{self.path_to_data}/{session_path}/results/active_states_spike.xlsx",
                index_col=0,
            ).astype(bool)

            ma_o.active_state = active_df_to_dict(ma_o.active_state_df)

            ma_s = ActiveStateAnalyzer(
                f"{self.path_to_data}/{session_path}/minian/", self.dates[date]["fps"]
            )

            ma_s.active_state_df = ma_o.active_state_df.copy()
            idx = ma_s.active_state_df.sample(axis=1, frac=shuffle_fraction).columns
            ma_s.active_state_df[idx] = (
                ma_s.active_state_df[idx].apply(self.shuffle_signal).astype(bool)
            )
            ma_s.active_state = active_df_to_dict(ma_s.active_state_df)

            self.original_data[date] = ma_o
            self.shuffled_data[date] = ma_s

    @staticmethod
    def shuffle_signal(signal):
        """
        Function for randomly shuffle signal.
        Duration of active states and interval between them change randomly.
        :param signal: binary series of signal
        :return: shuffled signal
        """

        signal_diff = np.diff([0] + signal.tolist() + [0])
        starts = np.where(signal_diff == 1)[0]
        ends = np.where(signal_diff == -1)[0]

        df = pd.DataFrame({"start": starts, "len": ends - starts})

        intervals = np.random.randint(100, size=len(df) + 1)
        intervals = intervals / intervals.sum() * len(signal[signal == 0])
        intervals = intervals.astype(int)
        intervals[-1] += len(signal[signal == 0]) - intervals.sum()

        order = np.arange(len(df))
        np.random.shuffle(order)

        shuff = []

        for i in range(len(order)):
            shuff += [0] * intervals[i]
            shuff += [1] * df.iloc[order[i]]["len"]

        shuff += [0] * intervals[-1]

        return shuff

    def correlation_ptp(self, corr_type="active", position=False):
        """
        Function for plotting correlation range
        :param corr_type: type of correlation
                * active
                * active_acc
        :param position: consideration of spatial position
        """
        df = pd.DataFrame(columns=["date", "model", "values"])

        for date in tqdm(self.dates):
            corr = corr_df_to_distribution(
                self.original_data[date].get_correlation(corr_type, position)
            )
            df = pd.concat([
                df,
                pd.DataFrame({"date": date, "model": "original", "values": corr})
            ])

            corr = corr_df_to_distribution(
                self.shuffled_data[date].get_correlation(corr_type, position)
            )
            df = pd.concat([
                df,
                pd.DataFrame({"date": date, "model": "shuffle", "values": corr})
            ])

        df = df.fillna(0)

        ptp = df.groupby(["model", "date"]).agg({"values": np.ptp}).reset_index()
        diff = (
            ptp.groupby(["date"]).agg({"model": list, "values": lambda x: -np.diff(x)[0]}).reset_index()
        )

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].set_title("Correlation range", fontsize=16)
        sns.barplot(data=ptp, hue="model", x="date", y="values", ax=ax[0])
        ax[0].tick_params(axis="x", rotation=45)
        ax[0].tick_params(axis="both", labelsize=13)
        ax[0].set_ylabel("Range", fontsize=14)
        ax[0].set_xlabel("Session", fontsize=14)

        ax[1].set_title("Change of range", fontsize=16)
        sns.barplot(data=diff, x="date", y="values", ax=ax[1])
        ax[1].tick_params(axis="x", rotation=45)
        ax[1].tick_params(axis="both", labelsize=13)
        ax[1].set_ylabel("Change", fontsize=14)
        ax[1].set_xlabel("Session", fontsize=14)

        plt.show()

    def statistic_info(self, stat_type="network_spike_rate"):
        """
        Function for plotting statistic info
        :param stat_type: type of statistic
            * network_spike_rate (default)
            * network_spike_peak
        """
        values = []
        models = []
        dates_df = []

        for data, name in zip(
            [self.original_data, self.shuffled_data], ["original", "shuffle"]
        ):
            for date in self.dates:
                if stat_type == "network_spike_peak":
                    val = data[date].network_spike_peak(1).T["peak"].tolist()
                else:
                    val = data[date].network_spike_rate(1).T["spike rate"].tolist()

                values += val
                models += [name] * len(val)
                dates_df += [date] * len(val)

        df = pd.DataFrame({"values": values, "model": models, "date": dates_df})
        maximum = df.groupby(["model", "date"]).agg({"values": np.max}).reset_index()
        mean = df.groupby(["model", "date"]).agg({"values": np.mean}).reset_index()

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].set_title(f"{stat_type} maximum", fontsize=16)
        sns.barplot(data=maximum, hue="model", x="date", y="values", ax=ax[0])
        ax[0].tick_params(axis="x", rotation=45)
        ax[0].tick_params(axis="both", labelsize=13)
        ax[0].set_ylabel(f"{stat_type} maximum value", fontsize=14)
        ax[0].set_xlabel("Session", fontsize=14)

        ax[1].set_title(f"{stat_type} mean", fontsize=16)
        sns.barplot(data=mean, hue="model", x="date", y="values", ax=ax[1])
        ax[1].tick_params(axis="x", rotation=45)
        ax[1].tick_params(axis="both", labelsize=13)
        ax[1].set_ylabel(f"{stat_type} mean value", fontsize=14)
        ax[1].set_xlabel("Session", fontsize=14)

        plt.show()

    def show_shuffling(self, date):
        """
        Function for plotting original and shuffled data
        :param date: name of session
        """
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))

        ax[0].set_title("Original data", fontsize=20)
        sns.heatmap(self.original_data[date].active_state_df.T, cbar=False, ax=ax[0])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_ylabel("Neurons", fontsize=18)

        ax[1].set_title("Shuffled data", fontsize=20)
        sns.heatmap(self.shuffled_data[date].active_state_df.T, cbar=False, ax=ax[1])
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_xlabel("Time \u2192", fontsize=18)
        ax[1].set_ylabel("Neurons", fontsize=18)

        plt.show()
