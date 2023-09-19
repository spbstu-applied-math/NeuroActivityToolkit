import numpy as np
import zarr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from scipy import ndimage
from os import path, mkdir
from analysis.functions import crosscorr
from pyitlib.discrete_random_variable import entropy_conditional, entropy_joint
from scipy.cluster.hierarchy import linkage, fcluster
from analysis.functions import corr_df_to_distribution
import itertools

sns.set(color_codes=True)


class ActiveStateAnalyzer:
    """
    Class for processing data after minian.
    It solves tasks:
    * active state detection
    * calculation of statistics
    * visualisation
    """

    def __init__(self, path_to_data, fps, path_to_results=None):
        """
        Initialization function
        :param path_to_data: path to minian output directory
        :param fps: frames per second
        :path_to_results: path to folder for results
        """
        if path_to_results is None:
            path_to_results = path_to_data + "../results"
        signals = zarr.open_group(path_to_data + "C.zarr")
        self.signals = (
            pd.DataFrame(signals.C)
            .set_index(pd.DataFrame(signals.unit_id)[0].rename("unit_id"))
            .T
        )

        positions = zarr.open_group(path_to_data + "A.zarr")
        positions_centers = np.array(
            [ndimage.measurements.center_of_mass(x) for x in np.array(positions.A)]
        )
        self.positions = pd.DataFrame(
            {
                "unit_id": positions.unit_id,
                "x": positions_centers[:, 0],
                "y": positions_centers[:, 1],
            }
        ).set_index("unit_id")

        self.fps = fps

        self.smooth_signals = None
        self.diff = None
        self.smooth_diff = None

        self.active_state = {}
        self.active_state_df = pd.DataFrame()

        self.type_of_activity = None
        self.results_folder = path_to_results

    @staticmethod
    def __get_active_states(signal, threshold):
        """
        Function for determining the active states of the input signal
        :param signal: signal values
        :param threshold: threshold for active state
        :return: list of lists with active states indexes
        """
        res = []
        sleep = signal[signal <= threshold].reset_index()
        sleep_min = sleep["index"].min()

        if len(sleep) == 0:
            return [np.arange(0, len(signal), dtype="int").tolist()]
        elif sleep_min > 0:
            # res.append(np.arange(0, sleep_min + 1, dtype='int').tolist())
            res.append(np.arange(0, sleep_min, dtype="int").tolist())

        sleep["index_diff"] = sleep["index"].diff()

        changes = sleep[sleep["index_diff"] > 1].copy()

        if len(changes) == 0:
            return res

        changes["start"] = changes["index"] - changes["index_diff"] + 1
        changes["end"] = changes["index"]  # + 1

        res += changes.apply(
            lambda x: np.arange(x["start"], x["end"], dtype="int").tolist(), axis=1
        ).tolist()

        sleep_max = sleep["index"].max() + 1
        if sleep_max < len(signal):
            res.append(np.arange(sleep_max, len(signal), dtype="int").tolist())
        return res

    @staticmethod
    def __get_peaks(spike, decay=None, cold=0, warm=0):
        """
        Function for post-processing of found activity states.
        :param spike: list of growing parts
        :param decay: list of decaying parts
        :param cold: minimum duration of active state
        :param warm: minimum duration of inactive state
        :return: list of lists with active states indexes
        """
        peaks_idx = spike + decay if decay else spike
        peaks_idx.sort(key=lambda x: x[0])
        new_peaks = [peaks_idx[0]]

        for i in range(1, len(peaks_idx)):
            gap = peaks_idx[i][0] - new_peaks[-1][-1]
            if gap <= warm:
                new_peaks[-1] = (
                    new_peaks[-1]
                    + [i for i in range(new_peaks[-1][-1] + 1, peaks_idx[i][0])]
                    + peaks_idx[i]
                )
            else:
                new_peaks.append(peaks_idx[i])

        peaks = []
        for i in new_peaks:
            if len(i) > cold:
                peaks.append(i)

        return peaks

    def find_active_state(self, window, cold, warm, method="spike", verbose=True):
        """
        Function for preprocessing signals and determining the active states
        :param window: size of the moving window for smoothing
        :param cold: minimum duration of active state
        :param warm: minimum duration of inactive state
        :param method: ['spike', 'full'] type of active state
            spike - only the stage of intensity growth
            full - the stage of growth and weakening of intensity
        :param verbose: verbose
        """
        self.type_of_activity = method

        # rolling mean
        self.smooth_signals = self.signals.rolling(
            window=window, center=True, min_periods=0
        ).mean()

        # derivative
        self.diff = self.signals.diff()[1:].reset_index(drop=True)
        self.smooth_diff = self.smooth_signals.diff()[1:].reset_index(drop=True)

        for num in tqdm(self.smooth_signals.columns):
            y = self.smooth_diff[num]

            y_pos = y[y >= 0]
            mad_pos = np.mean(np.abs(np.median(y_pos) - y_pos))
            threshold_pos = np.median(y_pos) + mad_pos
            peaks_pos_idx = self.__get_active_states(y, threshold_pos)

            if method == "full":
                y_neg = -y[y <= 0]
                mad_neg = np.mean(np.abs(np.median(y_neg) - y_neg))
                threshold_neg = np.median(y_neg) + mad_neg
                peaks_neg_idx = self.__get_active_states(-y, threshold_neg)
            else:
                peaks_neg_idx = []

            peaks_idx = self.__get_peaks(
                peaks_pos_idx, peaks_neg_idx, cold=cold, warm=warm
            )

            self.active_state[num] = peaks_idx

            if verbose:
                signal = self.signals[num]

                plt.figure(figsize=(15, 10))
                plt.title(f"Neuron {num}", fontsize=18)

                plt.plot(signal, label="inactive")
                for peak in peaks_idx:
                    plt.plot(signal.iloc[peak], c="r")

                if len(peaks_idx) > 0:
                    plt.plot(signal.iloc[peaks_idx[0]], c="r", label="active")

                plt.plot(range(len(signal)), [0] * len(signal), c="b", lw=3)
                for peak in peaks_idx:
                    plt.plot(peak, [0] * len(peak), c="r", lw=3)

                plt.legend(fontsize=18)
                plt.show()

        for neuron in self.active_state:
            self.active_state_df[neuron] = [False] * len(self.signals)
            for peak in self.active_state[neuron]:
                self.active_state_df[neuron].iloc[peak] = True

    def get_active_state(self, neuron, window, cold, warm, method="spike"):
        """
        Function for preprocessing neuron signal and determining the active states
        :param neuron: neuron number
        :param window: size of the moving window for smoothing
        :param cold: minimum duration of active state
        :param warm: minimum duration of inactive state
        :param method: ['spike', 'full'] type of active state
            spike - only the stage of intensity growth
            full - the stage of growth and weakening of intensity
        """
        signal = self.signals[neuron]
        # rolling mean
        smooth_signal = signal.rolling(window=window, center=True, min_periods=0).mean()

        # derivative
        diff = signal.diff()[1:].reset_index(drop=True)
        smooth_diff = smooth_signal.diff()[1:].reset_index(drop=True)

        y_pos = smooth_diff[smooth_diff >= 0]
        mad_pos = np.mean(np.abs(np.median(y_pos) - y_pos))
        threshold_pos = np.median(y_pos) + mad_pos
        peaks_pos_idx = self.__get_active_states(smooth_diff, threshold_pos)

        if method == "full":
            y_neg = -smooth_diff[smooth_diff <= 0]
            mad_neg = np.mean(np.abs(np.median(y_neg) - y_neg))
            threshold_neg = np.median(y_neg) + mad_neg
            peaks_neg_idx = self.__get_active_states(-smooth_diff, threshold_neg)
        else:
            peaks_neg_idx = []

        peaks_idx = self.__get_peaks(peaks_pos_idx, peaks_neg_idx, cold=cold, warm=warm)

        plt.figure(figsize=(15, 10))
        plt.title(f"Neuron {neuron}", fontsize=22)

        plt.plot(signal, label="inactive", c="b", lw=4)
        for peak in peaks_idx:
            plt.plot(signal.iloc[peak], c="r", lw=4)

        if len(peaks_idx) > 0:
            plt.plot(signal.iloc[peaks_idx[0]], c="r", label="active", lw=4)

        plt.plot(range(len(signal)), [0] * len(signal), c="b", lw=5)
        for peak in peaks_idx:
            plt.plot(peak, [0] * len(peak), c="r", lw=5)

        plt.legend(fontsize=20)
        plt.show()

    def burst_rate(self):
        """
        Function for computing burst rate
        Burst rate - number of cell activations per minute
        """
        num_of_activations = []
        for neuron in self.active_state:
            num_of_activations.append(len(self.active_state[neuron]))

        burst_rate = pd.DataFrame({"num_of_activations": num_of_activations})

        burst_rate["activations per min"] = (
            burst_rate["num_of_activations"] / len(self.active_state_df) * self.fps * 60
        )

        burst_rate["activations per min"] = burst_rate["activations per min"].round(2)

        return burst_rate

    def network_spike_rate(self, period):
        """
        Function for computing network spike rate
        Network spike rate - percentage of active neurons per period
        :param period: period in seconds
        """
        step = period * self.fps
        vals = self.active_state_df.values

        nsr = {}
        for i in range(0, len(self.active_state_df), step):
            nsr[f"{i}-{i + step}"] = vals[i : i + step].any(axis=0).sum()

        nsr = pd.DataFrame(nsr, index=["spike rate"])
        nsr = nsr / len(self.active_state_df.columns) * 100

        return nsr

    def network_spike_duration(self, thresholds, verbose=False):
        """
        Function for computing network spike duration
        Network spike duration - duration when the percentage of active cells is above the set thresholds
        :param thresholds: threshold values in percentages
        :param verbose: progressbar
        """
        spike_durations = []
        vals = self.active_state_df.values
        for thr in tqdm(thresholds, disable=(not verbose)):
            percent_thr = len(self.active_state_df.columns) * thr / 100
            duration = (vals.sum(axis=1) > percent_thr).sum()
            spike_durations.append(duration)

        nsd_df = pd.DataFrame(
            {
                "percentage": thresholds,
                "Network spike duration": np.array(spike_durations) / len(vals),
            }
        )
        return nsd_df

    def network_spike_peak(self, period):
        """
        Function for computing network spike peak
        Network spike peak - maximum percentage of active cells per second
        :param period: period in seconds
        """
        step = period * self.fps
        vals = self.active_state_df.values

        spike_peaks = {}
        for i in range(0, len(vals), step):
            peak = vals[i : i + step].sum(axis=1).max()

            spike_peaks[f"{i}-{i + step}"] = peak

        nsp_df = pd.DataFrame(spike_peaks, index=["peak"])
        nsp_df = nsp_df / len(self.active_state_df.columns) * 100

        return nsp_df

    def show_burst_rate(self, max_bins=15):
        """
        Function for plotting burst rate
        Burst rate - number of cell activations per minute
        :param max_bins: maximum number of columns
        """
        burst_rate = self.burst_rate()

        plt.figure(figsize=(8, 6))
        plt.title("Burst rate", fontsize=17)

        if burst_rate["activations per min"].nunique() > max_bins:
            sns.histplot(
                data=burst_rate, x="activations per min", bins=max_bins, stat="percent"
            )
        else:
            burst_rate = (
                burst_rate["activations per min"]
                .value_counts(normalize=True)
                .mul(100)
                .rename("percent")
                .reset_index()
                .rename(columns={"index": "activations per min"})
            )
            sns.barplot(data=burst_rate, x="activations per min", y="percent")

        plt.xlabel("activations per min", fontsize=16)
        plt.ylabel("percent of neurons", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def show_network_spike_rate(self, period):
        """
        Function for plotting network spike rate
        Network spike rate - percentage of active neurons per period
        :param period: period in seconds
        """
        nsr = self.network_spike_rate(period)
        plt.figure(figsize=(8, 6))
        plt.title("Network spike rate", fontsize=17)
        sns.histplot(data=nsr.T, x="spike rate", stat="percent")
        plt.xlabel(f"percentage of active neurons per {period} second", fontsize=16)
        plt.ylabel("percent of time", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def show_network_spike_duration(self, thresholds):
        """
        Function for plotting network spike duration
        Network spike duration - duration when the percentage of active cells is above the set thresholds
        :param thresholds: threshold values in percentages
        """
        nsd_df = self.network_spike_duration(thresholds)
        nsd_df["Network spike duration"] *= 100
        plt.figure(figsize=(8, 6))
        plt.title("Network spike duration", fontsize=17)
        sns.barplot(data=nsd_df, x="percentage", y="Network spike duration")
        plt.xlabel("percentage of active neurons", fontsize=16)
        plt.ylabel("percent of time", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def show_network_spike_peak(self, period):
        """
        Function for plotting network spike peak
        Network spike peak - maximum percentage of active cells per second
        :param period: period in seconds
        """
        nsp_df = self.network_spike_peak(period)
        plt.figure(figsize=(8, 6))
        plt.title("Network spike peak", fontsize=17)
        sns.histplot(data=nsp_df.T, x="peak", bins=8, stat="percent")
        plt.xlabel(f"max percentage of active neurons per {period} second", fontsize=16)
        plt.ylabel("percent of time", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def save_burst_rate(self):
        """
        Function for saving burst rate
        Burst rate - number of cell activations per minute
        """
        burst_rate = self.burst_rate()
        burst_rate.to_excel(self.results_folder + "/burst_rate.xlsx")

    def save_network_spike_rate(self, period):
        """
        Function for saving network spike rate
        Network spike rate - percentage of active neurons per period
        :param period: period in seconds
        """
        nsr = self.network_spike_rate(period)
        nsr.to_excel(self.results_folder + "/network_spike_rate.xlsx")

    def save_network_spike_duration(self, thresholds):
        """
        Function for saving network spike duration
        Network spike duration - duration when the percentage of active cells is above the set thresholds
        :param thresholds: threshold values in percentages
        """
        nsd_df = self.network_spike_duration(thresholds)
        nsd_df.to_excel(self.results_folder + "/network_spike_duration.xlsx")

    def save_network_spike_peak(self, period):
        """
        Function for saving network spike peak
        Network spike peak - maximum percentage of active cells per second
        :param period: period in seconds
        """
        nsp_df = self.network_spike_peak(period)
        nsp_df.to_excel(self.results_folder + "/network_spike_peak.xlsx")

    def compute_nzsfi(self):
        """
        Function for computing NonZeroSpikeFramesIntersection
        :return: FataFrame with NonZeroSpikeFramesIntersection values
        """
        nzsfi = pd.DataFrame(columns=self.active_state_df.columns)
        for i in self.active_state_df:
            nzsfi[i] = [
                self.active_state_df[i].sum()
                / (self.active_state_df[i] & self.active_state_df[j]).sum()
                for j in self.active_state_df
            ]

        return nzsfi.T

    def compute_spike_accuracy(self):
        """
        Function for computing spike accuracy (intersection / union)
        :return: FataFrame with spike accuracy
        """
        spike_acc = np.ones(
            (self.active_state_df.shape[1], self.active_state_df.shape[1])
        )
        columns = self.active_state_df.columns.tolist()
        vals = self.active_state_df.values.T

        for i, x in enumerate(vals):
            if x.sum() == 0:
                spike_acc[i, :] = 0
                spike_acc[:, i] = 0
                spike_acc[i, i] = 1
                continue

            for j, y in enumerate(vals[i + 1 :]):
                intersec = (x & y).sum()
                union = (x | y).sum()

                spike_acc[i, i + j + 1] = intersec / union
                spike_acc[i + j + 1, i] = intersec / union

        return (
            pd.DataFrame(spike_acc).set_axis(columns, axis=0).set_axis(columns, axis=1)
        )

    def compute_cross_correlation(self, data, lag=0):
        """
        Function for computing cross_correlation
        :param lag: lag radius
        :param data: dataframe with series
        :return: FataFrame with cross_correlation
        """
        if lag == 0:
            return data.corr()

        cols = data.columns
        cross_corr_df = pd.DataFrame(columns=cols)
        for d, i in enumerate(cols):
            row = cross_corr_df[i].tolist()
            for j in cols[d:]:
                row.append(crosscorr(data[i], data[j], lag=lag))
            cross_corr_df = cross_corr_df.append(pd.Series(row, name=i).set_axis(cols))

        return cross_corr_df

    def compute_transfer_entropy(self):
        """
        Function for computing transfer_entropy
        :return: FataFrame with transfer_entropy
        """
        te = np.zeros((self.signals.shape[1], self.signals.shape[1]))
        columns = self.signals.columns.tolist()

        vals = self.signals.values
        vals = (vals / vals.max(axis=0) * 100).astype(int).T

        for i, target in tqdm(
            enumerate(vals), total=len(vals), desc="Transfer entropy computing..."
        ):
            x = target[1:]
            z = target[:-1]

            entr_cond = entropy_conditional(x, z)

            for j, source in enumerate(vals):
                y = source[:-1]

                te[j, i] = entr_cond - entropy_joint([x, y, z]) + entropy_joint([y, z])

        return pd.DataFrame(te).set_axis(columns, axis=0).set_axis(columns, axis=1)

    def get_correlation(self, method="signal", position=False, lag=0):
        """
        Function for computing correlation
        :param method: ['signal', 'diff', 'active', 'active_acc'] type of correlated sequences
            signal - Pearson correlation for signal intensity
            diff   - Pearson correlation for derivative of signal intensity
            active - Pearson correlation for binary segmentation of active states (depends on the chosen method for find_active_state)
            active_acc - ratio of intersection to union of active states (depends on the chosen method for find_active_state)
        :param position: consideration of spatial position
        :param lag: lag radius (not used for 'active_acc' method)
        """
        if method == "signal":
            corr_df = self.compute_cross_correlation(self.signals, lag)
        elif method == "diff":
            corr_df = self.compute_cross_correlation(self.smooth_diff, lag)
        elif method == "active":
            corr_df = self.compute_cross_correlation(self.active_state_df, lag)
        elif method == "active_acc":
            corr_df = self.compute_spike_accuracy()
        elif method == "transfer_entropy":
            corr_df = self.compute_transfer_entropy()
        else:
            print(f"Method {method} is not supported!")
            return

        if position:
            distances = self.positions.apply(
                lambda x: (
                    (self.positions["x"] - x["x"]) ** 2
                    + (self.positions["y"] - x["y"]) ** 2
                )
                ** (1 / 2),
                axis=1,
            )

            corr_df = (
                1 - 100 / (distances + 100)
            ) * corr_df.values  # 100 is 25% of distance

        return corr_df

    def save_active_states(self):
        """
        Function for saving active states matrix to results folder (depends on the chosen method for find_active_state)
        """
        if len(self.active_state_df) == 0:
            raise Exception("Active states are not set!")

        if not path.exists(self.results_folder):
            mkdir(self.results_folder)
        self.active_state_df.astype(int).to_excel(
            path.join(
                self.results_folder, f"active_states_{self.type_of_activity}.xlsx"
            )
        )

    def save_correlation_matrix(self, method="signal", position=False, lag=0):
        """
        Function for saving correlation matrix to results folder
        :param method: ['signal', 'diff', 'active', 'active_acc'] type of correlated sequences
            signal - Pearson correlation for signal intensity
            diff   - Pearson correlation for derivative of signal intensity
            active - Pearson correlation for binary segmentation of active states (depends on the chosen method for find_active_state)
            active_acc - ratio of intersection to union of active states (depends on the chosen method for find_active_state)
        :param position: consideration of spatial position
        :param lag: lag radius (not used for 'active_acc' method)
        """
        corr_df = self.get_correlation(method, position=position, lag=lag)

        if not path.exists(self.results_folder):
            mkdir(self.results_folder)

        corr_df.to_excel(
            path.join(
                self.results_folder,
                f"correlation_{self.type_of_activity}_{method}{'_position' if position else ''}.xlsx",
            )
        )

    def show_corr(self, threshold, method="signal", position=False, lag=0):
        """
        Function for plotting correlation distribution and map
        :param threshold: threshold for displayed correlation
        :param method: ['signal', 'diff', 'active', 'active_acc'] type of correlated sequences
            signal - Pearson correlation for signal intensity
            diff   - Pearson correlation for derivative of signal intensity
            active - Pearson correlation for binary segmentation of active states (depends on the chosen method for find_active_state)
            active_acc - ratio of intersection to union of active states (depends on the chosen method for find_active_state)
        :param position: consideration of spatial position
        :param lag: lag radius (not used for 'active_acc' method)
        """
        corr_df = self.get_correlation(method, position=position, lag=lag)

        fig, ax = plt.subplots(2, 2, figsize=(20, 20))

        c = 1
        corr = []
        for i, row in corr_df.iterrows():
            for j in corr_df.columns.tolist()[c:]:
                corr.append(row[j])

            c += 1

        sns.histplot(corr, stat="percent", ax=ax[0][0])
        ax[0][0].set_ylabel("Percent of connections", fontsize=20)
        ax[0][0].set_xlabel("Correlation coefficient", fontsize=20)
        ax[0][0].set_title(f"Correlation distribution for {method} method", fontsize=24)

        clusters = self.get_corr_clustering(corr_df)
        cluster_corr_df = corr_df[clusters[:, 0]].loc[clusters[:, 0]]
        clusters = pd.DataFrame(clusters).groupby(1).agg(len)[0]

        sns.heatmap(cluster_corr_df, vmin=-1, vmax=1, cmap="coolwarm", ax=ax[1][0])
        ax[1][0].set_title(f"Correlation heatmap", fontsize=24)

        sns.heatmap(cluster_corr_df > threshold, cmap="binary", cbar=False, ax=ax[1][1])
        ax[1][1].set_title(f"Correlation binary heatmap", fontsize=24)

        lw = len(corr_df) * 0.005
        for i, n in enumerate(clusters):
            start = clusters[:i].sum()
            x = np.array([start, start, start + n, start + n, start])
            y = np.array([start, start + n, start + n, start, start])
            ax[1][0].plot(x, y, linewidth=lw, c="k")
            ax[1][1].plot(x, y, linewidth=lw, c="k")

        corr_df = corr_df[(corr_df > threshold) & (corr_df.abs() < 1)]
        corr_df.dropna(axis=0, how="all", inplace=True)
        corr_df.dropna(axis=1, how="all", inplace=True)

        ax[0][1].set_title(f"Correlation map for {method} method", fontsize=24)

        c = 0
        for i, row in corr_df.iterrows():
            for j in corr_df.columns.tolist()[c:]:
                if not np.isnan(row[j]):
                    ax[0][1].plot(
                        self.positions.loc[[i, j]]["y"],
                        self.positions.loc[[i, j]]["x"],
                        color="r",
                        lw=0.5 + (row[j] - threshold) / (1 - threshold) * 4,
                    )

            ax[0][1].scatter(
                x=self.positions.loc[i]["y"],
                y=self.positions.loc[i]["x"],
                color="w",
                zorder=5,
            )
            c += 1

        ax[0][1].scatter(x=self.positions["y"], y=self.positions["x"], s=100, zorder=4)
        ax[0][1].set_ylabel("pixels", fontsize=16)
        ax[0][1].set_xlabel("pixels", fontsize=16)

        plt.show()

    @staticmethod
    def get_corr_clustering(corr_df):
        """
        Function for getting correlation clusters
        :param corr_df: dataframe with correlation values
        :return: 2d array, where 1 - unit id, 2 cluster id
        """
        corr_df = (corr_df + corr_df.T) / 2

        dissimilarity = corr_df_to_distribution(
            corr_df.abs().max().max() - corr_df.abs()
        )
        hierarchy = linkage(dissimilarity, method="average")

        dissimilarity = np.array(dissimilarity)
        thr = 0
        if dissimilarity[dissimilarity < 1].sum() > 0:
            thr = np.quantile(dissimilarity, 0.2)
            if thr == 1:
                thr = dissimilarity[dissimilarity < 1].max()

        clusters = fcluster(hierarchy, thr, criterion="distance")
        clusters = [[col, cl] for col, cl in zip(corr_df.columns, clusters)]
        clusters.sort(key=lambda x: x[1])

        return np.array(clusters)

    @staticmethod
    def get_cluster_stats(corr_df):
        """
        Function for computing stats in correlation clusters
        :param corr_df: dataframe with correlation values
        :return: number of clusters, mean cluster size, mean intercluster distance, mean intracluster distance
        """
        clusters = ActiveStateAnalyzer.get_corr_clustering(corr_df)

        corr_df = corr_df[clusters[:, 0]].loc[clusters[:, 0]]
        lens = pd.DataFrame(clusters).groupby(1).agg(len)[0]

        intercluster_dist = []
        for i, n in enumerate(lens[:-1]):
            start_y = lens[:i].sum()
            for j, m in enumerate(lens[i + 1 :], start=i + 1):
                start_x = lens[:j].sum()
                intercluster_dist.append(
                    corr_df.values[start_y : start_y + n, start_x : start_x + m].mean()
                )

        intracluster_dist = []
        for i, n in enumerate(lens):
            start = lens[:i].sum()
            corr_distr = corr_df_to_distribution(
                corr_df.iloc[start : start + n].T.iloc[start : start + n]
            )
            intracluster_dist.append(np.mean(corr_distr) if len(corr_distr) > 0 else 1)

        clusters_num = len(lens)
        mean_cluster_size = lens.mean()
        mean_intercluster_dist = (
            np.mean(intercluster_dist) if len(intercluster_dist) > 0 else 0
        )
        mean_intracluster_dist = (
            np.mean(intracluster_dist) if len(intracluster_dist) > 0 else 1
        )

        return (
            clusters_num,
            mean_cluster_size,
            mean_intercluster_dist,
            mean_intracluster_dist,
        )

    def get_network_degree(self, method="signal", thrs=None):
        """
        Function for computing network degree
        Network degree - share of strong network connections
        :param method: method of correlation
        :param thrs: list of thresholds for strong correlation between neurons
        :return: network degree
        """
        if thrs is None:
            thrs = np.arange(0, 1, 0.05)

        corr = np.array(corr_df_to_distribution(self.get_correlation(method=method)))
        nd_values = [(corr > thr).sum() / len(corr) for thr in thrs]

        return pd.DataFrame({f"nd_{method}": nd_values, "threshold": thrs})

    def show_network_degree(self, method="signal", thr=None):
        """
        Function for plotting network_degree
        :param method: method of correlation
        :param thrs: list of thresholds for strong correlation between neurons
        """
        nd_df = self.get_network_degree(method, thr)
        plt.figure(figsize=(8, 6))
        plt.title("Network degree", fontsize=17)
        sns.lineplot(data=nd_df, x="threshold", y=f"nd_{method}")
        plt.xlabel("threshold", fontsize=16)
        plt.ylabel("number of co-active neurons, normalised", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def save_network_degree(self, method="signal", thr=None):
        """
        Function for saving network_degree
        :param method: method of correlation
        :param thrs: list of thresholds for strong correlation between neurons
        """
        nd_df = self.get_network_degree(method, thr)
        nd_df.to_excel(self.results_folder + f"/network_degree_{method}.xlsx")

    def get_connectivity(self, method="signal", thr=None, position=False):
        """
        Function for computing connectivity
        Connectivity - share of strong connections for each neuron
        :param method: method of correlation
        :param thr: threshold for strong correlation between neurons
        :param position: consideration of spatial position
        :return: connectivity
        """
        if not thr:
            m = {
                "signal": 0.5,
                "diff": 0.3,
                "active": 0.1,
                "active_acc": 0.1,
                "transfer_entropy": 0.1,
            }
            thr = m[method]

        df_conn = pd.DataFrame()
        corr = self.get_correlation(method=method, position=position)
        df_conn[f"connectivity_{method}"] = ((corr > thr).sum() - 1) / len(corr)
        return df_conn

    def show_connectivity(self, method="signal", thr=None):
        """
        Function for plotting connectivity distribution
        :param method: method of correlation
        :param thr: threshold for strong correlation between neurons
        """
        conn_df = self.get_connectivity(method, thr)
        plt.figure(figsize=(8, 6))
        plt.title("Connectivity", fontsize=17)
        sns.histplot(data=conn_df, x=f"connectivity_{method}", bins=8, stat="percent")
        plt.xlabel("percent of correlated neurons", fontsize=16)
        plt.ylabel("percent of neurons", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def save_connectivity(self, method="signal", thr=None):
        """
        Function for saving connectivity
        :param method: method of correlation
        :param thr: threshold for strong correlation between neurons
        """
        conn_df = self.get_connectivity(method, thr)
        conn_df.to_excel(self.results_folder + f"/connectivity_{method}.xlsx")

    def save_results(self):
        """
        Function for saving all ActiveStateAnalyzer results
        """
        self.save_active_states()
        self.save_burst_rate()
        self.save_network_spike_rate(1)
        self.save_network_spike_peak(1)
        self.save_network_spike_duration([5, 10, 20, 30, 50])

        all_corr_methods = ["signal", "diff", "active", "active_acc"]
        all_pos = [False, True]

        for method, pos in itertools.product(all_corr_methods, all_pos):
            self.save_correlation_matrix(
                method=method, position=pos
            )

        for method in all_corr_methods:
            self.save_network_degree(method)
            self.save_connectivity(method=method)
