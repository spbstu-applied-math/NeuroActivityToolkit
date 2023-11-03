import ipywidgets as widgets
from IPython.display import display
import itertools
import numpy as np
from tqdm.notebook import tqdm
from analysis.functions import stat_test


class ActiveStateAnalyzerWidgets:
    """Class with notebook widgets for ActiveStateAnalyzer"""

    @staticmethod
    def find_active_state(model):
        """
        Function for creating UI for find_active_state
        :param model: ActiveStateAnalyzer class
        """
        neuron = widgets.Dropdown(
            options=model.signals.columns,
            description="neuron",
            disabled=False,
        )
        cold = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description="cold",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        warm = widgets.IntSlider(
            value=50,
            min=0,
            max=100,
            step=1,
            description="warm",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        window = widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            description="window",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        method = widgets.Dropdown(
            options=["spike", "full"],
            value="spike",
            description="method",
            disabled=False,
        )

        wid = widgets.interactive_output(
            model.get_active_state,
            {
                "neuron": neuron,
                "window": window,
                "cold": cold,
                "warm": warm,
                "method": method,
            },
        )

        button = widgets.Button(
            description="Set parameters",
            button_style="success",  # 'success', 'info', 'warning', 'danger', ''
        )

        def on_button_clicked(b):
            model.find_active_state(
                window=window.value,
                cold=cold.value,
                warm=warm.value,
                method=method.value,
                verbose=False,
            )

        button.on_click(on_button_clicked)

        left_box = widgets.VBox([neuron, method])
        center_box = widgets.VBox([cold, warm])
        right_box = widgets.VBox([window, button])

        display(widgets.HBox([left_box, center_box, right_box]))
        display(wid)

    @staticmethod
    def burst_rate(model):
        """
        Function for creating UI for burst rate
        :param model: ActiveStateAnalyzer class
        """
        max_bins = widgets.IntSlider(
            value=15,
            min=1,
            max=40,
            step=1,
            description="max_bins",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )

        wid = widgets.interactive_output(model.show_burst_rate, {"max_bins": max_bins})

        button = widgets.Button(description="Save", button_style="success")

        def save_burst_rate(b):
            model.save_burst_rate()

        button.on_click(save_burst_rate)

        center_box = widgets.HBox([max_bins, button])

        display(center_box)
        display(wid)

    @staticmethod
    def network_spike_rate(model):
        """
        Function for creating UI for network spike rate
        :param model: ActiveStateAnalyzer class
        """
        period = widgets.IntSlider(
            value=1,
            min=1,
            max=60,
            step=1,
            description="period (sec)",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )

        wid = widgets.interactive_output(
            model.show_network_spike_rate, {"period": period}
        )

        button = widgets.Button(
            description="Save",
            button_style="success",  # 'success', 'info', 'warning', 'danger', ''
        )

        def save_network_spike_rate(b):
            model.save_network_spike_rate(period=period.value)

        button.on_click(save_network_spike_rate)

        center_box = widgets.HBox([period, button])

        display(center_box)
        display(wid)

    @staticmethod
    def network_spike_peak(model):
        """
        Function for creating UI for network spike peak
        :param model: ActiveStateAnalyzer class
        """
        period = widgets.IntSlider(
            value=1,
            min=1,
            max=60,
            step=1,
            description="period (sec)",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )

        wid = widgets.interactive_output(
            model.show_network_spike_peak, {"period": period}
        )

        button = widgets.Button(description="Save", button_style="success")

        def save_network_spike_peak(b):
            model.save_network_spike_peak(period=period.value)

        button.on_click(save_network_spike_peak)

        center_box = widgets.HBox([period, button])

        display(center_box)
        display(wid)

    @staticmethod
    def network_spike_duration(model, thresholds):
        """
        Function for creating UI for network spike duration
        :param model: ActiveStateAnalyzer class
        :param thresholds: threshold values in percentages
        """
        button = widgets.Button(description="Save", button_style="success")

        def save_network_spike_duration(b):
            model.save_network_spike_duration(thresholds)

        button.on_click(save_network_spike_duration)

        display(button)

        model.show_network_spike_duration(thresholds)

    @staticmethod
    def correlation(model):
        """
        Function for creating UI for correlation
        :param model: ActiveStateAnalyzer class
        """

        corr_method = widgets.Dropdown(
            options=["signal", "diff", "active", "active_acc", "transfer_entropy"]
        )

        threshold = widgets.FloatSlider(
            value=0.8,
            min=0,
            max=1,
            step=0.001,
            description="threshold",
            continuous_update=False,
            readout=True,
            readout_format=".3f",
        )

        position = widgets.Checkbox(
            value=False, description="Position", disabled=False, indent=False
        )

        lag = widgets.IntSlider(
            value=0,
            min=0,
            max=500,
            step=10,
            description="lag",
            continuous_update=False,
            readout=True,
        )

        corr = widgets.interactive_output(
            model.show_corr,
            {
                "method": corr_method,
                "threshold": threshold,
                "position": position,
                "lag": lag,
            },
        )

        left_box = widgets.VBox([corr_method, position])
        center_box = widgets.VBox([threshold, lag])

        display(widgets.HBox([left_box, center_box]))
        display(corr)

    @staticmethod
    def save_correlation(model):
        """
        Function for creating UI for correlation
        :param model: ActiveStateAnalyzer class
        """

        all_methods = ["signal", "diff", "active", "active_acc", "transfer_entropy"]

        corr_method = widgets.Dropdown(
            options=all_methods
        )

        position = widgets.Checkbox(
            value=False, description="Position", disabled=False, indent=False
        )

        button = widgets.Button(description="Save", button_style="success")

        def save_correlation(b):
            print("Saving...")
            model.save_correlation_matrix(
                method=corr_method.value, position=position.value
            )
            print("Done!")

        button.on_click(save_correlation)

        button2 = widgets.Button(description="Save all", button_style="success")

        def save_all_correlation(b):
            all_pos = [False, True]

            for method, pos in tqdm(itertools.product(all_methods, all_pos),
                                    total=len(all_methods)*len(all_pos),
                                    desc="Saving..."
                                    ):
                model.save_correlation_matrix(
                    method=method, position=pos
                )

        button2.on_click(save_all_correlation)

        first_line = widgets.HBox([corr_method, position])

        display(widgets.VBox([first_line, button, button2]))

    @staticmethod
    def network_degree(model):
        """
        Function for creating UI for network degree
        :param model: ActiveStateAnalyzer class
        """
        corr_method_nd = widgets.Dropdown(
            options=["signal", "diff", "active", "active_acc", "transfer_entropy"]
        )

        wid = widgets.interactive_output(
            model.show_network_degree, {"method": corr_method_nd}
        )

        button = widgets.Button(description="Save", button_style="success")

        def save_network_degree(b):
            model.save_network_degree(method=corr_method_nd.value)

        button.on_click(save_network_degree)

        center_box = widgets.HBox([corr_method_nd, button])

        display(center_box)
        display(wid)

    @staticmethod
    def connectivity(model):
        """
        Function for creating UI for connectivity
        :param model: ActiveStateAnalyzer class
        """
        corr_method_conn = widgets.Dropdown(
            options=["signal", "diff", "active", "active_acc", "transfer_entropy"]
        )

        threshold_conn = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.001,
            description="threshold",
            continuous_update=False,
            readout=True,
            readout_format=".3f",
        )
        wid = widgets.interactive_output(
            model.show_connectivity, {"method": corr_method_conn, "thr": threshold_conn}
        )

        button = widgets.Button(description="Save", button_style="success")

        def save_connectivity(b):
            model.save_connectivity(
                method=corr_method_conn.value, thr=threshold_conn.value
            )

        button.on_click(save_connectivity)

        center_box = widgets.HBox([corr_method_conn, threshold_conn, button])

        display(center_box)
        display(wid)


class StatisticWidgets:
    """Class with notebook widgets for statistical test"""

    tests = [
        "t-test_ind",
        "t-test_welch",
        "t-test_paired",
        "Mann-Whitney",
        "Mann-Whitney-gt",
        "Mann-Whitney-ls",
        "Levene",
        "Wilcoxon",
        "Kruskal"
    ]

    plot_format = ["bar", "box"]

    text_format = ["star", "simple", "full"]

    @staticmethod
    def get_stat_test(data, features, hue):
        """
        Function for creating UI for stat testing
        :param data: DataFrame with data
        :param features: list of features
        :param hue: group column
        """
        test = widgets.Dropdown(
            options=StatisticWidgets.tests,
            value="Kruskal",
            description="test",
            disabled=False,
        )

        text_format = widgets.Dropdown(
            options=StatisticWidgets.text_format,
            value="star",
            description="text format",
            disabled=False,
        )

        plot_format = widgets.Dropdown(
            options=StatisticWidgets.plot_format,
            value="bar",
            description="plot format",
            disabled=False,
        )

        feature = widgets.Dropdown(
            options=features,
            value=features[0],
            description="feature",
            disabled=False,
        )

        wid = widgets.interactive_output(
            stat_test,
            {"data": widgets.fixed(data),
             "x": widgets.fixed(hue),
             "y": feature,
             "test": test,
             "text_format": text_format,
             "kind": plot_format},
        )

        col1 = widgets.VBox([feature, test])
        col2 = widgets.VBox([text_format, plot_format])
        display(widgets.HBox([col1, col2]))
        display(wid)

    @staticmethod
    def show_features(model):
        """
        Function for creating UI for feature testing
        :param model: Data class
        """
        features = model.data.columns.tolist()

        data = model.data
        data["condition"] = [
            model.params[session]["condition"] for session in data.index
        ]

        StatisticWidgets.get_stat_test(data, features, "condition")

    @staticmethod
    def show_shuffling(model):
        """
        Function for creating UI for shuffling testing
        :param model: MultipleShuffler class
        """
        agg_stat = (
            model.stat_df.groupby(["date", "shuffle_fraction", "attempt"])
            .agg(["max", "mean", "std"])
            .reset_index().
            groupby(["date", "shuffle_fraction"])
            .agg("mean")
            .drop(columns=["attempt"])
        )
        agg_stat.columns = agg_stat.columns.map(' '.join)
        agg_stat = agg_stat.reset_index()
        agg_stat["shuffle_fraction"] = agg_stat["shuffle_fraction"].astype(str)

        agg_corr = (
            model.corr_df.groupby(["position", "date", "shuffle_fraction", "attempt"])
            .agg(["mean", "std", np.ptp])
            .reset_index()
            .groupby(["position", "date", "shuffle_fraction"])
            .agg("mean")
            .drop(columns=["attempt"])
        )

        agg_corr.columns = agg_corr.columns.map(' '.join)
        agg_corr = agg_corr.reset_index()

        agg_corr["shuffle_fraction"] = agg_corr["shuffle_fraction"].astype(str)
        agg_corr = agg_corr[~agg_corr['position']].drop(columns=["position"])

        data = agg_stat.merge(agg_corr, on=['date', 'shuffle_fraction'])

        features = data.drop(columns=['date', 'shuffle_fraction']).columns.tolist()
        StatisticWidgets.get_stat_test(data, features, "shuffle_fraction")

    @staticmethod
    def show_distance(model):
        """
        Function for creating UI for distance testing
        :param model: DistanceAnalysis class
        """
        data = model.distance_df

        data = data.rename(columns={'signal': 'signal correlation',
                                    'active': 'active correlation',
                                    'active_acc': 'active_acc correlation'})

        features = data.drop(columns=['date']).columns.tolist()

        StatisticWidgets.get_stat_test(data, features, 'date')


class ShufflingWidgets:
    """Class with notebook widgets for ShuffleAnalysis"""

    @staticmethod
    def show_shuffling(model):
        """
        Function for creating UI for show_shuffling
        :param model: ShuffleAnalysis class
        """
        return widgets.interact(
            model.show_shuffling,
            date=widgets.Dropdown(
                options=model.dates.keys(),
                description="Date",
                disabled=False,
            ),
        )

    @staticmethod
    def correlation_ptp(model):
        """
        Function for creating UI for correlation_ptp
        :param model: ShuffleAnalysis class
        """
        return widgets.interact(
            model.correlation_ptp,
            corr_type=widgets.Dropdown(
                options=["active", "active_acc"],
                value="active",
                description="Correlation type",
                disabled=False,
            ),
        )

    @staticmethod
    def statistic_info(model):
        """
        Function for creating UI for statistic_info
        :param model: ShuffleAnalysis class
        """
        widgets.interact(
            model.statistic_info,
            stat_type=widgets.Dropdown(
                options=["network_spike_rate", "network_spike_peak"],
                value="network_spike_rate",
                description="Statistic type",
                disabled=False,
            ),
        )


class MultipleShufflingWidgets:
    """Class with notebook widgets for MultipleShuffler"""

    @staticmethod
    def show_day_mean_correlation_range(model):
        """
        Function for creating UI for show_day_mean_correlation_range
        :param model: MultipleShuffler class
        """
        position = widgets.Checkbox(
            value=False, description="Position", disabled=False, indent=False
        )

        wid = widgets.interactive_output(
            model.show_day_mean_correlation_range, {"position": position}
        )

        display(widgets.HBox([position]))
        display(wid)

    @staticmethod
    def show_mean_statistic_peak(model):
        """
        Function for creating UI for show_mean_statistic_peak
        :param model: MultipleShuffler class
        """
        statistic_type = widgets.Dropdown(
            options=["network spike peak", "network spike rate"],
            value="network spike peak",
            description="Statistic type",
            disabled=False,
        )

        daily = widgets.Checkbox(
            value=False, description="Daily", disabled=False, indent=False
        )

        wid = widgets.interactive_output(
            model.show_mean_statistic_peak,
            {"daily": daily, "statistic_type": statistic_type},
        )

        display(widgets.HBox([statistic_type, daily]))
        display(wid)

    @staticmethod
    def save(model, path):
        """
        Function for creating UI for saving
        :param model: MultipleShuffler class
        :param path: path to target folder
        """
        button = widgets.Button(description="Save", button_style="success")

        def on_button_clicked(b):
            print("Saving...")
            model.save_results(path)
            print("Done!")

        button.on_click(on_button_clicked)
        display(button)


class DataWidgets:
    """Class with notebook widgets for Data"""

    @staticmethod
    def show_result(model, conditions_order):
        """
        Function for creating UI for show_map
        :param model: Data class
        :param conditions_order: conditions order
        """
        mouse = widgets.Dropdown(
            options=conditions_order.keys(),
            value=list(conditions_order.keys())[0],
            description="mouse",
            disabled=False,
        )

        def show_map(mouse_id):
            model.show_result(mouse_id, conditions_order[mouse_id])

        mouse_map = widgets.interactive_output(show_map, {"mouse_id": mouse})
        display(mouse)
        display(mouse_map)

    @staticmethod
    def show_stat(model, conditions_order):
        """
        Function for creating UI for show_stats
        :param model: Data class
        :param conditions_order: conditions order
        """
        stat = widgets.Dropdown(
            options=model.get_stat_list(),
            value=model.get_stat_list()[0],
            description="stat",
            disabled=False,
        )

        condition = widgets.Dropdown(
            options=model.data_reduced["condition"].unique().tolist() + ["all"],
            value="all",
            description="condition",
            disabled=False,
        )

        def show_stats(stat, condition):
            model.show_stat(
                stat=stat, condition=condition, conditions_order=conditions_order
            )

        stats_deviation = widgets.interactive_output(
            show_stats, {"stat": stat, "condition": condition}
        )

        display(stat)
        display(condition)
        display(stats_deviation)

    @staticmethod
    def stats_deviation(model, path):
        """
        Function for creating UI for show_stats_deviation
        :param model: Data class
        :param path: path to target folder
        """

        condition = widgets.Dropdown(
            options=model.data_reduced["condition"].unique().tolist() + ["all"],
            value="all",
            description="condition",
            disabled=False,
        )

        stats_deviation = widgets.interactive_output(
            model.show_stats_deviation, {"condition": condition}
        )

        button = widgets.Button(description="Save", button_style="success")

        def save_stats_deviation(b):
            print("Saving...")
            model.save_stats_deviation(path=path)
            print("Done!")

        button.on_click(save_stats_deviation)

        center_box = widgets.HBox([condition, button])

        display(center_box)
        display(stats_deviation)

    @staticmethod
    def save(model, path):
        """
        Function for creating UI for saving
        :param model: MultipleShuffler class
        :param path: path to target folder
        """
        button = widgets.Button(description="Save", button_style="success")

        def on_button_clicked(b):
            print("Saving...")
            model.save_results(path)
            print("Done!")

        button.on_click(on_button_clicked)
        display(button)


class DistanceAnalysisWidgets:
    """Class with notebook widgets for DistanceAnalysis"""

    @staticmethod
    def plot_dependency(model):
        """
        Function for creating UI for plot_dependency
        :param model: DistanceAnalysis class
        """
        x = widgets.Dropdown(
            options=["euclidean", "radial", "signal", "active", "active_acc"]
        )
        y = widgets.Dropdown(
            options=["euclidean", "radial", "signal", "active", "active_acc"],
            value="active",
        )

        out = widgets.interactive_output(
            model.plot_dependency,
            {
                "x": x,
                "y": y,
            },
        )

        display(widgets.HBox([x, y]))
        display(out)

    @staticmethod
    def plot_distance_distribution(model):
        """
        Function for creating UI for plot_distance_distribution
        :param model: DistanceAnalysis class
        """
        distance = widgets.Dropdown(options=["euclidean", "radial"])

        threshold = widgets.FloatSlider(
            value=-1,
            min=-1,
            max=1,
            step=0.001,
            description="threshold",
            continuous_update=False,
            readout=True,
            readout_format=".3f",
        )

        out = widgets.interactive_output(
            model.plot_distance_distribution, {"dist_type": distance, "thr": threshold}
        )

        display(widgets.HBox([distance, threshold]))
        display(out)

    @staticmethod
    def save(model, path):
        """
        Function for creating UI for saving
        :param model: DistanceAnalysis class
        :param path: path to target folder
        """
        button = widgets.Button(description="Save", button_style="success")

        def on_button_clicked(b):
            print("Saving...")
            model.save_results(path)
            print("Done!")

        button.on_click(on_button_clicked)
        display(button)


class GraphWidgets:
    """Class with notebook widgets for GraphAnalysis"""

    @staticmethod
    def show_clusters(model):
        """
        Function for creating UI for show_clusters
        :param model: GraphAnalysis class
        """
        corr_method = widgets.Dropdown(
                    options=model.corr_types
                )
        
        session = widgets.Dropdown(
                    options=model.sessions
                )

        resolution = widgets.FloatSlider(
            value=1,
            min=0,
            max=5,
            step=0.05,
            description="cluster resolution",
            continuous_update=False,
            readout=True,
            readout_format=".2f",
        )

        corr = widgets.interactive_output(
            model.show_clusters,
            {
                "session": session,
                "method": corr_method,
                "resolution": resolution,
            },
        )

        display(widgets.HBox([session, corr_method, resolution]))
        display(corr)

    @staticmethod
    def show_stats(model, conditions_order=None):
        """
        Function for creating UI for show_stat
        :param model: GraphAnalysis class
        :param conditions_order: conditions order
        """
        stat = widgets.Dropdown(
            options=["centrality", "global_efficiency", "local_efficiency", "z_score", "participation"],
            value="z_score",
            description="stat",
            disabled=False,
        )

        resolution = widgets.FloatSlider(
            value=1,
            min=0,
            max=5,
            step=0.05,
            description="cluster resolution",
            continuous_update=False,
            readout=True,
            readout_format=".2f",
        )

        def show_stats(stat, resolution):
            model.show_stat(
                stat=stat, resolution=resolution, conditions_order=conditions_order
            )

        stats = widgets.interactive_output(
            show_stats,
            {
                "stat": stat,
                "resolution": resolution,
            },
        )

        display(widgets.HBox([stat, resolution]))
        display(stats)

    @staticmethod
    def save(model, path):
        """
        Function for creating UI for saving
        :param model: GraphAnalysis class
        :param path: path to target folder
        """

        resolution = widgets.FloatSlider(
            value=1,
            min=0,
            max=5,
            step=0.05,
            description="cluster resolution",
            continuous_update=False,
            readout=True,
            readout_format=".2f",
        )
        button = widgets.Button(description="Save", button_style="success")

        def on_button_clicked(b):
            print("Saving...")
            model.save_results(path, resolution.value)
            print("Done!")

        button.on_click(on_button_clicked)
        display(widgets.HBox([resolution]))
        display(button)


def save_df_button(df, path):
    """
    Function for creating save button pd.DataFrame to excel file
    :param df: DataFrame
    :param path: path to save
    :return:
    """
    button = widgets.Button(description="Save", button_style="success")

    def on_button_clicked(b):
        print("Saving...")
        df.to_excel(path)
        print("Done!")

    button.on_click(on_button_clicked)
    display(button)


def save_button(on_button_func):
    """
    Function for creating save button using function
    :param on_button_func: external on button function
    """
    button = widgets.Button(description="Save", button_style="success")

    button.on_click(on_button_func)
    display(button)
