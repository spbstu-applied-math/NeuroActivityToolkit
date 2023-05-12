import ipywidgets as widgets
from IPython.display import display


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
        right_box = widgets.VBox([threshold, lag])

        display(widgets.HBox([left_box, right_box]))
        display(corr)

    @staticmethod
    def save_correlation(model):
        """
        Function for creating UI for correlation
        :param model: ActiveStateAnalyzer class
        """

        corr_method = widgets.Dropdown(
            options=["signal", "diff", "active", "active_acc", "transfer_entropy"]
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
        first_line = widgets.HBox([corr_method, position])

        display(widgets.VBox([first_line, button]))

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
    """Class with notebook widgets for StatTests"""

    @staticmethod
    def show_correlation_distribution(model):
        """
        Function for creating UI for show_correlation_distribution
        :param model: Statistic&Shuffling class
        """
        method = widgets.Dropdown(
            options=["box", "hist", "kde"],
            value="kde",
            description="method",
            disabled=False,
        )
        position = widgets.Checkbox(
            value=False, description="Position", disabled=False, indent=False
        )

        wid = widgets.interactive_output(
            model.show_correlation_distribution,
            {"method": method, "position": position},
        )

        display(widgets.HBox([method, position]))
        display(wid)

    @staticmethod
    def get_test(model):
        """
        Function for creating UI for get_test
        :param model: StatTests class
        """
        return widgets.interact(
            model.get_test,
            data_type=widgets.Dropdown(
                options=["corr", "stat"],
                value="stat",
                description="data type",
                disabled=False,
            ),
            test_type=widgets.Dropdown(
                options=["norm", "distr"],
                value="distr",
                description="test type",
                disabled=False,
            ),
        )

    @staticmethod
    def show_distribution_of_connectivity(model):
        """
        Function for creating UI for show_distribution_of_connectivity
        :param model: StatTests class
        """
        q = widgets.FloatSlider(
            value=0.9,
            min=0,
            max=1,
            step=0.01,
            description="q",
            continuous_update=False,
            readout=True,
        )
        position = widgets.Checkbox(
            value=False, description="Position", disabled=False, indent=False
        )

        wid = widgets.interactive_output(
            model.show_distribution_of_connectivity, {"q": q, "position": position}
        )

        test = widgets.interact(
            model.get_connectivity_distr_test, q=q, position=position
        )

        display(wid, test)

    @staticmethod
    def show_network_degree(model):
        """
        Function for creating UI for show_network_degree
        :param model: StatTests class
        """
        interval = widgets.FloatRangeSlider(
            value=[0, 1],
            min=0,
            max=1,
            step=0.001,
            description="interval:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".3f",
        )

        step = widgets.FloatSlider(
            value=10e-3,
            min=10e-4,
            max=0.1,
            step=10e-4,
            description="step",
            continuous_update=False,
            readout=True,
            readout_format=".3f",
        )

        position = widgets.Checkbox(
            value=False, description="Position", disabled=False, indent=False
        )

        wid = widgets.interactive_output(
            model.show_network_degree,
            {"interval": interval, "step": step, "position": position},
        )

        test = widgets.interact(
            model.get_nd_test, interval=interval, step=step, position=position
        )

        display(wid, test)


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
    def show_stats_deviation(model):
        """
        Function for creating UI for show_stats_deviation
        :param model: Data class
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
        display(condition)
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
