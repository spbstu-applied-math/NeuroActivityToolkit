import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from analysis.dim_reduction import Data, get_order

class GraphAnalysis(Data):
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
        super().__init__(path_to_data, sessions, verbose)
        self.corr_types = ["signal", "diff", "active", "active_acc"]
        self.corr_data = {}
        for corr in self.corr_types:
            self.corr_data[corr] = self._get_corr_data(corr)

        self.graph_stats = {}
        for corr in tqdm(self.corr_types, disable=not verbose):
            self.graph_stats[corr] = {}
            for x in self.corr_data[corr]:
                cl_stat = self.models[x].get_graph_stats(self.corr_data[corr][x])
                cl_stat['centrality'] = np.mean(cl_stat['centrality'])
                cl_stat['correlation_type'] = corr
                cl_stat['session'] = x
                cl_stat['mouse'] = self.params[x]['mouse']
                cl_stat['condition'] = self.params[x]['condition']

                self.graph_stats[corr][x] = cl_stat


    
    def show_clusters(self, session, method="signal", resolution=1):
        """
        Function for plotting clusters map and heatmap
        :param session: session for plotting
        :param method: ['signal', 'diff', 'active', 'active_acc'] type of correlated sequences
            signal - Pearson correlation for signal intensity
            diff   - Pearson correlation for derivative of signal intensity
            active - Pearson correlation for binary segmentation of active states (depends on the chosen method for find_active_state)
            active_acc - ratio of intersection to union of active states (depends on the chosen method for find_active_state)
        :param resolution: resolution parameter for modularity
        """
        corr_df = self.corr_data[method][session]
        model = self.models[session]
        clusters = model.get_corr_clustering(corr_df, resolution=resolution)
        cluster_corr_df = corr_df[clusters[:, 0]].loc[clusters[:, 0]]
        clusters_len = pd.DataFrame(clusters).groupby(1).agg(len)[0]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", ["red", "yellow", "green", "blue"], len(clusters_len)
        )

        palette = {}
        for i in range(len(clusters_len)):
            palette[i] = cmap(i)

        positions = model.positions.join(pd.DataFrame(clusters, columns=['unit_id', 'cluster']).set_index('unit_id'))

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        sns.scatterplot(data=positions, x="y", y="x", s=100, zorder=4, hue='cluster', palette=palette, ax=ax[0])
        ax[0].set_title(f"Cluster map", fontsize=24)
        ax[0].set_ylabel("pixels", fontsize=16)
        ax[0].set_xlabel("pixels", fontsize=16)

        sns.heatmap(cluster_corr_df, vmin=-1, vmax=1, cmap="coolwarm", ax=ax[1])
        ax[1].set_title(f"Correlation heatmap", fontsize=24)

        lw = len(corr_df) * 0.005
        for i, n in enumerate(clusters_len):
            start = clusters_len[:i].sum()
            x = np.array([start, start, start + n, start + n, start])
            y = np.array([start, start + n, start + n, start, start])
            ax[1].plot(x, y, linewidth=lw, c="k")

        plt.show()
        
        stat = self.graph_stats[method][session].copy()
        cl_stat = model.get_cluster_stats(corr_df, resolution=resolution)
        stat['z_score'] = np.mean(cl_stat['z_score'])
        stat['participation'] = np.mean(cl_stat['participation'])
        stat['number_of_clusters'] = cl_stat['number_of_clusters']
        stat['mean_cluster_size'] = cl_stat['mean_cluster_size']

        for x in stat:
            print(f"{x}: {stat[x]}\n")


    def get_all_stats(self, resolution=1):
        """
        Function for compution all graph statistic
        :param resolution: resolution parameter for modularity
        """
        all_stats = []
        for corr in self.corr_types:
            for x in self.corr_data[corr]:
                stat = self.graph_stats[corr][x].copy()
                cl_stat = self.models[x].get_cluster_stats(self.corr_data[corr][x], resolution=resolution)
                stat['z_score'] = np.mean(cl_stat['z_score'])
                stat['participation'] = np.mean(cl_stat['participation'])
                stat['number_of_clusters'] = cl_stat['number_of_clusters']
                stat['mean_cluster_size'] = cl_stat['mean_cluster_size']
                all_stats.append(stat)

        all_stats = pd.DataFrame(all_stats)

        return all_stats
    
    def show_stat(self, stat, resolution=1, conditions_order=None):
        """
        Function for plotting bars with information about graph statistic
        :param stat: statistic for plotting
        :param resolution: resolution parameter for modularity
        :param condition: {'all' or specific condition} condition for plotting
        :param conditions_order: order of conditions in time (used only if condition=='all')
        """
        df = self.get_all_stats(resolution=resolution)

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
        plt.show()

    def save_results(self, path, resolution=1):
        """
        Function for saving all data
        :param path: path to target folder
        :param resolution: resolution parameter for modularity
        """
        self.get_all_stats(resolution=resolution).to_excel(path + "/graph_stats.xlsx")