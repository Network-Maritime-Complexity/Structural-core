#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def graph_topo_measures():
    def _average(node_dic):
        if len(node_dic) < 1:
            return 0
        else:
            node_dic = dict(node_dic)
            return round(np.average(list(node_dic.values())), 3)

    graph_features = {}
    node_num = G.number_of_nodes()
    graph_features['# Nodes'] = node_num
    graph_features['# Edges'] = G.number_of_edges()
    graph_features['<K>'] = _average(G.degree())
    # ClusterCoefficient
    graph_features['<C>'] = round(nx.average_clustering(G), 3)
    graph_features['<L>'] = round(nx.average_shortest_path_length(G), 3)

    # assortativity
    '''
    The assortativity coefficient is a correlation coefficient between the
    degrees of all Nodes on two opposite ends of a link. A positive
    assortativity coefficient indicates that Nodes tend to link to other
    Nodes with the same or similar degree.

    Reference: M. E. J. Newman, Mixing patterns in Networks, Physical Review E, 67 026126, 2003
    '''
    assortativity = nx.degree_assortativity_coefficient(G)
    graph_features['Assortativity'] = round(assortativity, 3)

    graph_info = pd.Series(graph_features)
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Basic_topological_properties_and_economic_small_world_ness')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Fig. 2 Basic topological properties of the GLSN (d).csv'
        graph_info.to_csv(save_path + '/' + filename, index=True, header=False)
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()


def node_topo_measures():
    df_features = pd.DataFrame()
    df_features['K'] = pd.Series(dict(nx.degree(G)))
    df_features['BC'] = pd.Series(dict(nx.betweenness_centrality(G)))
    df_features['CC'] = pd.Series(dict(nx.closeness_centrality(G)))
    df_features['port'] = G.nodes()

    return df_features


def fitting(data, full_data=True, x_min=None, x_max=None, survival=True):
    if full_data:
        x_min = data.min()

    np.seterr(divide='ignore', invalid='ignore')  # https://github.com/jeffalstott/powerlaw/issues/28
    model = powerlaw.Fit(data=data.values, xmin=x_min, xmax=x_max)
    xdata, ydata = model.cdf(survival=survival)

    R, p = model.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    """
    R: The loglikelihood ratio of the two sets of likelihoods. 
    If positive, the first set of likelihoods is more likely 
    (and so the probability distribution that produced them is a better fit to the data).
    If negative, the reverse is true.
    p: The significance of the sign of R. 
    If below a critical value (typically .05) the sign of R is taken to be significant. 
    If above the critical value the sign of R is taken to be due to statistical fluctuations.
    """

    if R < 0:
        use_powerlaw = False
        use_exp = True
    else:
        use_powerlaw = True
        use_exp = False

    res = {}
    if use_powerlaw:
        alpha = model.power_law.alpha
        alpha = round(alpha, 3)

        res = {'xdata': xdata,
               'ydata': ydata,
               'coefficient': alpha,
               'R': R}

    elif use_exp:
        lam = model.exponential.parameter1

        res = {'xdata': xdata,
               'ydata': ydata,
               'coefficient': lam,
               'R': R}

    return res


def plot_fitting_res(data, dict_res):
    def _cumulative_distribution_function(data, xmin=None, xmax=None, survival=True):
        from numpy import array
        from numpy import sort
        data = array(data)
        if not data.any():
            from numpy import nan
            return array([nan]), array([nan])

        def trim_to_range(data, xmin=None, xmax=None):
            """
            Removes elements of the data that are above xmin or below xmax (if present)
            """
            from numpy import asarray
            data = asarray(data)
            if xmin:
                data = data[data >= xmin]
            if xmax:
                data = data[data <= xmax]
            return data

        data = trim_to_range(data, xmin=xmin, xmax=xmax)

        n = float(len(data))

        data = sort(data)
        all_unique = not (any(data[:-1] == data[1:]))

        if all_unique:
            from numpy import arange
            CDF = arange(n) / n
        else:
            # This clever bit is a way of using searchsorted to rapidly calculate the
            # CDF of data with repeated values comes from Adam Ginsburg's plfit code,
            # specifically https://github.com/keflavich/plfit/commit/453edc36e4eb35f35a34b6c792a6d8c7e848d3b5#plfit/plfit.py
            from numpy import searchsorted, unique
            CDF = searchsorted(data, data, side='right') / n

            unique_data, unique_indices = unique(data, return_index=True)
            data = unique_data
            CDF = CDF[unique_indices]

        if survival:
            CDF = 1 - CDF

        return data, CDF

    data = sorted(data)
    X = data
    x, ccdf = _cumulative_distribution_function(X)

    lam = dict_res['coefficient']
    y = [np.exp(-lam * (x - 1)) for x in X]

    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)

    ax.plot(x, ccdf, 'bo', label=r'ports', ms=6, alpha=0.9)

    if dict_res['R'] > 0:
        plot_pl = True
        plot_exp = False
    else:
        plot_pl = False
        plot_exp = True

    if plot_pl:
        ax.plot(data, y, 'G--', label='power-law fit')

    if plot_exp:
        ax.plot(data, y, 'r--', linewidth=3, label=r'Fit  $e^{-%.3fk}$' % dict_res['coefficient'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlim([0.8, 1000])

    ax.set_xlabel(r'$K$', fontsize=23, labelpad=5)
    ax.set_ylabel(r'$Pr\ (K>k)$', fontsize=23, labelpad=5)

    ax.set_title('a', loc='left', fontsize=20, pad=15, weight='bold')

    plt.legend(frameon=False, fontsize=18)
    pltstyle.axes_style(ax)

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Basic_topological_properties_and_economic_small_world_ness')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Fig. 2 Basic topological properties of the GLSN (a).png'
        plt.savefig(save_path + '/' + filename, transparent=False, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def cal_params():
    def _cal_ccdf(x):
        x = np.sort(x)

        from numpy import searchsorted, unique
        CDF = searchsorted(x, x, side='right') / float(len(x))
        unique_data, unique_indices = unique(x, return_index=True)

        data = unique_data
        CDF = CDF[unique_indices]

        ccdf = 1. - CDF

        return data, ccdf

    empirical_bc = pd.Series(nx.betweenness_centrality(G, normalized=True))
    empirical_bc_ccdf = _cal_ccdf(empirical_bc)
    empirical_cc = pd.Series(nx.closeness_centrality(G))
    empirical_cc_ccdf = _cal_ccdf(empirical_cc)

    df_bc = pd.read_csv('../data/Other data/2015_BC_Random.csv')
    df_cc = pd.read_csv('../data/Other data/2015_CC_Random.csv')
    random_bc = sorted(df_bc.mean(axis=0).values.tolist(), reverse=True)
    random_bc_ccdf = _cal_ccdf(random_bc)
    random_cc = sorted(df_cc.mean(axis=0).values.tolist(), reverse=True)
    random_cc_ccdf = _cal_ccdf(random_cc)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))

    axes[0].plot(random_bc_ccdf[0], random_bc_ccdf[1], color='gray', marker='o', ms=6.5, ls='', alpha=0.9,
                 label='Random')
    axes[0].plot(empirical_bc_ccdf[0], empirical_bc_ccdf[1], color='b', marker='o', ms=6.5, ls='', alpha=0.9, label='ports')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'${}$'.format('BC'), fontsize=23, labelpad=7)
    axes[0].set_ylabel(r'$Pr\ (BC>bc)$', size=23, labelpad=5)
    axes[0].set_title('b', loc='left', fontsize=20, pad=10, weight='bold')
    axes[0].set_xlim(axes[0].get_xlim()[0], 0.1)
    axes[0].xaxis.set_minor_locator(MultipleLocator(0.005))
    axes[0].xaxis.set_major_locator(MultipleLocator(0.025))

    axes[1].plot(empirical_cc_ccdf[0], empirical_cc_ccdf[1], color='b', marker='o', ms=6.5, ls='', alpha=0.9,
                 label='ports')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'${}$'.format('CC'), fontsize=23, labelpad=7)
    axes[1].set_ylabel(r'$Pr\ (CC>cc)$', size=23, labelpad=5)
    axes[1].set_title('c', loc='left', fontsize=20, pad=10, weight='bold')
    axes[1].set_xlim(0.2, 0.61)
    axes[1].xaxis.set_major_locator(MultipleLocator(0.1))
    axes[1].xaxis.set_minor_locator(MultipleLocator(0.02))

    axes[1].plot(random_cc_ccdf[0], random_cc_ccdf[1], color='gray', marker='o', ms=6.5, ls='', alpha=0.9,
                 label='Random')

    for rr in range(2):
        ax = axes[rr]
        ax.legend(frameon=False, fontsize=18)
        pltstyle.axes_style(ax)

    plt.tight_layout()

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Basic_topological_properties_and_economic_small_world_ness')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Fig. 2 Basic topological properties of the GLSN (b) and (c).png'
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')

    # subplot
    def _cal_sub_ccdf(x, alpha, xmin, a):
        n = len(x)
        c1 = sorted(x)
        cdf = np.arange(1, n + 1) / float(n)
        c2 = [1 - i for i in cdf]

        q = sorted(filter(lambda X: X >= xmin, x))
        cf = [a * (np.power(x / xmin, 1. - alpha)) for x in q]
        cf = [x * float(c2[c1.index(q[0])]) for x in cf]

        df_ccdf = pd.DataFrame()
        df_ccdf['c1'] = c1
        df_ccdf['c2'] = c2
        df_ccdf = df_ccdf[df_ccdf['c1'] >= xmin]

        df_ccdf['q'] = q
        df_ccdf['cf'] = cf
        return df_ccdf

    empirical_fit_res = [2.171, 0.0043]

    sub_fig = plt.figure(figsize=(4, 3))
    sub_ax = sub_fig.add_subplot(111)

    df_random_alpha = pd.read_csv('../data/Other data/BC_Random_Fit.csv')
    df_random_alpha = df_random_alpha[df_random_alpha['pval'] > 0.1]
    rand_alpha = df_random_alpha['Alpha'].mean().round(3)
    cal_cols = df_random_alpha['iterations'].values.tolist()

    df_fit_bc = df_bc[df_bc.index.isin(cal_cols)]
    random_bc = sorted(df_fit_bc.mean(axis=0).values.tolist(), reverse=True)

    random_plot_res = _cal_sub_ccdf(random_bc, rand_alpha, 0.0043, 0.4)
    empirical_plot_res = _cal_sub_ccdf(empirical_bc, empirical_fit_res[0], empirical_fit_res[1], 1.6)

    sub_ax.loglog(random_plot_res['c1'].values.tolist(), random_plot_res['c2'].values.tolist(), color='gray',
                  marker='o', ls='', markersize=8, alpha=0.9)
    sub_ax.loglog(random_plot_res['q'], random_plot_res['cf'], 'k--', linewidth=4.5,
                  label=r'Fit  $x^{%.3f}$' % (-1 * (rand_alpha - 1)))
    sub_ax.loglog(empirical_plot_res['c1'].values.tolist(), empirical_plot_res['c2'].values.tolist(),
                  'bo', markersize=8, alpha=0.9)
    sub_ax.loglog(empirical_plot_res['q'], empirical_plot_res['cf'], 'r--', linewidth=4.5,
                  label=r'Fit  $x^{%.3f}$' % (-1*(empirical_fit_res[0]-1)))

    sub_ax.set_ylabel(r'$Pr\ (BC>bc)$', fontsize=24)
    sub_ax.set_xlabel(r'$BC$', fontsize=26)
    pltstyle.axes_style(sub_ax)

    sub_ax.legend(fontsize=12)
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Basic_topological_properties_and_economic_small_world_ness')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Fig. 2 Basic topological properties of the GLSN (b) subplot.png'
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


class EffiCost:
    """
    References
    ----------
    [1] Latora, Vito, and Massimo Marchiori.
        "Efficient behavior of small-world Networks."
        *Physical Review Letters* 87.19 (2001): 198701.
        <http://dx.doi.org/10.1103/PhysRevLett.87.198701>
    [2] Latora, Vito, & Marchiori, Marchiori.
        "Economic small-world behavior in weighted Networks."
        The European Physical Journal Z-Condensed Matter and Complex Systems, (2003) 32, 249-263.
    """
    def __init__(self):
        self.WEIGHT = dis_col

    def global_efficiency(self, graph):
        sum_eff = 0
        n = len(graph)
        denom = n * (n - 1) / 2
        '''
        nx.all_pairs_dijkstra_path_length:
            Compute shortest path lengths between all nodes in a weighted graph.
        '''
        length = dict(nx.all_pairs_dijkstra_path_length(graph, weight=self.WEIGHT))
        if denom != 0:
            for key in length:
                for subkey in length[key].keys():
                    if key < subkey:
                        eff = 1 / length[key][subkey]
                        if eff != 0:
                            sum_eff += eff
            g_eff = sum_eff / denom
        else:
            g_eff = 0

        return g_eff

    def local_efficiency(self, graph, v):

        egoNet = nx.ego_graph(graph, v, center=False, undirected=True)
        GE_ego_real = self.global_efficiency(egoNet)

        return GE_ego_real

    def complete_graph(self, graph, v, dict_dis):

        list_source = []
        list_target = []
        list_dis = []
        list_neighbors = list(graph.neighbors(v))
        for i, node_i in enumerate(list_neighbors[:-1]):
            for j, node_j in enumerate(list_neighbors[i + 1:]):
                list_source.append(node_i)
                list_target.append(node_j)
                edge = str(node_i) + str('--') + str(node_j)
                list_dis.append(dict_dis.get(edge))
        data = pd.DataFrame()
        data['Source'] = list_source
        data['Target'] = list_target
        data[self.WEIGHT] = list_dis
        ego_graph_dis = nx.from_pandas_edgelist(data, 'Source', 'Target',
                                                edge_attr=self.WEIGHT, create_using=nx.Graph())
        ge_ego_ideal = self.global_efficiency(ego_graph_dis)

        return ge_ego_ideal

    def effi_cost(self):
        all_dis = pd.read_csv('../data/Other data/Distance_SR_GC_' + YEAR + '.csv')
        dict_dis = dict(zip(all_dis['Edge'], all_dis[self.WEIGHT]))

        df_edges = Edges.copy()
        df_edges['Edge'] = df_edges['source'].astype(str) + str('--') + df_edges['target'].astype(str)
        df_edges[self.WEIGHT] = df_edges['Edge'].apply(dict_dis.get)
        weighted_G = nx.from_pandas_edgelist(df_edges, 'source', 'target', edge_attr=self.WEIGHT, create_using=nx.Graph())

        cost_all = all_dis[self.WEIGHT].sum() / 2
        cost = round(df_edges[self.WEIGHT].sum() / cost_all * 100, 1)

        GE_dis = self.global_efficiency(weighted_G)
        GE_dis_ideal = sum(1 / all_dis[self.WEIGHT]) / len(all_dis)
        GE = round(GE_dis / GE_dis_ideal * 100, 1)

        portslist = list(nx.nodes(weighted_G))
        list_LE_real = []
        list_LE_ideal = []
        for port in portslist:
            LE_real = self.local_efficiency(weighted_G, port)
            list_LE_real.append(LE_real)
            LE_ideal = self.complete_graph(weighted_G, port, dict_dis)
            list_LE_ideal.append(LE_ideal)
        df_LE = pd.DataFrame()
        df_LE['LE_real'] = list_LE_real
        df_LE['LE_ideal'] = list_LE_ideal
        df_LE = df_LE[df_LE['LE_ideal'] > 0]
        LE = round(sum(df_LE['LE_real'] / df_LE['LE_ideal']) / G.number_of_nodes() * 100, 1)
        print('The in-text result:')
        print()
        print('"We found the GLSN configuration remarkably economic: its global efficiency and local efficiency '
              'respectively reach {:.1f}% and {:.1f}%, but its wiring cost only accounts for {:.1f}%."'.format(GE, LE, cost))
        print()


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Subsection titled "Basic topological properties and economic small-world-ness"')
    print('Section titled "Results"')
    print('*********************************')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()
    graph_topo_measures()
    data = node_topo_measures()
    k = data['K']
    res = fitting(k)
    plot_fitting_res(k, res)
    cal_params()

    instance = EffiCost()
    instance.effi_cost()
