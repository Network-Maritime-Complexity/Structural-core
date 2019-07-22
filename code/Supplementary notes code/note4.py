#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""

from configure import *

data_path = os.path.join('../data', 'downloaded data files', '1000 equivalent random networks')


def param_density(pp):
    data = Nodes.apply(pd.to_numeric, errors='coerce')
    param = pp[0]
    threshold = pp[1]

    list_comm = [1, 2, 3, 4, 6, 7]
    df_all_density = pd.DataFrame()
    coor_xs = []
    coor_ys = []
    num_scs = {}
    dict_num_ports = {}
    for i in list_comm:

        list_density = []
        list_nodes = []
        list_edges = []
        df_node = data[data['Community'] == i]
        num_sub_comms = len(pd.unique(df_node['sub_Community']))
        nodes = df_node['id'].values.tolist()
        graph = nx.subgraph(G, nodes)

        df_node = df_node.sort_values(param, ascending=True)
        list_value = list(pd.unique((df_node[param])))

        for value in list_value:
            index_param = df_node[param] >= value
            nodelist = df_node['id'][index_param].values.tolist()
            H = graph.subgraph(nodelist)
            list_density.append(round(nx.density(H), 4))
            list_edges.append(H.size())
            list_nodes.append(len(H))
        density = pd.DataFrame()
        density[param] = list_value
        density['num_nodes'] = list_nodes
        density['num_edges'] = list_edges
        density['Density'] = list_density
        density['Community'] = [i] * len(list_value)
        density.sort_values(param, ascending=False, inplace=True)

        ix = (density[param] >= threshold) & (density['Density'] >= 0.8)
        if density[ix].empty:
            coor_x = np.nan
            coor_y = np.nan
            num_sc_nodes = np.nan
        else:
            coor_x = density.loc[ix, param].values.tolist()[-1]
            coor_y = density.loc[ix, 'Density'].values.tolist()[-1]
            num_sc_nodes = density.loc[ix, 'num_nodes'].values.tolist()[-1]
            ix_sc_nodes = df_node[param] >= coor_x
            num_ports = df_node[ix_sc_nodes].groupby('sub_Community', as_index=False)['id'].count()
            ms = pd.DataFrame(pd.Series(np.arange(1, num_sub_comms + 1, 1), name='sub_Community'))
            num_ports = pd.merge(num_ports, ms, on='sub_Community', how='right')
            num_ports.fillna(0, inplace=True)
            num_ports.sort_values('sub_Community', inplace=True)
            dict_num_ports[i] = num_ports['id'].values.tolist()

        coor_xs.append(coor_x)
        coor_ys.append(coor_y)
        num_scs[i] = num_sc_nodes

        df_all_density = pd.concat([df_all_density, density], axis=0)
    if param == 'B_sub':
        from src import histgram_B_sub
        histgram_B_sub.startup(dict_num_ports)
    if param == 'Z_sub':
        from src import histgram_Z_sub
        histgram_Z_sub.startup(dict_num_ports)
    if param == 'P_sub':
        from src import histgram_P_sub
        histgram_P_sub.startup(dict_num_ports)

    return df_all_density, coor_xs, coor_ys, num_scs


def plot_result(data, coor_xs, coor_ys, pp):
    param = pp[0]
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(14, 5))
    data.sort_values(param, ascending=False, inplace=True)
    communities = [1, 2, 3, 4, 6, 7]
    for i, comm in enumerate(communities):
        ax = axes[i]
        df = data[data['Community'] == communities[i]][1:]
        x = df[param]
        y = df['Density']
        ax.plot(x, y, 'bo', markersize=3.5)
        pltstyle.plot_sub_basic(ax, coor_xs[i], coor_ys[i])

        if param == 'B_sub':
            ax.set_xlabel(r'$B$', fontsize=20)
            ax.text(0, 0.1, 'Module {}'.format(comm), fontsize=18)
            ax.set_xticks(np.arange(-1, 4.01, 1))
        elif param == 'Z_sub':
            ax.set_xlabel(r'$Z$', fontsize=20)
            ax.text(-0.5, 0.1, 'Module {}'.format(comm), fontsize=18)
            ax.set_xticks(np.arange(-1, 3.01, 1))
        else:
            ax.set_xlabel(r'$P$', fontsize=20)
            ax.text(0.1, 0.1, 'Module {}'.format(comm), fontsize=18)
            ax.set_xticks(np.arange(0, 0.81, 0.2))
    if param == 'B_sub':
        axes[0].set_ylabel(r'Density among ports of $B_{i}≥B$', fontsize=18)
    elif param == 'Z_sub':
        axes[0].set_ylabel(r'Density among ports of $Z_{i}≥Z$', fontsize=18)
    else:
        axes[0].set_ylabel(r'Density among ports of $P_{i}≥P$', fontsize=18)
        axes[0].set_xticks(np.arange(0, 0.81, 0.2))

    axes[0].set_yticks(np.arange(0, 1.05, 0.2))
    fig.suptitle('(a)', fontsize=20, style='italic', x=0.1, y=0.95)

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 4')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if param == 'B_sub':
            filename = 'Supplementary Fig. 4 Results for the structural-core...submodular gateway hubs (a).png'
        elif param == 'Z_sub':
            filename = 'Supplementary Fig. 5 Results for the structural-core...submodular provincial hubs (a).png'
        else:
            filename = 'Supplementary Fig. 6 Results for the structural-core...submodular connector hubs (a).png'
        plt.savefig(save_path + '/' + filename)
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()

    else:
        plt.show()
    plt.close('all')


def calculate_density(param, num_scs):
    list_mean = []
    list_max = []
    list_min = []
    list_std = []
    empirical_den = {}
    for module, num_core in num_scs.items():
        df_nodes = Nodes[Nodes['Community'] == module]
        sc_node = df_nodes.sort_values(param, ascending=False)[:num_core]['id'].values
        H = nx.subgraph(G, sc_node)
        density = round(nx.density(H), 4)
        empirical_den[module] = density

        list_density = []
        for rt in range(1, iters + 1):
            df_nodes = Nodes[Nodes['Community'] == module]
            edges = pd.read_csv(data_path + '/' + str(rt) + '.csv', header=None)
            edges.columns = ['source', 'target']
            g = nx.from_pandas_edgelist(edges, 'source', 'target', create_using=nx.Graph())
            sc_node = df_nodes.sort_values(param, ascending=False)[:num_core]['id'].values
            H = g.subgraph(sc_node)
            density = round(nx.density(H), 4)
            list_density.append(density)
        density = pd.Series(list_density, name='Density')

        den_max = density.max()
        den_mean = density.mean()
        den_min = density.min()
        den_std = density.std()
        list_mean.append(den_mean)
        list_max.append(den_max - den_mean)
        list_min.append(den_mean - den_min)
        list_std.append(den_std)
    df_plot = pd.DataFrame()
    df_plot['max'] = list_max
    df_plot['mean'] = list_mean
    df_plot['min'] = list_min
    df_plot['std'] = list_std

    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111)

    x = [1, 2, 3, 4, 5, 6]
    ax.plot(x, empirical_den.values(), 'rX', ms=12)

    ax.errorbar(x, df_plot['mean'], yerr=df_plot['std'], fmt='bo', ms=7, lw=3.5, capsize=6)
    ax.errorbar(x, df_plot['mean'], [df_plot['min'], df_plot['max']], fmt='bo', ms=7, lw=1.5, mfc='k', mec='k',
                capsize=6)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0.5, 6.5])
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xticklabels([0, 1, 2, 3, 4, 6, 7])
    ax.set_xlabel(r"Modules", weight='roman', fontsize=25)
    ax.set_ylabel(r"Connectivity density", weight='roman', fontsize=25)

    ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=2, length=9,
                   pad=5)
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor', width=1.7, length=6)

    t = plt.title('Note: The number of iterations of the experiment:' + '\n' +
                  'in your test, {}; in the manuscript, 1000.'.format(iters), color='red', style='italic',
                  fontsize=16, pad=25)
    t.set_bbox(dict(facecolor='gray', alpha=0.3, edgecolor=None))

    plt.xticks(fontproperties='Arial', fontsize=22, weight='bold')
    plt.yticks(fontproperties='Arial', fontsize=22, weight='bold')

    ax.spines['bottom'].set_linewidth(2.3)
    ax.spines['left'].set_linewidth(2.3)
    ax.spines['top'].set_linewidth(2.3)
    ax.spines['right'].set_linewidth(2.3)

    plt.tight_layout()

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 4')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Supplementary Fig. 7 Comparison with permutated networks for the connectivity ' \
                   'density...modules.png'
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def compare(shortest_pathes, central_edges, central_edges_re, central_ports):
    add_N = 0
    add_n = 0
    add_edges = 0

    for path in shortest_pathes:
        add_N += 1
        union_nodes = central_ports.intersection(path)
        num_union = len(union_nodes)
        if num_union < 1:
            pass
        elif num_union < 2:
            add_n += 1
        else:
            add_n += 1
            union_indexes = [path.index(each) for each in union_nodes]
            union_indexes = sorted(union_indexes)
            for i in range(len(union_indexes) - 1):
                index_diff = union_indexes[i + 1] - union_indexes[i]
                if index_diff < 2:
                    edge = (path[union_indexes[i]], path[union_indexes[i + 1]])
                    if edge in central_edges or edge in central_edges_re:
                        add_edges += 1
                        break
                    else:
                        pass
                else:
                    pass
    return add_N, add_n, add_edges


def sc_topological_centrality(param, dict_module):
    n = 0
    N = 0
    edge = 0
    list_pr = []
    list_pr_edge = []
    for comm, num_core in dict_module.items():
        df_nodes = Nodes[Nodes['Community'] == comm]
        nodelist = df_nodes['id'].values.tolist()
        g = nx.subgraph(G, nodelist)
        df_edges = nx.to_pandas_edgelist(g)
        # find gateway hub structural core: top core_num nodes
        list_core_ports = df_nodes.sort_values([param], ascending=False)['id'][:num_core].values
        set_central_ports = set(list_core_ports)
        # non-core-port
        list_non_core_port = list(set(df_nodes['id']).difference(set_central_ports))

        core_edges = df_edges[(df_edges['source'].isin(list_core_ports)) & (df_edges['target'].isin(list_core_ports))]
        list_source = core_edges['source'].values.tolist()
        list_target = core_edges['target'].values.tolist()
        tuple_edges = list((zip(list_source, list_target)))  # core edges
        tuple_edges_re = list((zip(list_target, list_source)))  # source-target reversed core edges

        # non-core edges compare with core edges
        for port_s in list_non_core_port:
            for port_t in list_non_core_port:
                if port_s < port_t:
                    path = nx.all_shortest_paths(g, source=port_s, target=port_t)
                    result = compare(path, tuple_edges, tuple_edges_re, set_central_ports)
                    N += result[0]
                    n += result[1]
                    edge += result[2]

        pr = round(n / N * 100, 2)
        pr_edge = round(edge / N * 100, 2)

        list_pr.append(pr)
        list_pr_edge.append(pr_edge)

        N = 0
        n = 0
        edge = 0

    df_res = pd.DataFrame()
    df_res['Module'] = [1, 2, 3, 4, 6, 7]
    df_res['by node (%)'] = list_pr
    df_res['by link (%)'] = list_pr_edge

    return df_res


def random_test(dict_module):
    n = 0
    N = 0
    edge = 0
    dict_discribe_node = {}
    dict_discribe_link = {}
    for comm, num_core in dict_module.items():
        df_nodes = Nodes[Nodes['Community'] == comm]
        nodelist = df_nodes['id'].values.tolist()
        g = nx.subgraph(G, nodelist)
        df_edges = nx.to_pandas_edgelist(g)
        list_id = list(set(df_nodes['id']))

        list_pr = []
        list_pr_edge = []
        for i in range(iters):
            list_core_ports = random.sample(list_id, num_core)
            set_central_ports = set(list_core_ports)
            list_none_central_port = list(set(df_nodes['id']).difference(set_central_ports))

            core_edges = df_edges[
                (df_edges['source'].isin(list_core_ports)) & (df_edges['target'].isin(list_core_ports))]
            list_source = core_edges['source'].values.tolist()
            list_target = core_edges['target'].values.tolist()
            tuple_edges = list((zip(list_source, list_target)))
            tuple_edges_re = list((zip(list_target, list_source)))

            for port_s in list_none_central_port:
                for port_t in list_none_central_port:
                    if port_s < port_t:
                        path = nx.all_shortest_paths(g, source=port_s, target=port_t)
                        result = compare(path, tuple_edges, tuple_edges_re, set_central_ports)
                        N += result[0]
                        n += result[1]
                        edge += result[2]
            pr = n / N * 100
            list_pr.append(pr)

            pr_edge = edge / N * 100
            list_pr_edge.append(pr_edge)

            N = 0
            n = 0
            edge = 0
        series_pr = pd.Series(list_pr, name='Pr_node')
        series_pr_edge = pd.Series(list_pr_edge, name='Pr_edge')
        ave_pr = series_pr.mean()
        std_pr = series_pr.std()
        max_pr = series_pr.max()
        min_pr = series_pr.min()
        ave_pr_edge = series_pr_edge.mean()
        std_pr_edge = series_pr_edge.std()
        max_pr_edge = series_pr_edge.max()
        min_pr_edge = series_pr_edge.min()
        dict_discribe_node[comm] = [ave_pr, std_pr, max_pr, min_pr]
        dict_discribe_link[comm] = [ave_pr_edge, std_pr_edge, max_pr_edge, min_pr_edge]

    df_node = pd.DataFrame(dict_discribe_node).T
    df_link = pd.DataFrame(dict_discribe_link).T
    df_node.columns = ['by node (Mean (%))', 'by node (SD (%))', 'by node (Max (%))', 'by node (Min (%))']
    df_link.columns = ['by link (Mean (%)', 'by link (SD (%))', 'by link (Max (%))', 'by link (Min (%))']
    df_rea_all = pd.concat([df_node, df_link], axis=1)

    return df_rea_all


def startup():
    if os.path.exists(data_path):
        print('*********************************')
        print("Location in the manuscript text: ")
        print('Section titled "Supplementary note 4: Gateway-hub-based structural core '
              'organization of the GLSN at modular level"')
        print('*********************************')
        print()
        print('***************************RUN TIME WARNING***************************')
        print('It needs 4 days for 1000 iterations of the corresponding experiments.')
        print()
        print('---------------------------------------------------------------------------------------------------')
        print('Output:')
        print()
        print('**********************************************************************************************')
        print('Note: The number of iterations of the experiment: in your test, {}; in '
              'the manuscript, 1000.'.format(iters))
        print('**********************************************************************************************')
        print()
        params = [['Z_sub', 1.5], ['P_sub', 0.7], ['B_sub', 1.5]]
        for pp in params:
            df_all_density, coor_xs, coor_ys, num_scs = param_density(pp)
            plot_result(df_all_density, coor_xs, coor_ys, pp)

            if pp[0] == 'B_sub':
                calculate_density(pp[0], num_scs)
                df_emp = sc_topological_centrality(pp[0], num_scs)
                df_rand = random_test(num_scs)
                df_emp.index = range(0, len(df_emp))
                df_rand.index = range(0, len(df_rand))
                df_res = pd.concat([df_emp, df_rand], axis=1,
                                   keys=['Structure-core ports', '{} random port sets'.format(iters)])
                if SAVE_RESULT:
                    save_path = os.path.join('output', 'Supplementary note 4')
                    if os.path.exists(save_path):
                        pass
                    else:
                        os.makedirs(save_path)
                    filename = 'Supplementary table 2 Summary of data...topological centrality of structural-' \
                               'core...modules.csv'
                    df_res.to_csv(save_path + '/' + filename, header=True, index=False)
                    print()
                    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
                    print()
    else:
        print()
        print('Please download *downloaded data files.zip* file first!')
        sys.exit()
