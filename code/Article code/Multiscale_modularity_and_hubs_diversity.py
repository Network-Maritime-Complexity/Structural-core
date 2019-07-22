#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def edge_plot():
    dict_q = {}
    dict_community = dict(zip(Nodes['id'], Nodes['Community']))
    q = round(community.modularity(dict_community, G), 3)
    dict_q['GLSN'] = q

    dict_num_nodes = {}
    num_nodes = Nodes.groupby('Community')['id'].count().values.tolist()
    dict_num_nodes['GLSN'] = num_nodes

    communities = sorted(pd.unique(Nodes['Community']))
    for module in communities:

        if module == 5:
            dict_q['Module 5'] = np.nan
            dict_num_nodes['Module 5'] = np.nan
        else:
            data = Nodes[Nodes['Community'] == module]
            sub_nodes = data['id']
            dict_comm = dict(zip(Nodes['id'], Nodes['sub_Community']))
            sub_g = nx.subgraph(G, sub_nodes)
            q = round(community.modularity(dict_comm, sub_g), 3)
            dict_q['Module {}'.format(module)] = q
            nnodes = data.groupby('sub_Community')['id'].count().values.tolist()
            dict_num_nodes['Module {}'.format(module)] = nnodes

    df_res = pd.DataFrame()
    df_res['Q'] = pd.Series(dict_q)
    df_res['the size of each module'] = pd.Series(dict_num_nodes)

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Multiscale_modularity_and_hubs_diversity')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Fig. 3 Multiscale modular communities in the GLSN (a) Left.xlsx'
        df_res.to_excel(save_path + '/' + filename, index=True, na_rep='--')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()

    degree = pd.Series(dict(nx.degree(G)))
    Nodes['K'] = Nodes['id'].apply(degree.get)
    Nodes.sort_values(['Community', 'K'], ascending=[True, False], inplace=True)

    nodelist = Nodes['id'].values
    am = nx.to_numpy_matrix(G, nodelist=nodelist)

    am_data = pd.DataFrame(data=am, index=nodelist, columns=nodelist)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    ax.imshow(am_data, cmap=plt.cm.Greys)

    ticks = Nodes.groupby('Community')['id'].count()
    ticks = ticks.cumsum().values.tolist()
    ticks.insert(0, 0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(color='gray', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Module', fontsize=26)
    ax.set_ylabel('Module', fontsize=26)

    pltstyle.axes_style(ax)
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7'], fontsize=14)
    ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7'], fontsize=14)

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Multiscale_modularity_and_hubs_diversity')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Fig. 3 Multiscale modular communities in the GLSN (b) Right.png'
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')

    Edges['source_comm'] = Edges['source'].apply(dict_community.get)
    Edges['target_comm'] = Edges['target'].apply(dict_community.get)
    Edges['Edge'] = Edges['source'].astype(str) + '--' + Edges['target'].astype(str)
    df_dis = pd.read_csv('../data/other data/Distance_SR_GC_2015.csv')
    dict_dis = dict(zip(df_dis['Edge'], df_dis[dis_col]))
    Edges[dis_col] = Edges['Edge'].apply(dict_dis.get)
    lone_edges = Edges[Edges[dis_col] >= 10000]
    inter_edge = lone_edges[lone_edges['source_comm'] != lone_edges['target_comm']]
    dis_perc = len(inter_edge) / len(lone_edges) * 100
    print('The in-text result:')
    print()
    print('"When investigating the geographical distance long-range inter-port shipping connections are found to be '
          'few, and amongst them the majority are inter-community connections, for instance, {:.1f} percent of '
          'inter-port connections longer than 10,000km are inter-community links."'.format(dis_perc))
    print()


def cal_zscore(data):
    means = np.mean(data)
    std = np.std(data, ddof=1)
    data = (data - means) / std

    return data


def cal_bzp(Edges, Nodes, community_col, degree_col):
    nodes_copy = Nodes.apply(pd.to_numeric, errors='coerce')
    if nodes_copy[community_col].isna().any():
        nas = [np.nan] * len(nodes_copy)
        nodes_copy['B'] = nas
        nodes_copy['Z'] = nas
        nodes_copy['P'] = nas
    else:
        edges_copy = Edges.copy()
        tmp = Edges['source'].copy()
        edges_copy['source'] = edges_copy['target']
        edges_copy['target'] = tmp
        Edges = pd.concat([Edges, edges_copy], axis=0)
        dict_module = dict(zip(nodes_copy['id'], nodes_copy[community_col]))
        dict_degree = dict(zip(nodes_copy['id'], nodes_copy[degree_col]))
        new_cols = ['Community_source', 'Community_target']
        Edges[new_cols] = Edges[['source', 'target']].applymap(dict_module.get)

        port_intra_links = []
        port_extra_links = []
        list_p = []
        for port in nodes_copy['id']:
            port_links = Edges.loc[(Edges['source'] == port)]
            intra_links = port_links[port_links['Community_source'] == port_links['Community_target']]
            port_intra_links.append(len(intra_links))
            extra_links = port_links[port_links['Community_source'] != port_links['Community_target']]
            port_extra_links.append(len(extra_links))

            k_is = port_links.groupby('Community_target')['Community_source'].count()
            k = dict_degree.get(port)
            if k_is.empty:
                p = np.nan
            else:
                p = round(1 - sum((k_is / k) ** 2), 4)
            list_p.append(p)

        nodes_copy['B'] = port_extra_links
        nodes_copy['Z'] = port_intra_links
        nodes_copy['P'] = list_p
        nodes_copy[['B', 'Z']] = nodes_copy.groupby(community_col, as_index=False)['B', 'Z'].apply(cal_zscore).round(4)
    return nodes_copy


def hubs_diversity():
    df_bzp = cal_bzp(Edges, Nodes, 'Community', 'K')
    df_bzp = df_bzp[['id', 'Community', 'K', 'B', 'Z', 'P']]

    communities = sorted(pd.unique(Nodes['Community']))
    df_bzp_sub_all = pd.DataFrame()
    for module in communities:
        df_nodes = Nodes[Nodes['Community'] == module]
        nodelist = df_nodes['id'].values
        sub_g = nx.subgraph(G, nodelist)
        df_edges = nx.to_pandas_edgelist(sub_g)
        df_bzp_sub = cal_bzp(df_edges, df_nodes, 'sub_Community', 'sub_K')
        df_bzp_sub_all = pd.concat([df_bzp_sub_all, df_bzp_sub], axis=0)
    df_bzp_sub_all = df_bzp_sub_all[['id', 'B', 'Z', 'P']]
    df_bzp_sub_all.rename(columns={'B': 'B_sub', 'Z': 'Z_sub', 'P': 'P_sub'}, inplace=True)
    df_nodes = pd.merge(df_bzp, df_bzp_sub_all, on='id', suffixes=('', '_sub'))

    df_b = df_nodes[(df_nodes['B'] >= 1.5)]
    df_z = df_nodes[(df_nodes['Z'] >= 1.5)]
    df_p = df_nodes[(df_nodes['P'] >= 0.7)]

    b = len(df_b) / len(df_nodes) * 100
    z = len(df_z) / len(df_nodes) * 100
    p = len(df_p) / len(df_nodes) * 100
    non_hub = len(df_nodes[(df_nodes['B'] < 1.5) & (df_nodes['Z'] < 1.5) & (df_nodes['P'] < 0.7)]) / len(df_nodes) * 100
    print('The in-text result:')
    print()
    print('"In the GLSN there exist only a few hub ports (Fig. 5): the fractions of provincial hubs, gateway hubs and '
          'connector hubs are {:.1f}%, {:.1f}% and {:.1f}% respectively, with {:.1f}% of the world ports being '
          'non-hubs."'.format(z, b, p, non_hub))
    print()

    b_zp = len(df_b[(df_b['Z'] >= 1.5) | (df_b['P'] >= 0.7)]) / len(df_b) * 100
    b_z = len(df_b[(df_b['Z'] >= 1.5) & (df_b['P'] < 0.7)]) / len(df_b) * 100
    b_p = len(df_b[(df_b['Z'] < 1.5) & (df_b['P'] >= 0.7)]) / len(df_b) * 100
    b_zp1 = len(df_b[(df_b['Z'] >= 1.5) & (df_b['P'] >= 0.7)]) / len(df_b) * 100

    print('The in-text result:')
    print()
    print('"As indicated in Fig. 5, {:.1f}% of those gateway hubs also play at least another hub role in the GLNS: '
          '{:.1f}% of them are provincial hubs within their individual communities, {:.1f}% connector hubs, and the '
          'rest {:.1f}% both provincial hubs and connector hubs."'.format(b_zp, b_z, b_p, b_zp1))
    print()
    z_bp = len(df_z[(df_z['B'] < 1.5) & (df_z['P'] < 0.7)]) / len(df_z) * 100
    p_zb = len(df_p[(df_p['B'] < 1.5) & (df_p['Z'] < 1.5)]) / len(df_p) * 100
    print('The in-text result:')
    print()
    print('"When examining those provincial hubs and connector hubs, however, {:.1f}% of the former and {:.1f}% of the '
          'latter turned out to be without any other hub roles (Fig. 5)."'.format(z_bp, p_zb))
    print()

    return df_nodes


def plot_role(data, param, threshold):
    data = data.apply(pd.to_numeric, errors='coerce')
    fig = plt.figure(figsize=(9, 1.3))
    ax1 = fig.add_subplot(121)
    data_b = data.sort_values(param, ascending=False)
    ax1.plot(range(1, len(data)+1), data_b[param], 'bo', markersize=1.5)
    ax1.set_xlim([-5, 980])

    anno_dict = dict(arrowstyle="-", linestyle=pltstyle.get_linestyles('loosely dashed'),
                      color='r', linewidth=3)

    pltstyle.axes_style(ax1)

    ax1.set_yticks([np.floor(data[param].min()), threshold, np.ceil(data[param].max())])
    ax1.set_ylabel(r'${}$'.format(param), fontsize=26)

    param_sub = param + '_sub'
    ax2 = fig.add_subplot(122)
    data_sub = data[data['Community'] != 5]
    data_b_sub = data_sub.sort_values(['Community', param_sub], ascending=[True, False])

    ax2.plot(range(1, len(data_sub) + 1), data_b_sub[param_sub], 'bo', markersize=threshold)
    ax2.set_xlim([1, len(data_sub)])

    ylims = [np.floor(data[param].min()) - 0.03, np.ceil(data[param].max())]
    ax1.set_ylim(ylims)
    ax2.set_ylim(ylims)
    if param == 'B':
        ax1.set_yticks([-1.0, 1.5, 8.0])
    if param == 'P':
        ax1.set_yticks([0, 0.7, 1.0])
    if param == 'Z':
        ax1.set_yticks([-2.0, 1.5, 5.0])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.annotate("", xy=(ax1.get_xlim()[0], threshold), xytext=(ax1.get_xlim()[1], threshold), arrowprops=anno_dict)
    ax2.annotate("", xy=(ax2.get_xlim()[0], threshold), xytext=(ax2.get_xlim()[1], threshold), arrowprops=anno_dict)

    arrow_dict = pltstyle.anno_style('k')
    nnodes = data_sub.groupby('Community')['id'].count().values
    nnodes = list(nnodes.cumsum())
    nnodes.pop()

    for i in nnodes:
        ax2.annotate("", xy=(i, ax2.get_ylim()[0]), xytext=(i, ax2.get_ylim()[1]), arrowprops=arrow_dict)

    pltstyle.axes_style(ax2)
    plt.tight_layout(pad=-0.35)
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Multiscale_modularity_and_hubs_diversity')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Fig. 4 Results for ports...(a) ' + param + '.png'
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def pr_plot(data):
    params = [['Z', 'B'], ['P', 'B'], ['P', 'Z']]
    thresholds = [[1.5, 1.5], [0.7, 1.5], [0.7, 1.5]]
    for param, threshold in zip(params, thresholds):
        param1 = param[0]
        param2 = param[1]
        threshold1 = threshold[0]
        threshold2 = threshold[1]
        fig = plt.figure(num=1, figsize=(5, 5))
        ax = fig.add_subplot(111)

        x = data[param1]
        y = data[param2]

        ax.plot(x, y, 'o', c='gray', alpha=0.9, mec='k', mew=1)
        ax.set_xlim([np.floor(data[param1].min()) - 0.03, np.ceil(data[param1].max())])
        ax.set_ylim([np.floor(data[param2].min()) - 0.015, np.ceil(data[param2].max())])

        anno_dict = dict(arrowstyle="-", linestyle=pltstyle.get_linestyles('loosely dashed'),
                         color='r', linewidth=3)

        if param1 == 'Z':
            ax.set_xlabel(r'Inside-module degree, ${}$'.format(param1), fontsize=20)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        if param1 == 'P':
            ax.set_xlim(ax.get_xlim()[0], 0.85)
            ax.set_xlabel(r'Participation coefficient, ${}$'.format(param1), fontsize=20)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        if param2 == 'B':
            ax.set_ylabel(r'Outside-module degree, ${}$'.format(param2), fontsize=20)
        if param2 == 'Z':
            ax.set_ylabel(r'Inside-module degree, ${}$'.format(param2), fontsize=20)

        ax.annotate("", xy=(threshold1, ax.get_ylim()[0]), xytext=(threshold1, ax.get_ylim()[1]),
                    arrowprops=anno_dict)
        ax.annotate("", xy=(ax.get_xlim()[0], threshold2), xytext=(ax.get_xlim()[1], threshold2),
                    arrowprops=anno_dict)

        f1 = np.polyfit(x, y, 1)
        p1 = np.poly1d(f1)
        yvals = p1(x)
        r2 = round(r2_score(y, yvals), 3)
        ax.plot(x, yvals, 'b-', linewidth=3.5)
        corr = round(stats.pearsonr(y, yvals)[0], 3)
        p = round(stats.pearsonr(y, yvals)[1], 4)

        left = ax.get_xlim()[0] + 0.065
        bottom = ax.get_ylim()[1] - 2.5
        ax.text(left, bottom, r'$R^2={}$'.format(r2) + '\n' + r'$corr= {}$'.format(corr) + '\n' +
                r'$p-value={}$'.format(p),
                fontsize=20)

        pltstyle.axes_style(ax)
        if SAVE_RESULT:
            save_path = os.path.join('output', 'Multiscale_modularity_and_hubs_diversity')
            if os.path.exists(save_path):
                pass
            else:
                os.makedirs(save_path)
            filename = 'Fig. 4 Results for ports...(b) ' + param1 + param2 + '.png'
            plt.savefig(save_path + '/' + filename, bbox_inches='tight')
            print('The result file "{}" saved at: "{}"'.format(filename, save_path))
            print()
        else:
            plt.show()
        plt.close('all')


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Subsection titled "Multiscale modular structure"')
    print('Section titled "Multiscale modularity and hubs diversity"')
    print('*********************************')
    print()
    print('-----------------------------------------------------------------------------')
    print('Output:')
    print()
    edge_plot()

    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Subsection titled "Hubs diversity"')
    print('Section titled "Multiscale modularity and hubs diversity"')
    print('*********************************')
    print()
    print('-----------------------------------------------------------------------------')
    print('Output:')
    print()
    df_nodes = hubs_diversity()
    params = ['P', 'B', 'Z']
    thresholds = [0.7, 1.5, 1.5]
    for param, threshold in zip(params, thresholds):
        plot_role(df_nodes, param, threshold)

    pr_plot(df_nodes)
