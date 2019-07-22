#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""

from configure import *


def cal_zscore(data):
    means = np.mean(data)
    std = np.std(data, ddof=1)
    data = (data - means) / std

    return data


def cal_corr(x, y):
    corr = round(stats.pearsonr(x, y)[0], 2)
    pval = round(stats.pearsonr(x, y)[1], 4)
    return corr, pval


def cal_bzp(df_nodes, df_edges):
    df_edges_copy = df_edges.copy()
    tmp = df_edges['source'].copy()
    df_edges_copy['source'] = df_edges_copy['target']
    df_edges_copy['target'] = tmp
    df_edges = pd.concat([df_edges, df_edges_copy], axis=0)

    dict_module = dict(zip(df_nodes['id'], df_nodes['Community']))
    dict_degree = dict(zip(df_nodes['id'], df_nodes['K']))
    new_cols = ['Community_source', 'Community_target']
    df_edges[new_cols] = df_edges[['source', 'target']].applymap(dict_module.get)

    idx_intra_links = []
    idx_extra_links = []
    list_p = []
    for port in df_nodes['id']:
        port_links = df_edges.loc[(df_edges['source'] == port)]
        intra_links = port_links[port_links['Community_source'] == port_links['Community_target']]
        idx_intra_links.append(len(intra_links))
        extra_links = port_links[port_links['Community_source'] != port_links['Community_target']]
        idx_extra_links.append(len(extra_links))

        k_is = port_links.groupby('Community_target')['Community_source'].count()
        k = dict_degree.get(port)
        p = round(1 - sum((k_is / k) ** 2), 4)
        list_p.append(p)
    df_nodes['B'] = idx_extra_links
    df_nodes['Z'] = idx_intra_links
    df_nodes['P'] = list_p
    df_nodes[['B', 'Z']] = df_nodes.groupby('Community')['B', 'Z'].apply(cal_zscore).round(4)

    return df_nodes


def find_sc(df_nodes, graph):
    params = ['B', 'Z', 'P', 'K', 'BC']
    num_modules = len(pd.unique(df_nodes['Community']))
    dict_freq = {'B': 0, 'Z': 0, 'P': 0, 'K': 0, 'BC': 0}
    num_sc_nodes = {'B': 0, 'Z': 0, 'P': 0, 'K': 0, 'BC': 0}
    for param in params:
        if param == 'K' or param == 'BC':
            df_nodes[param] = round(cal_zscore(df_nodes[param]), 4)
        df_nodes[param] = pd.to_numeric(df_nodes[param], errors='coerce')
        df_nodes = df_nodes.sort_values(param, ascending=True)
        list_value = list(set(df_nodes[pd.notnull(df_nodes[param])][param].round(4).values))

        list_density = []
        list_nodes = []
        for value in list_value:
            nodelist = df_nodes[df_nodes[param] >= value]['id']
            H = graph.subgraph(nodelist)
            density = round(nx.density(H), 4)
            list_density.append(density)
            list_nodes.append(len(nodelist))
        density = pd.DataFrame()
        density[param] = list_value
        density['num_nodes'] = list_nodes
        density['Density'] = list_density
        density.sort_values(param, ascending=False, inplace=True)

        if param == 'P':
            param_threshold = 0.7
        else:
            param_threshold = 1.5
        ix = (density[param] >= param_threshold) & (density['Density'] >= 0.8)
        if density[ix].empty:
            continue
        else:
            coor_x = density.loc[ix, param].values.tolist()[-1]
            num_sc_nodes[param] = density.loc[ix, 'num_nodes'].values.tolist()[-1]
            ix_sc_nodes = df_nodes[param] >= coor_x
            include_modules = len(pd.unique(df_nodes.loc[ix_sc_nodes, 'Community']))
            if include_modules == num_modules:
                num_sc_nodes[param] = density.loc[ix, 'num_nodes'].values.tolist()[-1]
                dict_freq[param] = 1
    return dict_freq, num_sc_nodes


def cal_pr(data):
    data = data.apply(pd.to_numeric, errors='coerce')
    modules = sorted(pd.unique(data['Community']))
    module_data = []
    for module in modules:
        module_data.append(data[data['Community'] == module])

    params = ['B', 'Z', 'P', 'K', 'BC', 'phi', 'rho_C', 'rho_CM']
    list_all_pearsonr = []
    for param in params:
        list_pearsonr = []
        data_dropna_glsn = data.dropna(subset=[param])
        glsn_pr = cal_corr(data_dropna_glsn[param], data_dropna_glsn['Capacity'])[0]
        for df in module_data:
            data_dropna = df.dropna(subset=[param])
            pearsonr = cal_corr(data_dropna[param], data_dropna['Capacity'])[0]
            list_pearsonr.append(pearsonr)
        list_pearsonr = pd.Series(list_pearsonr)
        list_pearsonr.dropna(inplace=True)
        module_pr = np.mean(list_pearsonr)
        list_all_pearsonr.append([glsn_pr, module_pr])
    df_pearsonr = pd.DataFrame(list_all_pearsonr)
    df_pearsonr.columns = ['GLSN', 'C']
    df_pearsonr['Network indicators'] = params
    return df_pearsonr


def country_level_pr(df_nodes, df_edges, year, num_sc):
    dict_port_country = dict(zip(df_nodes['id'], df_nodes['Country Code']))
    df_tv = pd.read_csv('../data/other data/TV_' + year + '.csv')

    source_cols = [col for col in df_edges.columns if 'source' in col]
    target_cols = [col for col in df_edges.columns if 'target' in col]
    tmp = df_edges[target_cols].copy()
    Edges_copy = df_edges.copy()
    Edges_copy[target_cols] = df_edges[source_cols]
    Edges_copy[source_cols] = tmp
    df_edges = pd.concat([df_edges, Edges_copy], axis=0)
    df_edges['Country Code_source'] = df_edges['source'].apply(dict_port_country.get)
    df_edges['Country Code_target'] = df_edges['target'].apply(dict_port_country.get)
    df_edges = df_edges[df_edges['Country Code_source'] != df_edges['Country Code_target']]
    num_outside_connections = df_edges.groupby('Country Code_source', as_index=False)['source'].count()
    num_outside_connections.columns = ['Country Code', '# all connections']

    sc_ports = df_nodes.sort_values('B', ascending=False)['id'][:num_sc]
    df_nodes['SC Type'] = np.where(df_nodes['id'].isin(sc_ports), 'SC', 'NSC')
    dict_sc_type = dict(zip(df_nodes['id'], df_nodes['SC Type']))

    df_edges['source_SC Type'] = df_edges['source'].apply(dict_sc_type.get)
    df_edges['target_SC Type'] = df_edges['target'].apply(dict_sc_type.get)
    ix_sc_ports = df_edges['source_SC Type'] == 'SC'
    ix_nsc_ports = df_edges['source_SC Type'] == 'NSC'
    df_res_sc = df_edges[ix_sc_ports].groupby(['Country Code_source'], as_index=False)['source'].count()
    df_res_sc.columns = ['Country Code', '# SC connections']
    df_res_nsc = df_edges[ix_nsc_ports].groupby(['Country Code_source'], as_index=False)['source'].count()
    df_res_nsc.columns = ['Country Code', '# NSC connections']
    df_res = pd.merge(df_res_sc, df_res_nsc, on='Country Code', how='outer')
    df_res = pd.merge(num_outside_connections, df_res, on='Country Code', how='left')
    df_lsc = pd.merge(df_res, df_tv, on='Country Code')
    df_lsc.fillna(0, inplace=True)

    cal_cols = [col for col in df_lsc.columns if 'connections' in col]
    list_corr = []
    for col in cal_cols:
        corr, p_value = cal_corr(df_lsc[col], df_lsc['TV'])
        list_corr.append(corr)

    return list_corr


def country_pair_level_pr(df_nodes, df_edges, year, num_sc):
    def _process_edges(edge_data, list_core_ports, list_none_core_port):
        dict_port_country = dict(zip(df_nodes['id'], df_nodes['Country Code']))
        edge_data['Country Code_source'] = edge_data['source'].apply(dict_port_country.get)
        edge_data['Country Code_target'] = edge_data['target'].apply(dict_port_country.get)

        # core connections
        ix_core = (edge_data['source'].isin(list_core_ports)) & (edge_data['target'].isin(list_core_ports))
        edge_data.loc[ix_core, 'property'] = 'core'
        # feeder connections
        ix_feeder1 = (edge_data['source'].isin(list_core_ports)) & (edge_data['target'].isin(list_none_core_port))
        ix_feeder2 = (edge_data['target'].isin(list_core_ports)) & (edge_data['source'].isin(list_none_core_port))
        ix_feeder = ix_feeder1 | ix_feeder2
        edge_data.loc[ix_feeder, 'property'] = 'feeder'
        # local connections
        ix_local = (edge_data['source'].isin(list_none_core_port)) & (edge_data['target'].isin(list_none_core_port))
        edge_data.loc[ix_local, 'property'] = 'local'

        df_edges_copy = edge_data.copy()
        source_cols = [col for col in edge_data.columns if 'source' in col]
        target_cols = [col for col in edge_data.columns if 'target' in col]
        tmp = edge_data[source_cols].copy()
        df_edges_copy[source_cols] = df_edges_copy[target_cols]
        df_edges_copy[target_cols] = tmp
        edges = pd.concat([edge_data, df_edges_copy], axis=0)

        edges = edges[edges['Country Code_source'] != edges['Country Code_target']]

        edges_core_related = edges[(edges['property'] == 'core') | (edges['property'] == 'feeder')]
        edges_local = edges[edges['property'] == 'local']
        edgelist = [edges_core_related, edges_local]
        return edges, edgelist

    df_btv = pd.read_csv('../data/other data/BTV_' + year + '.csv')

    list_core_ports = df_nodes.sort_values('B', ascending=False)['id'][:num_sc].values.tolist()
    list_none_core_port = list(set(df_nodes['id']).difference(set(list_core_ports)))
    edges, edgelist = _process_edges(df_edges, list_core_ports, list_none_core_port)

    num_connections = edges.groupby(['Country Code_source', 'Country Code_target'], as_index=False)['source'].count()
    num_connections.rename(columns={'source': '# all connections'}, inplace=True)
    for data in edgelist:
        lfc = data.groupby(['Country Code_source', 'Country Code_target'], as_index=False)['source'].count()
        num_connections = pd.merge(num_connections, lfc, on=['Country Code_source', 'Country Code_target'], how='outer')
    num_connections.columns = ['Country Code_source', 'Country Code_target', '# all connections',
                               '# SC connections', '# NSC connections']

    df_res = pd.merge(num_connections, df_btv, on=['Country Code_source', 'Country Code_target'])
    df_res.fillna(0, inplace=True)

    connection_cols = [col for col in df_res.columns if 'connections' in col]
    list_corr = []
    for col in connection_cols:
        corr, p_value = cal_corr(df_res[col], df_res['BTV'])
        list_corr.append(corr)

    return list_corr


def plot_sc_freq(data, iterations, year):
    percs = [0.1, 0.2, 0.3, 0.4, 0.5]

    n_rows = 1
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5))
    width = 0.4

    x = range(1, 6)

    for i in range(n_cols):
        ax = axes[i]
        perc = percs[i]
        lab = 'method #3'
        data_copy = data.loc[i, ['B', 'Z', 'P', 'K', 'BC']]
        ax.bar(x, data_copy, width, color='k', alpha=0.75, label=lab)
        pltstyle.axes_style(ax)
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=1.8, length=7,
                       pad=7, labelsize=22)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(data.columns, style='italic')
        ax.set_title('GLSN of ' + str(year) + '\n' + '{:.0f}% edges'.format(perc * 100), fontsize=24)
        ax.set_ylim(0, iterations + iterations*0.15)

    axes[0].set_ylabel(r'$Frequency$', fontsize=26)
    t = plt.suptitle('Note: The number of iterations of the experiment: in your test, {}; in the '
                     'manuscript, 1000.'.format(iterations), color='red', style='italic',
                     fontsize=32, x=0.5, y=1.1)
    t.set_bbox(dict(facecolor='gray', alpha=0.3, edgecolor=None))

    axes[0].legend(fontsize=24)
    plt.tight_layout(h_pad=2)

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 12_2')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)

        if year == '2015':
            filename = 'Supplementary Fig. 20 Frequency...percent of edges-method #3-(a) GLSN of 2015.png'
        else:
            filename = 'Supplementary Fig. 20 Frequency...percent of edges-method #3-(b) GLSN of 2017.png'
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def autolabel1(ax, rects, label):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    for rect in rects:
        height = rect.get_height()
        if height == 0:
            pass
        elif label == '#1':
            ax.text(rect.get_x() + rect.get_width() * 0.5, 1.01 * height,
                    '{}'.format(label), ha='center', va='bottom', fontsize=14, weight='bold')
        elif label == '#2':
            ax.text(rect.get_x() + rect.get_width() * 0.5, 1.01 * height,
                    '{}'.format(label), ha='center', va='bottom', fontsize=14, weight='bold')
        elif label == '#3':
            ax.text(rect.get_x() + rect.get_width() * 0.5, 1.01 * height,
                    '{}'.format(label), ha='center', va='bottom', fontsize=14, weight='bold')


def autolabel(ax, rects, xpos='center'):

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    labels = [r'$\circled1$', r'$\circled2$', r'$\circled3$', r'$\circled4$', r'$\circled5$']

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        if height == 0:
            pass
        else:
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height, label,
                    ha=ha[xpos], va='bottom', fontsize=28)


def plot_hist(year, tv_mean, tv_se, btv_mean, btv_se, iterations):
    tv_mean.fillna(0, inplace=True)
    btv_mean.fillna(0, inplace=True)
    mean_list = [tv_mean, btv_mean]
    se_list = [tv_se, btv_se]

    n_rows = 1
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7))

    step = 0.3
    X = [1, 2.6, 4.2]
    a_list = [-2, -1, 0, 1, 2]

    xtick_list = []
    for x in X:
        for a in a_list:
            xtick_list.append(x + a * step)

    width = 0.2

    for i in range(n_cols):
        ax = axes[i]

        data = mean_list[i]
        ses = se_list[i]
        rects1 = ax.bar(xtick_list[:5], data['# all connections'], width,
                        yerr=ses['# all connections'],
                        color='k', capsize=7, error_kw={'lw': 3}, alpha=0.75, label='# all connections')
        rects2 = ax.bar(xtick_list[5:10], data['# SC connections'], width,
                        yerr=ses['# SC connections'],
                        color='red', capsize=7, error_kw={'lw': 3}, alpha=0.75, label='# SC connections')
        rects3 = ax.bar(xtick_list[10:15], data['# NSC connections'], width,
                        yerr=ses['# NSC connections'],
                        color='green', capsize=7, error_kw={'lw': 3}, alpha=0.75, label='# NSC connections')

        autolabel(ax, rects1)
        autolabel(ax, rects2)
        autolabel(ax, rects3)
        plot_style(ax, xtick_list)
    axes[0].set_xlabel('at the country level', labelpad=10, fontsize=26, weight='medium')
    axes[1].set_xlabel('at the country pair level', labelpad=10, fontsize=26, weight='medium')
    axes[0].set_ylabel(r'pearson r', style='italic', fontsize=28, weight='medium')
    axes[0].set_title('(c) GLSN of {}, method #3'.format(year), loc='left', fontsize=24, pad=10)

    axes[0].text(x=0, y=1.2, s='Percent of edges:' +'\n' + '$\circled1$ 10%  $\circled2$ 20% ' + '\n' +
                               '$\circled3$ 30%  $\circled4$ 40%   $\circled5$ 50%', fontsize=22)

    t = plt.suptitle('Note: The number of iterations of the experiment: in your test, {}; in the '
                     'manuscript, 1000.'.format(iterations) + '', color='red', style='italic',
                     fontsize=22, x=0.5, y=1.03)
    t.set_bbox(dict(facecolor='gray', alpha=0.3, edgecolor=None))

    plt.tight_layout()
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 12_2')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if year == '2015':
            filename = 'Supplementary Fig. 21 Pearson correlation coefficients...socio-economic...2015 ' \
                       'dataset (c).png'
        else:
            filename = 'Supplementary Fig. 22 Pearson correlation coefficients...socio-economic...2017 ' \
                       'dataset (c).png'
        plt.savefig(save_path + '/' + filename, transparent=False, bbox_inches='tight')
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def plot_style(ax, xtick_list):
    ax.set_ylim(0, 1.05)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    xlabels = ['# all\n connections', '# SC\n connections', '# NSC\n connections']
    pos_ticks = [xtick_list[2], xtick_list[7], xtick_list[12]]

    ax.set_xticks(pos_ticks)
    ax.set_xticklabels(xlabels, fontsize=20, weight='medium')

    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    ax.tick_params(axis='y', direction='in', top=False, right=False, which='major', width=1.8, length=7,
                   pad=5, labelsize=20)
    ax.tick_params(axis='x', direction='in', top=True, right=True, which='major', width=1.8, length=7,
                   pad=5, labelsize=20)


def permutation_heat_map(year, iterations):
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    percentage_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    data_list = []
    for percentage in percentage_list:
        data = pd.read_csv('output/dis_process/Permutation-KK-Dis_' + year + '_method3_' + str(percentage) + '.csv')
        data = data[data[dis_col] != 0]
        data.sort_values(dis_col, ascending=True, inplace=True)
        data_list.append(data)

    n_rows = 1
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.3, 3.8))

    area = 20
    for i in range(n_cols):
        ax = axes[i]
        data = data_list[i]

        bounds = [1, 10, 100, 1000, 10000, 100000]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

        p = ax.scatter(data['source_Degree'], data['target_Degree'], s=area, c=data[dis_col],
                       norm=norm, cmap='Reds', edgecolors='white', linewidths=0.05)
        ax.set_title("rewire {:.0f}% edges".format(percentage_list[i]*100), fontsize=18, pad=8, weight='medium')

        if i != 0:
            ax.set_yticklabels(ax.get_yticks(), visible=False)
        if i == 0:
            ax.set_ylabel(r'$K$', fontsize=22)
        if i == 4:
            axins = inset_axes(ax,
                               width="10%",
                               height="100%",
                               loc='upper left',
                               bbox_to_anchor=(1.1, 0., 1, 1),
                               bbox_transform=ax.transAxes,
                               borderpad=0)
            cbar = fig.colorbar(p, cax=axins)

            plt.setp(axins.get_yticklabels(), visible=True)
            axins.set_yticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$'], fontsize=14)
            axins.set_ylabel('Distance (km)', fontsize=20, rotation='vertical')
            cbar.ax.tick_params(axis='y', which='major', labelsize=18, direction='in', width=2, length=5)
        pltstyle.axes_style(ax)
        ax.loglog()

        ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=2.0, length=7.5,
                       labelsize=16, pad=5)
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor', width=1.5, length=5)
        ax.set_xlabel(r'$K$', fontsize=22)

        if year == '2015':
            ax.set_xlim([1, 350])
            ax.set_ylim([1, 350])
        else:
            ax.set_xlim([1, 300])
            ax.set_ylim([1, 300])

    plt.suptitle('Note: The number of iterations of the experiment: in your test, {}; in the '
                 'manuscript, 1000.'.format(iterations) + '\n' + '\n' + '(e) modified GLSNs of {} by method #3'.format(year), color='k', fontsize=22, x=0.5, y=1.25)

    plt.tight_layout()

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 12_2')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)

        if year == '2015':
            filename = 'Supplementary Fig. 18 Edge distributions...(e).png'
        else:
            filename = 'Supplementary Fig. 19 Edge distributions...(e).png'

        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def permutation_kk_distance(df_nodes, df_edges):

    dict_degree = dict(zip(df_nodes['id'], df_nodes['K']))
    df_edges['source_Degree'] = df_edges['source'].apply(dict_degree.get)
    df_edges['target_Degree'] = df_edges['target'].apply(dict_degree.get)

    return df_edges


def sc_robustness(iterations):
    years = ['2015']
    percs = [0.1, 0.2, 0.3, 0.4, 0.5]

    for year in years:
        df_freq_all = pd.DataFrame()
        list_prob = []
        df_pr_all = pd.DataFrame()
        df_country_pr_all = pd.DataFrame()
        df_country_pair_pr_all = pd.DataFrame()
        nodes = pd.read_csv('../data/other data/Nodes_' + year + '_All_P.csv')
        edges = pd.read_csv('../data/GLSN data/Edges_' + year + '_All_P.csv', header=None)
        edges.columns = ['source', 'target']
        dict_country = dict(zip(nodes['id'], nodes['Country Code']))
        dict_cap = dict(zip(nodes['id'], nodes['Capacity']))

        df_dis = pd.read_csv('../data/other data/Distance_SR_GC_' + year + '.csv')
        dict_dis = dict(zip(df_dis['Edge'], df_dis[dis_col]))
        edges['Edge'] = edges['source'].astype(str) + str('--') + edges['target'].astype(str)
        edges[dis_col] = edges['Edge'].apply(dict_dis.get)
        prob_dis_emp = len(edges[edges[dis_col] >= 10000]) / len(edges) * 100

        for perc in percs:
            data_path = os.path.join('../', 'data', 'downloaded data files', 'note12_2', year, 'method3', str(perc))
            all_freq = []
            country_pr = []
            country_pair_pr = []

            edgedata_all = pd.DataFrame()
            list_prob_dis = []
            for rt in range(1, iterations+1):
                df_nodes = pd.read_csv(data_path + '/Nodes/' + str(rt) + '.csv')
                df_nodes['Country Code'] = df_nodes['id'].apply(dict_country.get)
                df_nodes['Capacity'] = df_nodes['id'].apply(dict_cap.get)
                df_edges = pd.read_csv(data_path + '/Edges/' + str(rt) + '.csv', header=None)
                df_edges.columns = ['source', 'target']
                graph = nx.from_pandas_edgelist(df_edges, 'source', 'target', create_using=nx.Graph())

                df_edges['Edge'] = df_edges['source'].astype(str) + str('--') + df_edges['target'].astype(str)
                df_edges[dis_col] = df_edges['Edge'].apply(dict_dis.get)
                prob_dis = len(df_edges[df_edges[dis_col] >= 10000]) / len(df_edges) * 100
                list_prob_dis.append(prob_dis)

                edgedata = permutation_kk_distance(df_nodes, df_edges)
                edgedata = edgedata.groupby(['source_Degree', 'target_Degree'], as_index=False)[
                    dis_col].mean().round()
                edgedata = merge_undirected(edgedata)
                edgedata = edgedata.groupby(['source_Degree', 'target_Degree'], as_index=False)[
                    dis_col].mean().round()
                edgedata_all = pd.concat([edgedata_all, edgedata], axis=0)

                df_nodes = cal_bzp(df_nodes, df_edges)
                df_pr = cal_pr(df_nodes)
                df_pr['percent of edges'] = len(df_pr) * [perc*100]
                df_pr_all = pd.concat([df_pr_all, df_pr], axis=0, sort=False)

                param_freq, num_sc = find_sc(df_nodes, graph)
                all_freq.append(param_freq)
                if param_freq['B'] == 1:
                    n_sc = num_sc['B']
                    country_pr.append(country_level_pr(df_nodes, df_edges, year, n_sc))
                    country_pair_pr.append(country_pair_level_pr(df_nodes, df_edges, year, n_sc))
                else:
                    country_pr.append([np.nan, np.nan, np.nan])
                    country_pair_pr.append([np.nan, np.nan, np.nan])

            list_prob.append(np.mean(list_prob_dis))

            df_freq = pd.DataFrame(all_freq)
            df_freq['percent of edges'] = len(df_freq) * [perc]
            df_freq_all = pd.concat([df_freq_all, df_freq], axis=0)

            df_country_pr = pd.DataFrame(country_pr)
            df_country_pr.columns = ['# all connections', '# SC connections', '# NSC connections']
            df_country_pr['percent of edges'] = len(df_country_pr) * [perc]
            df_country_pr_all = pd.concat([df_country_pr_all, df_country_pr], axis=0)
            df_country_pair_pr = pd.DataFrame(country_pair_pr)
            df_country_pair_pr.columns = ['# all connections', '# SC connections', '# NSC connections']
            df_country_pair_pr['percent of edges'] = len(df_country_pair_pr) * [perc]
            df_country_pair_pr_all = pd.concat([df_country_pair_pr_all, df_country_pair_pr], axis=0)

            df_rand_kk = edgedata_all.groupby(['source_Degree', 'target_Degree'], as_index=False)[
                dis_col].mean().round()
            df_rand_kk = merge_undirected(df_rand_kk)
            df_rand_kk = df_rand_kk.groupby(['source_Degree', 'target_Degree'], as_index=False)[
                dis_col].mean().round()
            df_full = full_degree(nodes)
            df_rand_kk = pd.merge(df_rand_kk, df_full, on=['source_Degree', 'target_Degree'], how='right')
            df_rand_kk.fillna(0, inplace=True)
            df_rand_kk.sort_values(dis_col, inplace=True)
            spath = os.path.join('output', 'dis_process')
            if os.path.exists(spath):
                pass
            else:
                os.makedirs(spath)
            df_rand_kk.to_csv(spath + '/Permutation-KK-Dis_' + year + '_method3_' + str(perc) + '.csv',
                              index=False, header=True)

        if df_country_pr_all.empty or df_country_pair_pr_all.empty:
            print("No structural-core organization has found.")
        else:
            country_pr_mean = df_country_pr_all.groupby(['percent of edges'])[
                '# all connections', '# SC connections', '# NSC connections'].apply(np.mean).round(2)
            country_pair_pr_mean = df_country_pair_pr_all.groupby(['percent of edges'])[
                '# all connections', '# SC connections', '# NSC connections'].apply(np.mean).round(2)
            country_pr_se = df_country_pr_all.groupby(['percent of edges'])[
                                '# all connections', '# SC connections', '# NSC connections'].sem()
            country_pair_pr_se = df_country_pair_pr_all.groupby(['percent of edges'])[
                                     '# all connections', '# SC connections', '# NSC connections'].sem()
            plot_hist(year, country_pr_mean, country_pr_se, country_pair_pr_mean, country_pair_pr_se, iterations)

        permutation_heat_map(year, iterations)

        df_prob_res = pd.DataFrame()
        list_prob.insert(0, prob_dis_emp)
        df_prob_res['Probability (Distance >= 10^4 km)'] = list_prob
        df_prob_res_t = df_prob_res.T
        df_prob_res_t.columns = ['real GLSN', 'percent of edges: 10%', 'percent of edges: 20%',
                                 'percent of edges: 30%', 'percent of edges: 40%', 'percent of edges: 50%']
        df_prob_res_t.index = ['method #3']

        if SAVE_RESULT:
            save_path = os.path.join('output', 'Supplementary note 12_2')
            if os.path.exists(save_path):
                pass
            else:
                os.makedirs(save_path)
            if year == '2015':
                filename = 'Supplementary Fig. 18 Edge distributions....(b)-method #3.xlsx'
                s_name = 'GLSN of 2015_method #3'
            else:
                filename = 'Supplementary Fig. 19 Edge distributions....(b)-method #3.xlsx'
                s_name = 'GLSN of 2017_method #3'
            df_prob_res_t.to_excel(save_path + '/' + filename, index=False, float_format="%.1f", sheet_name=s_name)
            print()
            print('The result file "{}" saved at: "{}"'.format(filename, save_path))
            print()

        param_freq = df_freq_all.groupby(['percent of edges'], as_index=False)[
            'B', 'Z', 'P', 'K', 'BC'].apply(np.sum)
        plot_sc_freq(param_freq, iterations, year)

        param_pr_mean = df_pr_all.groupby(['Network indicators', 'percent of edges'],
                                          as_index=False)['GLSN', 'C'].apply(np.mean).round(2)

        if SAVE_RESULT:
            save_path = os.path.join('output', 'Supplementary note 12_2')
            if os.path.exists(save_path):
                pass
            else:
                os.makedirs(save_path)
            if year == '2015':
                filename = 'Supplementary table 8 Pearson correlation coefficients...2015...-method #3.xlsx'
                s_name = 'GLSN of 2015_method #3'
            else:
                filename = 'Supplementary table 9 Pearson correlation coefficients...2017...-method #3.xlsx'
                s_name = 'GLSN of 2017_method #3'
            param_pr_mean.to_excel(save_path + '/' + filename, sheet_name=s_name)
            print()
            print('The result file "{}" saved at: "{}"'.format(filename, save_path))
            print()


def merge_undirected(df_edges):
    df_edges2 = df_edges.copy()
    tmp = df_edges['target_Degree'].copy()
    df_edges2['target_Degree'] = df_edges2['source_Degree']
    df_edges2['source_Degree'] = tmp
    df_edges = df_edges.append(df_edges2, ignore_index=True)
    return df_edges


def full_degree(df_nodes):
    degree_max = df_nodes['K'].max()

    degrees = np.arange(1, degree_max+1, 1, dtype=int)
    s_degree = []
    t_degree = []
    for i in degrees:
        for j in degrees:
            s_degree.append(i)
            t_degree.append(j)
    df_full = pd.DataFrame()
    df_full['source_Degree'] = s_degree
    df_full['target_Degree'] = t_degree
    return df_full


def kk_dis_empirical(df_nodes, df_edges):

    dict_degree = dict(zip(df_nodes['id'], df_nodes['K']))
    df_edges['source_Degree'] = df_edges['source'].apply(dict_degree.get)
    df_edges['target_Degree'] = df_edges['target'].apply(dict_degree.get)
    df_edges = merge_undirected(df_edges)
    df_res = df_edges.groupby(['source_Degree', 'target_Degree'], as_index=False)[dis_col].mean().round()
    df_full = full_degree(df_nodes)
    df_res = pd.merge(df_res, df_full, on=['source_Degree', 'target_Degree'], how='right')
    df_res.fillna(0, inplace=True)
    df_res.sort_values(dis_col, inplace=True)

    return df_res


def startup(iterations):
    sc_robustness(iterations)
