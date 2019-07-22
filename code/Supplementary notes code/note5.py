#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


data_path = os.path.join('../data', 'downloaded data files', 'note5')


def cal_zscore(data):
    means = np.mean(data)
    std = np.std(data, ddof=1)
    data = (data - means) / std

    return data


def cal_corr(x, y):
    corr = round(stats.pearsonr(x, y)[0], 2)
    pval = round(stats.pearsonr(x, y)[1], 4)
    return corr, pval


def cal_bzp(df_nodes):
    df_edges_copy = Edges.copy()
    tmp = Edges['source'].copy()
    df_edges_copy['source'] = df_edges_copy['target']
    df_edges_copy['target'] = tmp
    df_edges = pd.concat([Edges, df_edges_copy], axis=0)

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


def find_sc(df_nodes):
    params = ['B', 'Z', 'P', 'K', 'BC']
    num_modules = len(pd.unique(df_nodes['Community']))
    dict_freq = {'B': 0, 'Z': 0, 'P': 0, 'K': 0, 'BC': 0}
    num_sc_nodes = {'B': 0, 'Z': 0, 'P': 0, 'K': 0, 'BC': 0}
    coor_x = {'B': 0, 'Z': 0, 'P': 0, 'K': 0, 'BC': 0}
    for param in params:
        if param == 'K' or param == 'BC':
            df_nodes[param] = cal_zscore(df_nodes[param])
        df_nodes = df_nodes.sort_values(param, ascending=True)
        df_nodes[param] = pd.to_numeric(df_nodes[param], errors='coerce')
        list_value = list(set(df_nodes[pd.notnull(df_nodes[param])][param].round(4).values))

        list_density = []
        list_nodes = []
        for value in list_value:
            nodelist = df_nodes[df_nodes[param] >= value]['id']
            H = G.subgraph(nodelist)
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
            coor_x[param] = density.loc[ix, param].values.tolist()[-1]
            num_sc_nodes[param] = density.loc[ix, 'num_nodes'].values.tolist()[-1]
            ix_sc_nodes = df_nodes[param] >= coor_x[param]
            include_modules = len(pd.unique(df_nodes.loc[ix_sc_nodes, 'Community']))
            if include_modules == num_modules:
                num_sc_nodes[param] = density.loc[ix, 'num_nodes'].values.tolist()[-1]
                dict_freq[param] = 1
    return dict_freq, num_sc_nodes, coor_x


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
        module_pr = np.mean(list_pearsonr)
        list_all_pearsonr.append([glsn_pr, module_pr])
    df_pearsonr = pd.DataFrame(list_all_pearsonr)
    df_pearsonr.columns = ['GLSN', 'Communities']
    df_pearsonr['Network indicators'] = params
    return df_pearsonr


def country_level_pr(num_sc):
    dict_port_country = dict(zip(Nodes['id'], Nodes['Country Code']))
    df_tv = pd.read_csv('../data/other data/TV_' + YEAR + '.csv')

    source_cols = [col for col in Edges.columns if 'source' in col]
    target_cols = [col for col in Edges.columns if 'target' in col]
    tmp = Edges[target_cols].copy()
    Edges_copy = Edges.copy()
    Edges_copy[target_cols] = Edges[source_cols]
    Edges_copy[source_cols] = tmp
    df_edges = pd.concat([Edges, Edges_copy], axis=0)
    df_edges['Country Code_source'] = df_edges['source'].apply(dict_port_country.get)
    df_edges['Country Code_target'] = df_edges['target'].apply(dict_port_country.get)
    df_edges = df_edges[df_edges['Country Code_source'] != df_edges['Country Code_target']]
    num_outside_connections = df_edges.groupby('Country Code_source', as_index=False)['source'].count()
    num_outside_connections.columns = ['Country Code', '# all connections']

    nodedata = Nodes.copy()
    sc_ports = nodedata.sort_values('B', ascending=False)['id'][:num_sc]
    nodedata['SC Type'] = np.where(nodedata['id'].isin(sc_ports), 'SC', 'NSC')
    dict_sc_type = dict(zip(nodedata['id'], nodedata['SC Type']))

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


def country_pair_level_pr(num_sc):
    def _process_edges(edge_data, list_core_ports, list_none_core_port):
        dict_port_country = dict(zip(Nodes['id'], Nodes['Country Code']))
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

    df_btv = pd.read_csv('../data/other data/BTV_' + YEAR + '.csv')
    nodedata = Nodes.copy()
    list_core_ports = nodedata.sort_values('B', ascending=False)['id'][:num_sc].values.tolist()
    list_none_core_port = list(set(nodedata['id']).difference(set(list_core_ports)))
    edgedata = Edges.copy()
    edges, edgelist = _process_edges(edgedata, list_core_ports, list_none_core_port)

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


def sc_robustness():
    all_freq = []
    country_pr = []
    country_pair_pr = []
    df_pr_all = pd.DataFrame()
    nodedata = Nodes.copy()
    for rt in range(1, iters+1):
        nodes = pd.read_csv(data_path + '/' + str(rt) + '.csv')
        dict_module = dict(zip(nodes['id'], nodes['Community']))
        nodedata['Community'] = nodedata['id'].apply(dict_module.get)
        df_nodes = cal_bzp(nodedata)
        df_pr = cal_pr(df_nodes)
        df_pr_all = pd.concat([df_pr_all, df_pr], axis=0, sort=False)
        param_freq, num_sc, coor_x = find_sc(df_nodes)
        if param_freq['B'] == 1:
            n_sc = num_sc['B']
            country_pr.append(country_level_pr(n_sc))
            country_pair_pr.append(country_pair_level_pr(n_sc))
        all_freq.append(param_freq)

    df_freq = pd.DataFrame(all_freq)
    param_freq = pd.DataFrame(df_freq.sum())
    param_freq.columns = ['frequency']
    param_freq['Network indicators'] = param_freq.index
    param_freq = param_freq.reset_index()
    plot_sc_freq(param_freq)

    param_pr_mean = df_pr_all.groupby('Network indicators').apply(np.mean).round(2)
    param_pr_se = round(df_pr_all.groupby('Network indicators').apply(np.std) / np.sqrt(iters), 3)
    list_params_custom = ['B', 'Z', 'P', 'K', 'BC', 'phi', 'rho_C', 'rho_CM']
    param_pr_mean = param_pr_mean.reset_index()
    param_pr_mean['Network indicators'] = param_pr_mean['Network indicators'].astype('category')
    param_pr_mean['Network indicators'].cat.reorder_categories(list_params_custom, inplace=True)
    param_pr_mean.sort_values('Network indicators', inplace=True)
    param_pr_se = param_pr_se.reset_index()
    param_pr_se['Network indicators'] = param_pr_se['Network indicators'].astype('category')
    param_pr_se['Network indicators'].cat.reorder_categories(list_params_custom, inplace=True)
    param_pr_se.sort_values('Network indicators', inplace=True)
    df_pr_res = pd.merge(param_pr_mean, param_pr_se, on='Network indicators', suffixes=(' (Pearson correlation coefficients)', ' (Standard errors)'))
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 5')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Supplementary table 3 Pearson correlation coefficients....xlsx'
        df_pr_res.to_excel(save_path + '/' + filename, index=False)
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))

    df_country_pr = pd.DataFrame(country_pr)
    df_country_pair_pr = pd.DataFrame(country_pair_pr)
    if df_country_pr.empty or df_country_pair_pr.empty:
        print("No structural-core organization has found.")
    else:
        plot_hist(df_country_pr, df_country_pair_pr)


def plot_sc_freq(data):
    list_params_custom = ['B', 'Z', 'P', 'K', 'BC']
    data['Network indicators'] = data['Network indicators'].astype('category')
    data['Network indicators'].cat.reorder_categories(list_params_custom, inplace=True)
    data.sort_values('Network indicators', inplace=True)

    fig = plt.figure(figsize=(5.5, 6))
    ax = fig.add_subplot(111)
    x = range(1, 6)
    width = 0.38
    ax.bar(x, data['frequency'], width, color='k', alpha=0.75)
    plt.ylabel(r'$Frequency$', fontsize=24)

    t = plt.title('Note: The number of iterations of the experiment:' + '\n' +
                  'in your test, {}; in the manuscript, 1000.'.format(iters), color='red', style='italic',
                  fontsize=14, pad=30)
    t.set_bbox(dict(facecolor='gray', alpha=0.3, edgecolor=None))

    plt.xticks(x, data['Network indicators'], style='italic', fontsize=22)
    ax.set_ylim(0, iters + iters*0.15)
    pltstyle.axes_style(ax)
    plt.tight_layout()
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 5')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Supplementary Fig. 9 Frequency with which a structural core was ' \
                   'detected...over {} realisations.png'.format(iters)
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    else:
        plt.show()
    plt.close()


def plot_hist(df_country_pr, df_country_pair_pr):
    tv_mean = df_country_pr.mean().round(2)
    btv_mean = df_country_pair_pr.mean().round(2)
    tv_se = stats.sem(df_country_pr).round(4)
    btv_se = stats.sem(df_country_pair_pr).round(4)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)

    x = [2, 4, 6]
    width = 0.5
    ax1.bar(x, tv_mean, width, color=['black', 'red', 'green'], alpha=0.8,
            yerr=tv_se, capsize=12, error_kw={'lw': 5})
    ax1.set_xlabel('at the country level', labelpad=10, fontsize=24)
    ax1.set_ylabel(r'$pearson\ r$', fontsize=24)
    ax1.set_ylim(0, 1.05)
    pltstyle.axes_style(ax1)

    ax2 = fig.add_subplot(122)
    ax2.bar(x, btv_mean, width, color=['black', 'red', 'green'], alpha=0.8,
            yerr=btv_se, capsize=10, error_kw={'lw': 5})
    ax2.set_xlabel('at the country pair level', labelpad=10, fontsize=24)
    ax2.set_ylim(0, 1)
    pltstyle.axes_style(ax2)
    xlabels = ['0', '# all\n connections', '# SC\n connections', '# NSC\n connections']
    ax1.set_xticklabels(xlabels, fontsize=20)
    ax2.set_xticklabels(xlabels, fontsize=20)

    t = plt.suptitle('Note: The number of iterations of the experiment: in your test, {}; in the '
                     'manuscript, 1000.'.format(iters), color='red', style='italic',
                     fontsize=20, x=0.5, y=1.1)
    t.set_bbox(dict(facecolor='gray', alpha=0.3, edgecolor=None))

    plt.tight_layout()
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 5')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Supplementary Fig. 10 Pearson correlation coefficients...averaged over {} repetitions ' \
                   'of the experiment.png'.format(iters)
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    else:
        plt.show()
    plt.close('all')


def startup():

    if os.path.exists(data_path):
        print('*********************************')
        print("Location in the manuscript text: ")
        print('Section titled "Supplementary note 5: Robustness of empirical findings on the '
              'structural-core organization of the GLSN to the non-detrimental property of the Louvain algorithm '
              'in community division"')
        print('*********************************')
        print()
        print('***************************RUN TIME WARNING***************************')
        print('It needs 7 hours for 1000 iterations of the corresponding experiments.')
        print()
        print('---------------------------------------------------------------------------------------------------')
        print('Output:')
        print()
        print('**********************************************************************************************')
        print('Note: The number of iterations of the experiment: in your test, {}; in '
              'the manuscript, 1000.'.format(iters))
        print('**********************************************************************************************')
        print()
        sc_robustness()
    else:
        print()
        print('Please download *downloaded data files.zip* file first!')
        sys.exit()
