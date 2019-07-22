#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""

from configure import *


data_path = os.path.join('../', 'data', 'downloaded data files', 'note11')


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

    port_intra_links = []
    port_extra_links = []
    list_p = []
    for port in df_nodes['id']:
        port_links = df_edges.loc[(df_edges['source'] == port)]
        intra_links = port_links[port_links['Community_source'] == port_links['Community_target']]
        port_intra_links.append(len(intra_links))
        extra_links = port_links[port_links['Community_source'] != port_links['Community_target']]
        port_extra_links.append(len(extra_links))
        k_is = port_links.groupby('Community_target')['Community_source'].count()
        k = dict_degree.get(port)
        p = round(1 - sum((k_is / k) ** 2), 4)
        list_p.append(p)
    df_nodes['B'] = port_extra_links
    df_nodes['Z'] = port_intra_links
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


def plot_sc_freq(data, year):
    list_datasets_custom = ['All', 'FC', 'International', 'FC International']
    data = data.reset_index()
    data['dataset'] = data['dataset'].astype('category')
    data['dataset'].cat.reorder_categories(list_datasets_custom, inplace=True)
    data.sort_values('dataset', inplace=True)
    data.index = range(0, len(data))
    file_list = ['dataset' + '\n' + '(all routes)', 'sub-dataset' + '\n' + '(FC routes)',
                 'sub-dataset' + '\n' + '(International routes)', 'sub-dataset' + '\n' + '(FC International routes)']

    n_rows = 1
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 5))
    width = 0.38
    x = range(1, 6)

    for i in range(n_cols):
        ax = axs[i]
        data_copy = data.loc[i, ['B', 'Z', 'P', 'K', 'BC']]
        ax.bar(x, data_copy, width, color='k', alpha=0.75)

        pltstyle.axes_style(ax)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(['B', 'Z', 'P', 'K', 'BC'], style='italic')
        ax.set_title("{}".format(file_list[i]), pad=10, fontsize=24)
        ax.set_ylim(0, iters + iters*0.15)
        if year == '2015':
            ax.set_xlabel('GLSN of 2015', fontsize=28)
        else:
            ax.set_xlabel('GLSN of 2017', fontsize=28)

    axs[0].set_ylabel(r'$Frequency$', fontsize=30)

    t = plt.suptitle('Note: The number of iterations of the experiment: in your test, {}; in the '
                     'manuscript, 1000.'.format(iters), color='red', style='italic',
                     fontsize=30, x=0.5, y=1.1)
    t.set_bbox(dict(facecolor='gray', alpha=0.3, edgecolor=None))

    plt.tight_layout(h_pad=2)

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 11')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if year == '2015':
            filename = 'Supplementary Fig. 14 Frequency...across datasets-(a) GLSN of 2015.png'
        else:
            filename = 'Supplementary Fig. 14 Frequency...across datasets-(b) GLSN of 2017.png'
        plt.savefig(save_path + '/' + filename, transparent=False, bbox_inches='tight')
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def plot_hist(tv_mean, tv_se, btv_mean, btv_se, year):
    tv_mean.fillna(0, inplace=True)
    btv_mean.fillna(0, inplace=True)
    datalist = [tv_mean, tv_se, btv_mean, btv_se]
    sorted_data = []
    list_datasets_custom = ['All', 'FC', 'International', 'FC International']
    for data in datalist:
        data = data.reset_index()
        data['dataset'] = data['dataset'].astype('category')
        data['dataset'].cat.reorder_categories(list_datasets_custom, inplace=True)
        data.sort_values('dataset', inplace=True)
        data = data.set_index(['dataset'])
        sorted_data.append(data)
    mean_list = [sorted_data[0], sorted_data[2]]
    se_list = [sorted_data[1], sorted_data[3]]

    n_rows = 1
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))

    xlabel_list = ['at the country level', 'at the country pair level']
    width = 0.2

    X = [1, 1.9, 2.8]
    a_list = [-1.5, -0.5, 0.5, 1.5]
    xtick_list = []
    for x in X:
        for a in a_list:
            xtick_list.append(x + a * width)

    for i in range(n_cols):
        ax = axes[i]
        data = mean_list[i]
        ses = se_list[i]
        rects1 = ax.bar(xtick_list[:4], data['# all connections'], 0.13,
                        yerr=ses['# all connections'],
                        color='k', capsize=8, error_kw={'lw': 3}, alpha=0.75, label='# all connections')
        rects2 = ax.bar(xtick_list[4:8], data['# SC connections'], 0.13,
                        yerr=ses['# SC connections'],
                        color='red', capsize=8, error_kw={'lw': 3}, alpha=0.75, label='# SC connections')
        rects3 = ax.bar(xtick_list[8:12], data['# NSC connections'], 0.13,
                        yerr=ses['# NSC connections'],
                        color='green', capsize=8, error_kw={'lw': 3}, alpha=0.75, label='# NSC connections')
        autolabel(ax, rects1)
        autolabel(ax, rects2)
        autolabel(ax, rects3)
        ax.set_xlabel(xlabel_list[i], labelpad=10, fontsize=28, weight='medium')

        plot_style(ax, xtick_list)
        if i == 0:
            ax.set_ylabel(r'pearson r', style='italic', fontsize=26, weight='medium')
    if year == '2015':
        axes[0].set_title('(a) GLSN of 2015', loc='left', fontsize=26, pad=10)
    else:
        axes[0].set_title('(b) GLSN of 2017', loc='left', fontsize=26, pad=10)

    plt.suptitle('Note: The number of iterations of the experiment:' + '\n' + 'in your test, {}; in the '
                     'manuscript, 1000.'.format(iters) + '\n' + '\n' + '\n' + 'Datasets:' + '\n'+ '$\circled1$ dataset (all routes)  $\circled3$ sub-dataset (International routes)' + '\n' +
              '$\circled2$ sub-dataset (FC routes)  $\circled4$ sub-dataset (FC International routes)', color='k',
                     fontsize=26, x=0, y=1.45, horizontalalignment='left')

    plt.tight_layout(h_pad=2)
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 11')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if year == '2015':
            filename = 'Supplementary Fig. 15 Pearson correlation coefficients...socio-economic...' \
                       'across datasets-(a) GLSN of 2015.png'
        else:
            filename = 'Supplementary Fig. 15 Pearson correlation coefficients...socio-economic...' \
                       'across datasets-(b) GLSN of 2017.png'
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
    pos_ticks = [(xtick_list[1] + xtick_list[2]) / 2, (xtick_list[5] + xtick_list[6]) / 2,
                 (xtick_list[9] + xtick_list[10]) / 2]

    ax.set_xticks(pos_ticks)
    ax.set_xticklabels(xlabels, fontsize=20, weight='medium')

    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    ax.tick_params(axis='y', direction='in', top=True, right=True, which='major', width=1.8, length=7,
                   pad=5, labelsize=20)
    ax.tick_params(axis='x', direction='in', bottom=False, top=False, right=False, which='major', width=1.8, length=7,
                   pad=5)


def autolabel(ax, rects, xpos='center'):
    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    labels = [r'$\circled1$', r'$\circled2$', r'$\circled3$', r'$\circled4$']
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.02*height, label, ha=ha[xpos],
                va='bottom', weight='bold', fontsize=24)


def sc_robustness():
    years = ['2015']
    datasets = ['All', 'FC', 'International', 'FC International']

    for year in years:
        df_freq_all = pd.DataFrame()
        df_pr_all = pd.DataFrame()
        df_country_pr_all = pd.DataFrame()
        df_country_pair_pr_all = pd.DataFrame()

        for dataset in datasets:
            nodes = pd.read_csv('../data/other data/Nodes_' + year + '_' + dataset + '_P.csv')
            edges = pd.read_csv('../data/GLSN data/Edges_' + year + '_' + dataset + '_P.csv', header=None)
            edges.columns = ['source', 'target']
            graph = nx.from_pandas_edgelist(edges, 'source', 'target', create_using=nx.Graph())
            dict_cap = dict(zip(nodes['id'], nodes['Capacity']))
            dict_country = dict(zip(nodes['id'], nodes['Country Code']))

            file_path = os.path.join(data_path, year, dataset)

            all_freq = []
            country_pr = []
            country_pair_pr = []
            for rt in range(1, iters+1):
                df_nodes = pd.read_csv(file_path + '/Nodes/' + str(rt) + '.csv', header=0)
                df_nodes['K'] = df_nodes['id'].apply(dict(nx.degree(graph)).get)
                df_nodes['Capacity'] = df_nodes['id'].apply(dict_cap.get)
                df_nodes['Country Code'] = df_nodes['id'].apply(dict_country.get)
                df_nodes = cal_bzp(df_nodes, edges)

                df_pr = cal_pr(df_nodes)
                df_pr['GLSN of'] = len(df_pr) * [year]
                df_pr['dataset'] = len(df_pr) * [dataset + str(' routes')]
                df_pr_all = pd.concat([df_pr_all, df_pr], axis=0, sort=False)

                param_freq, num_sc = find_sc(df_nodes, graph)
                all_freq.append(param_freq)
                if param_freq['B'] == 1:
                    n_sc = num_sc['B']
                    country_pr.append(country_level_pr(df_nodes, edges, year, n_sc))
                    country_pair_pr.append(country_pair_level_pr(df_nodes, edges, year, n_sc))
                else:
                    country_pr.append([np.nan, np.nan, np.nan])
                    country_pair_pr.append([np.nan, np.nan, np.nan])

            df_freq = pd.DataFrame(all_freq)
            df_freq['dataset'] = len(df_freq) * [dataset]
            df_freq_all = pd.concat([df_freq_all, df_freq], axis=0)

            df_country_pr = pd.DataFrame(country_pr)
            df_country_pr.columns = ['# all connections', '# SC connections', '# NSC connections']
            df_country_pr['dataset'] = len(df_country_pr) * [dataset]
            df_country_pr_all = pd.concat([df_country_pr_all, df_country_pr], axis=0)

            df_country_pair_pr = pd.DataFrame(country_pair_pr)
            df_country_pair_pr.columns = ['# all connections', '# SC connections', '# NSC connections']
            df_country_pair_pr['dataset'] = len(df_country_pair_pr) * [dataset]
            df_country_pair_pr_all = pd.concat([df_country_pair_pr_all, df_country_pair_pr], axis=0)

        param_freq = df_freq_all.groupby(['dataset'], as_index=True)['B', 'Z', 'P', 'K', 'BC'].apply(np.sum)
        plot_sc_freq(param_freq, year)

        if df_country_pr_all.empty or df_country_pair_pr_all.empty:
            print("No structural-core organization has found.")
        else:
            country_pr_mean = df_country_pr_all.groupby(['dataset'])[
                '# all connections', '# SC connections', '# NSC connections'].apply(np.mean).round(2)
            country_pair_pr_mean = df_country_pair_pr_all.groupby(['dataset'])[
                '# all connections', '# SC connections', '# NSC connections'].apply(np.mean).round(2)

            country_pr_se = df_country_pr_all.groupby(['dataset'])[
                                      '# all connections', '# SC connections', '# NSC connections'].sem()
            country_pair_pr_se = df_country_pair_pr_all.groupby(['dataset'])[
                                           '# all connections', '# SC connections', '# NSC connections'].sem()
            plot_hist(country_pr_mean, country_pr_se, country_pair_pr_mean, country_pair_pr_se, year)

        param_pr_mean = df_pr_all.groupby(['GLSN of', 'Network indicators', 'dataset'], as_index=False)['GLSN', 'C'].apply(np.mean).round(2)

        if SAVE_RESULT:
            save_path = os.path.join('output', 'Supplementary note 11')
            if os.path.exists(save_path):
                pass
            else:
                os.makedirs(save_path)
            filename = 'Supplementary table 5 Pearson correlation coefficients...across datasets-GLSN of ' + year + '.xlsx'
            param_pr_mean.to_excel(save_path + '/' + filename)
            print()
            print('The result file "{}" saved at: "{}"'.format(filename, save_path))
            print()


def startup():

    if os.path.exists(data_path):
        print('*********************************')
        print("Location in the manuscript text: ")
        print('Section titled "Supplementary note 11: Robustness of the structural-core organization of '
              'the GLSN across multiple datasets"')
        print('*********************************')
        print()
        print('***************************RUN TIME WARNING***************************')
        print('It needs 18 hours for 1000 iterations of the corresponding experiments.')
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
