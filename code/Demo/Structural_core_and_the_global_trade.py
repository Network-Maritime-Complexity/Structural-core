#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def cal_pearson_corr(conn, trade):
    p_son = stats.pearsonr(conn, trade)
    corr = round(p_son[0], 2)
    p_value = round(p_son[1], 4)
    return corr, p_value


def country_level_pr():
    def _random(num_sc_ports, df_edges):
        all_pearson = []
        for rt in range(1, iters+1):
            sc_ports = random.sample(Nodes['id'].values.tolist(), num_sc_ports)
            Nodes['SC Type'] = np.where(Nodes['id'].isin(sc_ports), 'SC', 'NSC')
            dict_sc_type = dict(zip(Nodes['id'], Nodes['SC Type']))

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
                corr, p_value = cal_pearson_corr(df_lsc[col], df_lsc['TV'])
                list_corr.append(corr)
            all_pearson.append(list_corr)
        df_pearson = pd.DataFrame(all_pearson)
        df_pearson.columns = cal_cols
        pearson_mean = df_pearson.mean(axis=0).round(2)
        pearson_se = (df_pearson.std(axis=0) / np.sqrt(len(df_pearson))).round(3)

        return pearson_mean['# SC connections'], pearson_se['# SC connections']

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
    num_outside_connections.columns = ['Country Code', '# connections']

    num_sc_ports = 37
    sc_ports = Nodes.sort_values('B', ascending=False)['id'][:num_sc_ports]
    Nodes['SC Type'] = np.where(Nodes['id'].isin(sc_ports), 'SC', 'NSC')
    dict_sc_type = dict(zip(Nodes['id'], Nodes['SC Type']))

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
        corr, p_value = cal_pearson_corr(df_lsc[col], df_lsc['TV'])
        list_corr.append(corr)

    country_pearson_mean, country_pearson_se = _random(num_sc_ports, df_edges)

    return list_corr, country_pearson_mean, country_pearson_se


def country_pair_level_pr():
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

    def _country_pair_random(num_sc_ports):
        list_pearson = []
        for i in range(1, iters+1):
            sc_ports = random.sample(Nodes['id'].values.tolist(), num_sc_ports)
            list_nsc_port = list(set(Nodes['id']).difference(set(sc_ports)))

            edges, edgelist = _process_edges(Edges, sc_ports, list_nsc_port)

            num_connections = edges.groupby(['Country Code_source', 'Country Code_target'], as_index=False)[
                'source'].count()
            num_connections.rename(columns={'source': '# connections'}, inplace=True)

            for data in edgelist:
                lfc = data.groupby(['Country Code_source', 'Country Code_target'], as_index=False)['source'].count()
                num_connections = pd.merge(num_connections, lfc, on=['Country Code_source', 'Country Code_target'],
                                           how='outer')
            num_connections.columns = ['Country Code_source', 'Country Code_target', '# connections',
                                       '# SC connections', '# NSC connections']

            df_res = pd.merge(num_connections, df_btv, on=['Country Code_source', 'Country Code_target'])
            df_res.fillna(0, inplace=True)

            connection_cols = [col for col in df_res.columns if 'connections' in col]
            list_corr = []
            for col in connection_cols:
                corr, p_value = cal_pearson_corr(df_res[col], df_res['BTV'])
                list_corr.append(corr)
            list_pearson.append(list_corr)
        df_pearson = pd.DataFrame(list_pearson)
        df_pearson.columns = connection_cols
        pearson_mean = df_pearson.mean(axis=0).round(2)
        pearson_se = (df_pearson.std(axis=0) / np.sqrt(len(df_pearson))).round(3)

        return pearson_mean['# SC connections'], pearson_se['# SC connections']

    df_btv = pd.read_csv('../data/other data/BTV_' + YEAR + '.csv')

    num_sc_ports = 37
    list_core_ports = Nodes.sort_values('B', ascending=False)['id'][:num_sc_ports].values.tolist()
    list_none_core_port = list(set(Nodes['id']).difference(set(list_core_ports)))
    edges, edgelist = _process_edges(Edges, list_core_ports, list_none_core_port)

    num_connections = edges.groupby(['Country Code_source', 'Country Code_target'], as_index=False)['source'].count()
    num_connections.rename(columns={'source': '# connections'}, inplace=True)
    for data in edgelist:
        lfc = data.groupby(['Country Code_source', 'Country Code_target'], as_index=False)['source'].count()
        num_connections = pd.merge(num_connections, lfc, on=['Country Code_source', 'Country Code_target'], how='outer')
    num_connections.columns = ['Country Code_source', 'Country Code_target', '# connections',
                               '# SC connections', '# NSC connections']

    df_res = pd.merge(num_connections, df_btv, on=['Country Code_source', 'Country Code_target'])
    df_res.fillna(0, inplace=True)

    connection_cols = [col for col in df_res.columns if 'connections' in col]
    list_corr = []
    for col in connection_cols:
        corr, p_value = cal_pearson_corr(df_res[col], df_res['BTV'])
        list_corr.append(corr)

    country_pair_pearson_mean, country_pair_pearson_se = _country_pair_random(num_sc_ports)

    return list_corr, country_pair_pearson_mean, country_pair_pearson_se


def cal_tv():
    df_world = 32271530872712  # https://comtrade.un.org/
    df_tv = pd.read_csv('../data/other data/TV_' + YEAR + '.csv')
    df_country = Nodes[['Country Code']].drop_duplicates()
    df_tv = pd.merge(df_tv, df_country, on='Country Code')
    pr = df_tv['TV'].sum() / df_world * 100
    print('The in-text result:')
    print()
    print('"Note that maritime countries altogether account for about {:.0f}% of international trade in terms of '
          'value, according to the import and export data of world countries released dby the UN '
          'Comtrade database."'.format(pr))
    print()


def plot_hist(y_c, tv_mean, tv_se, y_cp, btv_mean, btv_se):
    fig = plt.figure(figsize=(14.5, 6))
    ax1 = fig.add_subplot(121)

    x = [2, 4, 6, 8]
    y_c.append(0.0)
    ax1.bar(x, y_c, color=['black', 'red', 'green', 'white'], alpha=0.8, width=0.6)

    ax1.bar(8, tv_mean, color='gray', alpha=0.4, width=0.6)
    ax1.errorbar(8, tv_mean, yerr=tv_se, marker='', fmt='k', lw=2, capsize=12)
    label_font = {'family': 'Arial', 'weight': 'medium', 'size': 24}
    ax1.set_xlabel('at the country level', fontdict=label_font, labelpad=10)
    plot_style(x, ax1)

    ax1.set_ylabel('Pearson r', style='italic', fontsize=28, labelpad=10)
    ax1.set_ylim([0, 1])
    ax1.set_title('(a)', fontsize=26, style='italic', pad=10, loc='left')

    y_cp.append(0.0)
    ax2 = fig.add_subplot(122)
    ax2.bar(x, y_cp, color=['black', 'red', 'green', 'white'], alpha=0.8, width=0.6)

    ax2.bar(8, btv_mean, color='gray', alpha=0.4, width=0.6)
    ax2.errorbar(8, btv_mean, yerr=btv_se, marker='', fmt='k', lw=2, capsize=12)
    ax2.set_ylim([0, 1])
    ax2.set_title('(b)', fontsize=26, style='italic', pad=10, loc='left')
    label_font = {'family': 'Arial', 'weight': 'medium', 'size': 24}
    ax2.set_xlabel('at the country pair level', fontdict=label_font, labelpad=10)
    plot_style(x, ax2)

    plt.tight_layout(w_pad=3)

    save_path = os.path.join('output', 'Structural_core_and_the_global_trade')
    if SAVE_RESULT:
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = "Fig. 11 Correlation between the GLSN topological indicators and socio-economic indicators " \
                   "of world maritime countries.png"
        fig.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def plot_style(x, ax):
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=1.5, length=5.5, pad=7)
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor', width=1.5, length=4, pad=7)
    ax.set_xlim([ax.get_xlim()[0] - 0.3, ax.get_xlim()[1] + 0.3])
    plt.xticks(fontproperties='Arial', fontsize=18, weight='medium')
    plt.yticks(fontproperties='Arial', fontsize=22, weight='medium')
    xlabels = ['# all\n connections', '# SC\n connections', '# NSC\n connections',
               '# SC connections\n (random)']
    plt.xticks(x, xlabels)
    ax.set_ylim([0, 1])
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ymajorlocator = MultipleLocator(0.2)
    yminorlocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(ymajorlocator)
    ax.yaxis.set_minor_locator(yminorlocator)


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Section titled "Structural core and the global trade"')
    print('*********************************')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()
    # cal_tv()
    country_corr, country_pearson_mean, country_pearson_se = country_level_pr()
    cp_mean, country_pair_pearson_mean, country_pair_pearson_se = country_pair_level_pr()
    plot_hist(country_corr, country_pearson_mean, country_pearson_se, cp_mean, country_pair_pearson_mean, country_pair_pearson_se)
