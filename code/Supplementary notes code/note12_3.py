#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


data_path = os.path.join('../', 'data', 'downloaded data files', 'note12_3')


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


def sc_robustness():
    nodes_15 = pd.read_csv('../data/other data/Nodes_2015_All_L.csv')
    edges_15 = pd.read_csv('../data/GLSN data/Edges_2015_All_L.csv', header=None)
    edges_15.columns = ['source', 'target']
    dict_cap_15 = dict(zip(nodes_15['id'], nodes_15['Capacity']))
    dict_country_15 = dict(zip(nodes_15['id'], nodes_15['Country Code']))

    df_pr_all_15 = pd.DataFrame()
    for rt in range(1, iters+1):
        df_nodes_15 = pd.read_csv(data_path + '/2015/' + str(rt) + '.csv')
        df_nodes_15['Capacity'] = df_nodes_15['id'].apply(dict_cap_15.get)
        df_nodes_15['Country Code'] = df_nodes_15['id'].apply(dict_country_15.get)
        df_nodes_15 = cal_bzp(df_nodes_15, edges_15)
        df_pr_15 = cal_pr(df_nodes_15)
        df_pr_all_15 = pd.concat([df_pr_all_15, df_pr_15], axis=0, sort=False)

    param_pr_mean_15 = df_pr_all_15.groupby('Network indicators').apply(np.mean).round(2)

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 12_3')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Supplementary table 10 Pearson correlation coefficients...GLSNs of 2015.xlsx'
        param_pr_mean_15.to_excel(save_path + '/' + filename, index=True)
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()


def startup():
    if os.path.exists(data_path):
        print('*********************************')
        print("Location in the manuscript text: ")
        print('Subsection titled "(3) The constraints of the economy of liner shipping network"')
        print('Section titled "Supplementary note 12: Influence of the constraints on the structural core of the GLSN"')
        print('*********************************')
        print()
        print('***************************RUN TIME WARNING***************************')
        print('It needs 40 minutes for 1000 iterations of the corresponding experiments.')
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
