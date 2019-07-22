#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


data_path = os.path.join('../data', 'Other data', '1000 Community divisions')


def cal_pr(data):
    def _cal_corr(x, y):
        corr = round(stats.pearsonr(x, y)[0], 2)
        pval = round(stats.pearsonr(x, y)[1], 3)
        if pval < 0.001:
            pval = '**'
        elif 0.001 <= pval < 0.01:
            pval = '*'
        else:
            pval = np.nan

        return corr, pval

    params = ['B', 'Z', 'P', 'K', 'BC', 'phi', 'rho_C', 'rho_CM']
    df_nodes = data.apply(pd.to_numeric, errors='coerce')
    list_data = [df_nodes]
    modules = sorted(pd.unique(df_nodes['Community']))
    for module in modules:
        list_data.append(df_nodes[df_nodes['Community'] == module])

    list_pearsonr_all = []
    for param in params:
        list_pearsonr = []
        for data in list_data:
            data_dropna = data.dropna(subset=[param])
            if data_dropna[param].std() == np.nan or data_dropna.empty:
                pearsonr = '--'
            else:
                pearsonr = _cal_corr(data_dropna[param], data_dropna['Capacity'])
            list_pearsonr.append(pearsonr)
        list_pearsonr_all.append(list_pearsonr)
    df_pearsonr = pd.DataFrame(list_pearsonr_all)
    df_pearsonr.index = params
    new_cols = ['C' + str(i) for i in range(1, len(df_pearsonr.columns))]
    new_cols.insert(0, 'GLSN')
    df_pearsonr.columns = new_cols
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Structural_embeddedness_and_economic_performance_of_world_ports')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Table 1 Pearson correlation coefficients...port capacity.csv'
        df_pearsonr.to_csv(save_path + '/' + filename, index=True, index_label='Network indicators')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()


def cal_zscore(data):
    means = np.mean(data)
    std = np.std(data, ddof=1)
    data = (data - means) / std

    return data


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
    params = ['B']
    num_modules = len(pd.unique(df_nodes['Community']))
    dict_freq = {'B': 0}
    num_sc_nodes = {'B': 0}
    coor_x = {'B': 0}

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


def jaccard_similarity_coefficient(df_nodes, sc):
    def _jaccard(a, b):
        c = a.intersection(b)
        jsc = round(float(len(c)) / (len(a) + len(b) - len(c)), 6)
        return jsc

    nodedata = Nodes.sort_values('K', ascending=False)
    degrees = pd.unique(nodedata['K'])[1:]
    dict_jsc = {}
    for degree in degrees:
        degree_ports = Nodes[Nodes['K'] > degree]['id'].values.tolist()
        sc_ports = df_nodes[df_nodes['B'] >= sc]['id'].values.tolist()
        jsc = _jaccard(set(sc_ports), set(degree_ports))
        dict_jsc[degree] = jsc
    return dict_jsc


def sc_robustness():
    list_js = []
    for rt in range(1, iters+1):
        nodes = pd.read_csv(data_path + '/' + str(rt) + '.csv')
        dict_module = dict(zip(nodes['id'], nodes['Community']))
        Nodes['Community'] = Nodes['id'].apply(dict_module.get)
        df_nodes = cal_bzp(Nodes)
        param_freq, num_sc, coor_x = find_sc(df_nodes)
        if param_freq['B'] == 1:
            c_x = coor_x['B']
            js = jaccard_similarity_coefficient(df_nodes, c_x)
            list_js.append(js)

    df_js = pd.DataFrame(list_js)
    plot_js_error(df_js)


def plot_js_error(data):
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111)
    means = data.mean()
    ax.plot(means.index, means, 'k-')

    ax.text(55, 0.64, r'$K=${}'.format(means.idxmax()), fontsize=16, color='k', alpha=0.95)
    ax.tick_params(axis='y', direction='in', right=False, which='major',
                   width=1.8, color='k', length=7, pad=5, labelsize=16, labelcolor='k')
    ax.tick_params(axis='x', direction='in', top=True, which='major',
                   width=1.8, length=7, pad=5, labelsize=16)

    ax.set_xlabel(r'$K$', fontsize=18)
    ax.set_ylabel(r'$Jaccard\ similarity$', fontsize=16, labelpad=15, color='k')

    ax.set_title('(a)', style='italic', fontsize=18, pad=10, loc='left')

    ax2 = ax.twinx()
    df_nodes = Nodes.apply(pd.to_numeric, errors='coerce')
    df_nodes.sort_values('K', inplace=True)
    ax2.plot(df_nodes['K'], df_nodes['rho_CM'], 'r-')
    ax2.set_ylabel(r'$\rho_{\rmCM}$', fontsize=18, labelpad=10, color='r')

    ix_max = df_nodes['rho_CM'].idxmax()
    k_max = df_nodes.loc[ix_max, 'K']
    ax2.text(187, 0.42, r'$K={}$'.format(k_max), fontsize=16, color='r', alpha=0.9)
    ax2.annotate(s='', xy=(173, ax2.get_ylim()[0]),
                 xytext=(173, df_nodes['rho_CM'].max()), arrowprops=dict(arrowstyle='-', lw=2, color='r', ls='--', alpha=0.45))

    ax2.set_ylim(ax2.get_ylim()[0], 1)
    ax.set_ylim(ax2.get_ylim()[0], 1)

    ax.annotate(s='', xy=(135, ax2.get_ylim()[0]),
                xytext=(135, max(means)), arrowprops=dict(arrowstyle='-', lw=2, color='k', ls='--', alpha=0.6))
    ax2.tick_params(axis='y', direction='in', right=True, which='major',
                    width=1.8, color='r', length=7, pad=5, labelsize=16, labelcolor='r')
    ax2.tick_params(axis='x', direction='in', top=False, which='major',
                    width=1.8, length=7, pad=5, labelsize=16)
    plt.tight_layout()
    ax2.spines['right'].set_color('r')

    plot_style(ax)
    plot_style(ax2)
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Structural_embeddedness_and_economic_performance_of_world_ports')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Fig. 10 Overlap between the rich club and the structural core of the GLSN (a).png'
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def plot_style(ax):
    xmajorlocator = MultipleLocator(50)
    ax.xaxis.set_major_locator(xmajorlocator)
    ymajorlocator = MultipleLocator(0.2)
    ax.yaxis.set_major_locator(ymajorlocator)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Section titled "Structural embeddedness and economic performance of world ports"')
    print('*********************************')
    print()
    print()
    print('***************************RUN TIME WARNING***************************')
    print('It needs 7 hours for 1000 iterations of the corresponding experiments.')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()

    cal_pr(Nodes)
    sc_robustness()
