#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def hist_style(ax):
    ax.tick_params(axis='both', direction='in', which='major', width=2.5, length=8, pad=8, labelsize=22)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1*height, '%d' % int(height),
                 ha='center', va='bottom', fontsize=22)


def plot_hist(param, data):
    ms = pd.DataFrame(pd.Series(np.arange(1, 8, 1), name='Module'))
    data = pd.merge(data, ms, on='Module', how='right')
    data.fillna(0, inplace=True)
    data.sort_values('Module', inplace=True)
    x = data['Module']
    y = data['# ports']

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.family': 'Arial'})
    colors = ['#660096', '#FF00C5', '#D98500', '#0069D6', '#000000', '#E84C00', '#3FA623']
    rect = ax.bar(x, y, color=colors, width=0.7)

    ax.set_xlabel(r'Modules', fontsize=24, labelpad=5)
    if param == 'B':
        ax.set_ylabel(r'# ports of $B\geq$2.43', fontsize=20, labelpad=5)
    elif param == 'Z':
        ax.set_ylabel(r'# ports of $Z\geq$3.25', fontsize=20, labelpad=5)
    elif param == 'P':
        ax.set_ylabel(r'# ports of $P\geq$0.78', fontsize=20, labelpad=5)

    autolabel(rect)
    plt.xticks([1, 2, 3, 4, 5, 6, 7])
    hist_style(ax)

    if param == 'B':
        ax.set_ylim([1, max(y) + 1])
        plt.yticks(np.arange(0, max(y) + 2, 2))
    elif param == 'Z':
        ax.set_ylim([1, max(y)])
        plt.yticks(np.arange(0, max(y) + 1, 1))
    elif param == 'P':
        ax.set_ylim([1, max(y)])
        plt.yticks(np.arange(0, max(y) + 1, 2))

    plt.tight_layout()

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Gateway_hub_structural_core')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if param == 'B':
            filename = 'Fig. 6 Structural core detection of the GLSN (b).png'
        elif param == 'Z':
            filename = 'Fig. 6 Structural core detection of the GLSN (d).png'
        else:
            filename = 'Fig. 6 Structural core detection of the GLSN (f).png'
        fig.savefig(save_path + '/' + filename, bbox_inches='tight')
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    else:
        plt.show()
    plt.close('all')

    plt.show()


def defining_sc(pp):
    param = pp[0]
    param_threshold = pp[1]

    list_density = []
    list_nodes = []
    list_edges = []
    list_value = sorted(list(set(Nodes[param].values)))
    list_value = list_value[:-1]

    df_edges = Edges.copy()
    for value in list_value:
        index_param = Nodes[param] >= value
        nodelist = Nodes["id"][index_param].values.tolist()
        sub_edges = df_edges[(df_edges['source'].isin(nodelist)) & (df_edges['target'].isin(nodelist))]
        H = G.subgraph(nodelist)
        list_density.append(round(nx.density(H), 4))

        list_edges.append(len(sub_edges))
        list_nodes.append(len(nodelist))
    density = pd.DataFrame()
    density[param] = list_value
    density['num_nodes'] = list_nodes
    density['num_edges'] = list_edges
    density['Density'] = list_density
    density.sort_values(param, ascending=False, inplace=True)
    ix = (density[param] >= param_threshold) & (density['Density'] >= 0.8)

    if density[ix].empty:
        coor_x = np.nan
        coor_y = np.nan
    else:
        coor_x = density.loc[ix, param].values.tolist()[-1]
        coor_y = density.loc[ix, 'Density'].values.tolist()[-1]

        ix_sc_nodes = Nodes[param] >= coor_x
        num_ports = Nodes[ix_sc_nodes].groupby('Community', as_index=False)['id'].count()
        num_ports.columns = ['Module', '# ports']
        plot_hist(param, num_ports)

    xdata = density[param]
    ydata = density['Density']
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)
    ax.plot(xdata, ydata, 'bo', ms=5.5)
    ax.set_xlim([np.floor(xdata.min()), np.ceil(xdata.max())])
    if param == 'B' or param == 'Z':
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))

    pltstyle.plot_glsn_basic(ax, coor_x, coor_y)
    ax.set_xlabel(r'${}$'.format(param), fontsize=26)
    ax.set_ylabel(r'Density among ports of ${}_i≥{}$'.format(param, param), fontsize=20)
    plt.tight_layout()

    if SAVE_RESULT:
        save_path = os.path.join('output', 'Gateway_hub_structural_core')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if param == 'B':
            filename = 'Fig. 6 Structural core detection of the GLSN (a).png'
        elif param == 'Z':
            filename = 'Fig. 6 Structural core detection of the GLSN (c).png'
        else:
            filename = 'Fig. 6 Structural core detection of the GLSN (e).png'
        fig.savefig(save_path + '/' + filename, bbox_inches='tight', pad_inches=0.08)
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


def sc_topological_centrality(num_sc):
    n = 0
    N = 0
    edge = 0

    list_core_ports = Nodes.sort_values(['B'], ascending=False)['id'][:num_sc].values
    set_central_ports = set(list_core_ports)
    list_non_core_port = list(set(Nodes['id']).difference(set_central_ports))

    core_edges = Edges[(Edges['source'].isin(list_core_ports)) & (Edges['target'].isin(list_core_ports))]
    list_source = core_edges['source'].values.tolist()
    list_target = core_edges['target'].values.tolist()
    tuple_edges = list((zip(list_source, list_target)))
    tuple_edges_re = list((zip(list_target, list_source)))

    for s, port_s in enumerate(list_non_core_port[:-1]):
        for port_t in list_non_core_port[s + 1:]:
            path = nx.all_shortest_paths(G, source=port_s, target=port_t)
            result = compare(path, tuple_edges, tuple_edges_re, set_central_ports)
            N += result[0]
            n += result[1]
            edge += result[2]

    pr = n / N * 100
    pr_edge = edge / N * 100
    print('The in-text result:')
    print()
    print('"We report that, the percentage of the number of shortest paths that pass through at least one of the core '
          'ports, normalized to the number of all shortest paths among the non-core ports, reaches as high as {:.2f}%; '
          'and the percentage of the number of shortest paths that pass through at least one core '
          'connection is {:.2f}%."'.format(pr, pr_edge))
    print()


class LFCConnection:
    def __init__(self, num_sc):
        self.NUM_SC = num_sc
        self.list_core_ports = Nodes.sort_values('B', ascending=False)['id'][:self.NUM_SC].values
        self.list_non_core_port = list(set(Nodes['id']).difference(set(self.list_core_ports)))

    def core_connections_importance(self):
        df_dis = pd.read_csv('../data/Other data/Distance_SR_GC_' + YEAR + '.csv')
        dict_dis = dict(zip(df_dis['Edge'].astype(str), df_dis[dis_col]))
        df_edges = Edges.copy()
        df_edges['Edge'] = df_edges['source'].astype(str) + '--' + df_edges['target'].astype(str)
        df_edges['Distance'] = df_edges['Edge'].apply(dict_dis.get)

        # core connections
        core_ix = df_edges['source'].isin(self.list_core_ports) & df_edges['target'].isin(self.list_core_ports)
        df_edges.loc[core_ix, 'LFC'] = 'core'

        # local connections
        local_ix = df_edges['source'].isin(self.list_non_core_port) & df_edges['target'].isin(self.list_non_core_port)
        df_edges.loc[local_ix, 'LFC'] = 'local'

        # feeder connections
        feeder_ix = (~core_ix) & (~local_ix)
        df_edges.loc[feeder_ix, 'LFC'] = 'feeder'

        link_percentage = round(df_edges.groupby('LFC')['source'].count() / len(df_edges) * 100, 1)
        len_percentage = round(df_edges.groupby('LFC')['Distance'].sum() / df_edges['Distance'].sum() * 100, 1)

        df_res_b = pd.concat([link_percentage, len_percentage], axis=1)
        df_res_b['types of connections'] = df_res_b.index
        df_res_b.rename(columns={'source': 'Link percentage', 'Distance': 'Length percentage'}, inplace=True)
        df_res_b['Length percentage / Link percentage'] = round(df_res_b['Length percentage'] / df_res_b['Link percentage'], 1)
        df_res_b = df_res_b[['types of connections', 'Link percentage', 'Length percentage', 'Length percentage / Link percentage']]

        dis_mean = df_edges.groupby('LFC')['Distance'].mean()
        dis_std = df_edges.groupby('LFC')['Distance'].std()
        all_edges_mean = df_edges['Distance'].mean()
        all_edges_std = df_edges['Distance'].std()

        print('The in-text result:')
        print()
        print('"First, core connections tend to be longer than feeder and local connections (Fig. 8b). '
              'Measured by real nautical distance, the average length of core '
              'connections (average = {:.0f} km, SD = {:.0f} km) is {:.1f} times of the average over all inter-port '
              'connections (average = {:.0f} km, SD = {:.0f} km); feeder connections (average= {:.0f} km, SD = {:.0f} km)'
              ', {:.1f} times; local connections (average = {:.0f} km, SD= {:.0f} km), {:.1f} times."'.format(
            dis_mean['core'], dis_std['core'], dis_mean['core'] / all_edges_mean, all_edges_mean, all_edges_std,
            dis_mean['feeder'], dis_std['feeder'], dis_mean['feeder'] / all_edges_mean, dis_mean['local'],
            dis_std['local'], dis_mean['local'] / all_edges_mean))
        print()

        if SAVE_RESULT:
            save_path = os.path.join('output', 'Gateway_hub_structural_core')
            if os.path.exists(save_path):
                pass
            else:
                os.makedirs(save_path)
            filename = 'Fig. 8 Statistics of the core, feeder and local connections (b).xlsx'
            df_res_b.to_excel(save_path + '/' + filename, index=False)
            print('The result file "{}" saved at: "{}"'.format(filename, save_path))
            print()

        return df_edges

    def write_glsn_non_core_sp(self):
        list_non_core_sp = []
        for s, port_s in enumerate(self.list_non_core_port[:-1]):
            for port_t in self.list_non_core_port[s + 1:]:
                path = nx.all_shortest_paths(G, source=port_s, target=port_t)
                for p in path:
                    list_non_core_sp.append(p)
        non_core_sp = pd.Series(list_non_core_sp)
        save_path = os.path.join('output', 'process')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        non_core_sp.to_csv(save_path + '/nsc_sp.csv', header=False, index=False)

        core_edges = Edges[(Edges['source'].isin(self.list_core_ports)) & (Edges['target'].isin(self.list_core_ports))]
        list_source = core_edges['source'].values.tolist()
        list_target = core_edges['target'].values.tolist()
        tuple_edges = list((zip(list_source, list_target)))
        tuple_edges_re = list((zip(list_target, list_source)))
        central_ports = set(self.list_core_ports)
        list_sp = []
        for s, port_s in enumerate(self.list_non_core_port[:-1]):
            for port_t in self.list_non_core_port[s + 1:]:
                shortest_paths = nx.all_shortest_paths(G, source=port_s, target=port_t)

                for path in shortest_paths:
                    union_nodes = central_ports.intersection(path)
                    num_union = len(union_nodes)
                    if num_union >= 2:
                        union_indexes = [path.index(each) for each in union_nodes]
                        union_indexes = sorted(union_indexes)
                        for i in range(len(union_indexes) - 1):
                            index_diff = union_indexes[i + 1] - union_indexes[i]
                            if index_diff < 2:
                                edge = (path[union_indexes[i]], path[union_indexes[i + 1]])
                                if edge in tuple_edges or edge in tuple_edges_re:
                                    list_sp.append(path)
                                    break

        sp = pd.Series(list_sp)
        save_path = os.path.join('output', 'process')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        sp.to_csv(save_path + "/nsc_sp_travel_through_sc.csv", header=False, index=False)

    @staticmethod
    def pr_lfc(edges):
        def process_edge_data(edges):
            df_edge_copy = edges.copy()
            s_cols = [col for col in edges.columns if 'source' in col]
            t_cols = [col for col in edges.columns if 'target' in col]
            tmp = edges[s_cols].copy()
            df_edge_copy[s_cols] = df_edge_copy[t_cols]
            df_edge_copy[t_cols] = tmp
            df_edge = pd.concat([edges, df_edge_copy], axis=0)
            df_edge.index = range(0, len(df_edge))

            edge_merged = []
            for ind in df_edge.index:
                edge = (df_edge.loc[ind, 'source'], df_edge.loc[ind, 'target'])
                edge_merged.append(edge)

            df_edge['Edge'] = edge_merged
            return df_edge

        def process_sp_data(spdata, path_sp_split):
            spdata.columns = ['SP']

            pair_cols = ['E' + str(i) for i in range(10)]

            data_split_sp = []
            for ind in spdata.index:
                sp = eval(spdata.ix[ind, 'SP'])
                sp_pair = list(zip(sp, np.roll(sp, -1)))[:-1]
                pair_series = dict(zip(*[pair_cols, sp_pair]))
                data_split_sp.append(pair_series)

            data_split_sp = pd.DataFrame(data_split_sp)
            data_split_sp.to_csv(path_sp_split, index=False)
            del data_split_sp

        def get_propertys_(df_edge, data_splited):
            edge2distance = dict(zip(df_edge['Edge'].astype(str), df_edge['Distance']))
            edge2property = dict(zip(df_edge['Edge'].astype(str), df_edge['LFC']))
            col_origin = data_splited.columns
            property_cols = ['LFC' + str(i) for i in range(10)]
            distance_cols = ['Distance' + str(i) for i in range(10)]
            for i, col_i in enumerate(col_origin):
                data_splited[property_cols[i]] = data_splited[col_i].apply(edge2property.get)
                data_splited[distance_cols[i]] = data_splited[col_i].apply(edge2distance.get)
            data_splited.drop(columns=col_origin, inplace=True)
            return data_splited

        PATH_SPS = ['output/process/nsc_sp.csv', 'output/process/nsc_sp_travel_through_sc.csv']
        PATH_SP_SPLIT = ['output/process/nsc_sp_splited_st.csv',
                         'output/process/nsc_sp_travel_through_sc_splited_st.csv']

        df_res_c = pd.DataFrame()
        for ix, PATH_SP in enumerate(PATH_SPS):
            df_edge = process_edge_data(edges)
            spdata = pd.read_csv(PATH_SP, header=None)
            process_sp_data(spdata, PATH_SP_SPLIT[ix])

            data_split_sp = pd.read_csv(PATH_SP_SPLIT[ix], low_memory=False)
            data_sp = get_propertys_(df_edge, data_split_sp)

            d_cols = [col for col in data_sp.columns if 'Distance' in col]
            data_sp['Distance_sum'] = data_sp[d_cols].sum(axis=1)

            sp_data_ori = pd.read_csv(PATH_SP, header=None)
            sp_data_ori.columns = ['SP']
            data_all = pd.concat([sp_data_ori, data_sp], axis=1)

            distance_col = [col for col in data_all.columns if 'Distance' in col and col != 'Distance_sum']
            property_col = [col for col in data_all.columns if 'LFC' in col]

            df_property_dis = pd.DataFrame()
            for d_col, p_col in zip(distance_col, property_col):
                lfc_dis = data_all.groupby(p_col, as_index=True)[d_col].sum()
                df_property_dis = pd.concat([df_property_dis, lfc_dis], axis=0)

            df_property_dis.columns = ['Distance']
            df_property_dis['LFC'] = df_property_dis.index

            df_res = df_property_dis.groupby('LFC', as_index=False)['Distance'].sum()
            if PATH_SP == 'output/process/nsc_sp.csv':
                df_res['shipping distance of all paths between non-core ports'] = round(df_res['Distance'] / (data_all['Distance_sum'].sum()) * 100, 2)
            else:
                df_res['shipping distance of paths between non-core ports traveling through structural-core'] = round(
                    df_res['Distance'] / (data_all['Distance_sum'].sum()) * 100, 1)
            df_res.drop(columns=['LFC', 'Distance'], inplace=True)
            df_res_c = pd.concat([df_res_c, df_res], axis=1)

        df_res_c.index = range(len(df_res_c))
        df_res_c['types of connections'] = ['core', 'feeder', 'local']
        ix_1 = df_res_c['types of connections'] == 'core'
        ix_2 = df_res_c['types of connections'] == 'feeder'
        ix_3 = df_res_c['types of connections'] == 'local'
        df_res_c.loc[ix_1, 'Distance percentage / Link percentage (Left)'] = round(df_res_c.loc[ix_1, 'shipping distance of all paths between non-core ports'] / 3.2, 1)
        df_res_c.loc[ix_2, 'Distance percentage / Link percentage (Left)'] = round(df_res_c.loc[ix_2, 'shipping distance of all paths between non-core ports'] / 32.7, 1)
        df_res_c.loc[ix_3, 'Distance percentage / Link percentage (Left)'] = round(df_res_c.loc[ix_3, 'shipping distance of all paths between non-core ports'] / 64.1, 1)

        df_res_c.loc[ix_1, 'Distance percentage / Link percentage (Right)'] = round(df_res_c.loc[
                                                                       ix_1, 'shipping distance of paths between non-core ports traveling through structural-core'] / 3.2, 1)
        df_res_c.loc[ix_2, 'Distance percentage / Link percentage (Right)'] = round(df_res_c.loc[
                                                                       ix_2, 'shipping distance of paths between non-core ports traveling through structural-core'] / 32.7, 1)
        df_res_c.loc[ix_3, 'Distance percentage / Link percentage (Right)'] = round(df_res_c.loc[
                                                                       ix_3, 'shipping distance of paths between non-core ports traveling through structural-core'] / 64.1, 1)

        df_res_c = df_res_c[['types of connections', 'shipping distance of all paths between non-core ports',
                             'Distance percentage / Link percentage (Left)',
                             'shipping distance of paths between non-core ports traveling through structural-core',
                             'Distance percentage / Link percentage (Right)']]
        if SAVE_RESULT:
            save_path = os.path.join('output', 'Gateway_hub_structural_core')
            if os.path.exists(save_path):
                pass
            else:
                os.makedirs(save_path)
            filename = 'Fig. 8 Statistics of the core, feeder and local connections (c).xlsx'
            df_res_c.to_excel(save_path + '/' + filename, index=False)
            print('The result file "{}" saved at: "{}"'.format(filename, save_path))
            print()

        del_path = 'output/process'
        if os.path.exists(del_path):
            shutil.rmtree(del_path)
        else:
            pass


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Subsection titled "Gateway-hub structural core"')
    print('Section titled "Results"')
    print('*********************************')
    print()
    print('***************************RUN TIME WARNING***************************')
    print('It needs 3 hours for corresponding experiments.')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    num_sc_nodes = 37

    params = [['B', 1.5], ['Z', 1.5], ['P', 0.7]]
    for pp in params:
        defining_sc(pp)

    sc_topological_centrality(num_sc_nodes)

    instance = LFCConnection(num_sc_nodes)
    df_edges = instance.core_connections_importance()
    instance.write_glsn_non_core_sp()
    instance.pr_lfc(df_edges)
