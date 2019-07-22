#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


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
        all_dis = pd.read_csv('../data/other data/Distance_SR_GC_' + YEAR + '.csv')
        dict_dis = dict(zip(all_dis['Edge'], all_dis[self.WEIGHT]))

        df_edges = Edges.copy()
        df_edges['Edge'] = df_edges['source'].astype(str) + str('--') + df_edges['target'].astype(str)
        df_edges[self.WEIGHT] = df_edges['Edge'].apply(dict_dis.get)
        weighted_G = nx.from_pandas_edgelist(df_edges, 'source', 'target', edge_attr=self.WEIGHT, create_using=nx.Graph())

        # ideal cost
        cost_all = all_dis[self.WEIGHT].sum() / 2
        cost = round(df_edges[self.WEIGHT].sum() / cost_all * 100, 1)

        # E(G)
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
        print('"Excitingly, we found the underlying structure of the GLSN remarkably economic. Indeed, it has '
              'global and local efficiency arriving to {:.1f}% and {:.1f}%, respectively, '
              'but holds an extremely small network cost that only accounts for {:.1f}%."'.format(GE, LE, cost))
        print()


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Subsection titled "Economic small-world-ness"')
    print('Section titled "Basic topological properties"')
    print('*********************************')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()
    instance = EffiCost()
    instance.effi_cost()
