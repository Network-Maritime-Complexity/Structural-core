#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


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


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Subsection titled "Topological centrality of the structural core"')
    print('Section titled "Gateway-hub-based structural core"')
    print('*********************************')
    print()
    print('***************************RUN TIME WARNING***************************')
    print('It needs 45 minutes for corresponding experiments.')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()
    num_sc_nodes = 37
    sc_topological_centrality(num_sc_nodes)
