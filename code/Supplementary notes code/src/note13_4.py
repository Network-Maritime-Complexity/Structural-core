
from configure import *

sw_path = os.path.join('../data', 'downloaded data files', 'note13')


def cal_smallworld_c_spl(iterations):
    num_nodes = len(Nodes)
    density = 2 * len(Edges) / (num_nodes * (num_nodes - 1))
    ave_k = round(density * (num_nodes - 1))

    p_list = np.linspace(0.1, 1.0, 10)
    g0 = nx.connected_watts_strogatz_graph(n=num_nodes, k=ave_k, p=0.0, tries=100, seed=None)
    c0 = round(nx.average_clustering(g0), 5)
    spl0 = round(nx.average_shortest_path_length(g0), 3)

    ave_clustering = []
    list_spl = []
    ps = []
    for rt in range(1, iterations + 1):
        for p_val in p_list:
            p_val = round(p_val, 4)
            sw = pd.read_csv(sw_path + '/' + 'p=' + str(p_val) + '/' + str(rt) + '.csv', header=None)
            sw.columns = ['source', 'target']
            graph = nx.from_pandas_edgelist(sw, 'source', 'target', create_using=nx.Graph())
            ave_clustering.append(nx.average_clustering(graph))
            list_spl.append(nx.average_shortest_path_length(graph))
            ps.append(p_val)
    graph_fearture = pd.DataFrame()
    graph_fearture['C'] = ave_clustering
    graph_fearture['L'] = list_spl
    graph_fearture['p'] = ps
    graph_fearture['C(p)/C(0)'] = graph_fearture['C'] / c0
    graph_fearture['L(p)/L(0)'] = graph_fearture['L'] / spl0

    sa_path = os.path.join('output', 'sw_process')
    if os.path.exists(sa_path):
        pass
    else:
        os.makedirs(sa_path)
    graph_fearture.to_csv(sa_path + '/' + '/sw_4.csv', index=False)


def startup(iterations):
    cal_smallworld_c_spl(iterations)
