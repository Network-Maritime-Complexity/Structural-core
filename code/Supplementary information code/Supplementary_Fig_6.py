#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""

from configure import *


def _compute_rc(g):

    """Returns the rich-club coefficient for each degree in the graph
    `g`.

    `g` is an undirected graph without multiedges.

    Returns a dictionary mapping degree to rich-club coefficient for
    that degree.

    """
    from networkx.utils import accumulate
    deghist = nx.degree_histogram(g)
    total = sum(deghist)
    # Compute the number of nodes with degree greater than `k`, for each
    # degree `k` (omitting the last entry, which is zero).
    nks = (total - cs for cs in accumulate(deghist) if total - cs > 1)
    # Create a sorted list of pairs of edge endpoint degrees.
    #
    # The list is sorted in reverse order so that we can pop from the
    # right side of the list later, instead of popping from the left
    # side of the list, which would have a linear time cost.
    edge_degrees = sorted((sorted(map(g.degree, e)) for e in g.edges()),
                          reverse=True)
    ek = g.number_of_edges()
    k1, k2 = edge_degrees.pop()
    rc = {}
    for d, nk in enumerate(nks):
        while k1 <= d:
            if len(edge_degrees) == 0:
                ek = 0
                break
            k1, k2 = edge_degrees.pop()
            ek -= 1
        rc[d] = 2 * ek / (nk * (nk - 1))
    return rc


def rich_club_coefficient(g, normalized=True, Q=100, seed=None):

    if nx.number_of_selfloops(g) > 0:
        raise Exception('rich_club_coefficient is not implemented for '
                        'graphs with self loops.')
    rc = _compute_rc(g)
    rc_normed = {}

    if normalized:
        # make R a copy of g, randomize with Q*|E| double edge swaps
        # and use rich_club coefficient of R to normalize
        for rt in range(500):
            R = g.copy()
            E = R.number_of_edges()
            nx.double_edge_swap(R, Q * E, max_tries=Q * E * 10, seed=seed)
            rcran = _compute_rc(R)
            if 0 in rcran.values():
                continue
            else:
                rc_normed = {k: v / rcran[k] for k, v in rc.items()}
                break
    return rc_normed


def cal_rc():
    dict_rc = _compute_rc(G)
    nodedata = Nodes.copy()
    nodedata['phi'] = nodedata['K'].apply(dict_rc.get)

    dict_rc_normed = {}
    for ix in range(1, iters+1):
        dict_rc_normed[ix] = rich_club_coefficient(G, normalized=True, Q=100, seed=None)

    df_rc_normed = pd.DataFrame(dict_rc_normed)
    df_rc_normed['rho_C'] = df_rc_normed.mean(axis=1)

    dict_rc_normed = dict(zip(df_rc_normed.index, df_rc_normed['rho_C']))
    nodedata['rho_C'] = nodedata['K'].apply(dict_rc_normed.get)

    return nodedata


def plot_k_rc(df_nodes):
    df_nodes.sort_values('K', inplace=True)
    cols = ['phi', 'rho_C', 'rho_CM']
    data = df_nodes.apply(pd.to_numeric, errors='coerce')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

    for ax, col in zip(axes, cols):
        ax.plot(data['K'], data[col], 'r-', lw=2.5, alpha=0.73)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.set_xlabel(r'$K$', fontsize=24)
        pltstyle.axes_style(ax)

    axes[0].set_ylabel(r'$\varphi(k)$', fontsize=24)
    axes[1].set_ylabel(r'$\rho_{\rmC}(k)$', fontsize=24)
    axes[2].set_ylabel(r'$\rho_{\rmCM}(k)$', fontsize=24)
    t = plt.suptitle('Note: The number of iterations of the experiment: in your test, {}; in the '
                     'manuscript, 1000.'.format(iters), color='red', style='italic',
                     fontsize=22, x=0.5, y=1.2)
    t.set_bbox(dict(facecolor='gray', alpha=0.3, edgecolor=None))
    plt.tight_layout()
    if SAVE_RESULT:
        save_path = os.path.join('output')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Supplementary Fig. 6 Rich-club coefficients of world ports.png'
        plt.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Section titled "Supplementary Figures"')
    print('This script reproduce the result figure titled "Supplementary Fig. 6 Rich-club coefficients of world ports"')
    print('*********************************')
    print()
    print('***************************RUN TIME WARNING***************************')
    print('It needs 15 hours for 1000 iterations of the corresponding experiments.')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()
    print('**********************************************************************************************')
    print('Note: The number of iterations of the experiment: in your test, {}; in '
          'the manuscript, 1000.'.format(iters))
    print('**********************************************************************************************')
    print()
    df_nodes = cal_rc()
    plot_k_rc(df_nodes)
