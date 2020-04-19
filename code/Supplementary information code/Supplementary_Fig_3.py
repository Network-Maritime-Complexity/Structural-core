#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


data_path = os.path.join('../data', '1000 equivalent random networks')


def significance_test_1():
    param = 'B'
    den_empirical = 2.43
    df_density = pd.DataFrame()
    test_1 = []
    for i in range(1, 1001):
        df_nodes = pd.read_csv(data_path + '/Nodes/' + str(i) + '.csv')
        df_edges = pd.read_csv(data_path + '/Edges/' + str(i) + '.csv', header=None)
        df_edges.columns = ['source', 'target']
        graph = nx.from_pandas_edgelist(df_edges, 'source', 'target')

        df_nodes.sort_values(param, ascending=True, inplace=True)
        list_value = sorted(list(set(df_nodes[param].values)))
        list_value = list_value[:-1]

        list_density = []
        for value in list_value:
            index_param = df_nodes[param] >= value
            nodelist = df_nodes["id"][index_param].values.tolist()
            H = graph.subgraph(nodelist)
            list_density.append(round(nx.density(H), 4))
        density = pd.DataFrame()
        density[param] = list_value
        density['Density'] = list_density
        density.sort_values(param, ascending=False, inplace=True)

        df_density = pd.concat([df_density, density], axis=0)
        den_ix = density[param] >= den_empirical
        if density[den_ix].empty:
            nm_density = np.nan
        else:
            nm_density = float(density.loc[den_ix, 'Density'].tail(1).values)
        test_1.append(nm_density)
    data = df_density.groupby(param, as_index=False)['Density'].mean()

    xdata = data[param]
    ydata = data['Density']
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)
    ax.plot(xdata, ydata, color='k', marker='o', ls='', ms=5.5, alpha=0.5, markeredgecolor='w', markeredgewidth=0.15)
    ax.set_xlim([np.floor(xdata.min()), np.ceil(xdata.max())])
    coor_x = 2.43
    x_ix = data[param] == coor_x
    coor_y = round(float(data.loc[x_ix, 'Density'].values), 2)
    ax.set_ylim(-0.05, 1.04)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.04))

    ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=1.8, length=7,
                   pad=5, labelsize=16)
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor', width=1.5, length=4.5)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.annotate("", xy=(coor_x, ax.get_ylim()[0]),
                 xytext=(coor_x, ax.get_ylim()[1]),
                 arrowprops=dict(arrowstyle='-', linestyle=pltstyle.get_linestyles('loosely dashed'), color='gray', lw=3))
    plt.annotate("", xy=(ax.get_xlim()[0], coor_y),
                 xytext=(ax.get_xlim()[1], coor_y),
                 arrowprops=dict(arrowstyle='-', linestyle=pltstyle.get_linestyles('loosely dashed'), color='gray', lw=3))
    ax.text(0.3, 0.65, '$B≥2.43$', fontsize=18)
    ax.text(0.3, 0.55, 'Density$=0.69$', fontsize=18)
    ax.set_xlabel(r'${}$'.format(param), fontsize=22)
    ax.set_ylabel(r'Density among ports of ${}_i≥{}$'.format(param, param), fontsize=16)
    ax.set_title('Null model', fontsize=16, x=0.5, y=1.0, horizontalalignment='center')

    plt.tight_layout()

    save_path = os.path.join('output')
    if SAVE_RESULT:
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = "Supplementary Fig. 3 Statistical significance of the structural core in the real GLSN of 2015 (a) Upper Left.png"
        fig.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')

    fig = plot_probability_density(test_1, '1')

    save_path = os.path.join('output')
    if SAVE_RESULT:
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = "Supplementary Fig. 3 Statistical significance of the structural core in the real GLSN of 2015 (a) Lower.png"
        fig.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def significance_test_2():
    num_core = 37
    test_2 = []
    core_ports = Nodes.sort_values('B', ascending=False)[:num_core]['id'].values
    for rt in range(1, 1001):
        edges = pd.read_csv(data_path + '/Edges/' + str(rt) + '.csv', header=None)
        edges.columns = ['source', 'target']
        g = nx.from_pandas_edgelist(edges, 'source', 'target', create_using=nx.Graph())
        H = g.subgraph(core_ports)
        density = round(nx.density(H), 4)
        test_2.append(density)

    fig = plot_probability_density(test_2, '2')
    save_path = os.path.join('output')
    if SAVE_RESULT:
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = "Supplementary Fig. 3 Statistical significance of the structural core in the real GLSN of 2015 (b) Lower.png"
        fig.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def plot_probability_density(x, test):
    fig = plt.figure(figsize=(6.7, 4.8))
    ax = fig.add_subplot(111)
    sns.kdeplot(x, color="k", legend=False, lw=2, alpha=1)

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel('probability density', fontsize=26, alpha=1)

    ax.set_title('p-value < 0.001', fontsize=24, color='r', loc='right', pad=10, weight='medium')

    if test == '1':
        ax.text(0.59, 13, r'null' + '\n' + 'distribution', fontsize=23, color='k', horizontalalignment='center', alpha=1)
        ax.text(0.87, 13, r'observed' + '\n' + 'value', fontsize=23, color='red', horizontalalignment='center')
        ax.set_xlabel('Density among ports of $B≥2.43$', family='Arial', fontsize=26, weight='medium', alpha=1)
        ax.set_ylim(0, 20)
        ax.set_xlim(0.48, 0.95)
        ax.yaxis.set_major_locator(MultipleLocator(5))
    if test == '2':
        ax.text(0.71, 20, r'null' + '\n' + 'distribution', fontsize=22, color='k', horizontalalignment='center', alpha=1)
        ax.text(0.865, 20, r'observed' + '\n' + 'value', fontsize=22, color='red', horizontalalignment='center')
        ax.set_xlabel('Density among structural-core ports', family='Arial', fontsize=26, weight='medium', alpha=1)
        ax.set_ylim(0, 30)
        ax.set_xlim(0.48, 0.93)
        ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=2.5, length=8,
                   pad=5, labelsize=24)

    ax.annotate(s='', xy=(0.80, ax.get_ylim()[0]),
                xytext=(0.80, ax.get_ylim()[1]), arrowprops=dict(arrowstyle='-', lw=2.5, color='red',
                                                                 ls=pltstyle.get_linestyles('loosely dashed'), alpha=1))

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.tight_layout()
    return fig


def startup():
    if os.path.exists(data_path):
        print()
        print('*********************************')
        print("Location in the manuscript text: ")
        print('Section titled "Supplementary Figures"')
        print('This script reproduce the result figure titled "Supplementary Fig. 3 Statistical significance of the '
              'structural core in the real GLSN of 2015"')
        print('*********************************')
        print()
        print('***************************RUN TIME WARNING***************************')
        print('It needs 45 minutes for corresponding experiments.')
        print()
        print('---------------------------------------------------------------------------------------------------')
        print('Output:')
        significance_test_1()
        significance_test_2()
    else:
        print()
        print('Please download (in this link: https://doi.org/10.6084/m9.figshare.12136236.v1) zip files first!')
        sys.exit()
