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
        if height != 0:
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
    ax.set_ylabel(r'# ports', fontsize=24, labelpad=5)

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
        save_path = os.path.join('output', 'Defining_structural_core')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if param == 'B':
            filename = 'Fig. 6 Results for the structural core detection of the GLSN (a) Right.png'
        elif param == 'Z':
            filename = 'Fig. 6 Results for the structural core detection of the GLSN (b) Right.png'
        else:
            filename = 'Fig. 6 Results for the structural core detection of the GLSN (c) Right.png'
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
        save_path = os.path.join('output', 'Defining_structural_core')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if param == 'B':
            filename = 'Fig. 6 Results for the structural core detection of the GLSN (a) Left.png'
        elif param == 'Z':
            filename = 'Fig. 6 Results for the structural core detection of the GLSN (b) Left.png'
        else:
            filename = 'Fig. 6 Results for the structural core detection of the GLSN (c) Left.png'
        fig.savefig(save_path + '/' + filename, bbox_inches='tight', pad_inches=0.08)
        print()
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')


def significance_test_1():
    param = 'B'
    den_empirical = 2.43
    df_density = pd.DataFrame()
    test_1 = []
    for i in range(1, iters+1):
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

    save_path = os.path.join('output', 'Defining_structural_core')
    if SAVE_RESULT:
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = "Fig. 7 Statistical significance of a structural core in the real GLSN (a) Upper Left.png"
        fig.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()
    else:
        plt.show()
    plt.close('all')

    fig = plot_probability_density(test_1, '1')

    save_path = os.path.join('output', 'Defining_structural_core')
    if SAVE_RESULT:
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = "Fig. 7 Statistical significance of a structural core in the real GLSN (a) Lower.png"
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
    for rt in range(1, iters+1):
        edges = pd.read_csv(data_path + '/Edges/' + str(rt) + '.csv', header=None)
        edges.columns = ['source', 'target']
        g = nx.from_pandas_edgelist(edges, 'source', 'target', create_using=nx.Graph())
        H = g.subgraph(core_ports)
        density = round(nx.density(H), 4)
        test_2.append(density)

    fig = plot_probability_density(test_2, '2')
    save_path = os.path.join('output', 'Defining_structural_core')
    if SAVE_RESULT:
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = "Fig. 7 Statistical significance of a structural core in the real GLSN (b) Lower.png"
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
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Subsection titled "Defining structural core"')
    print('Section titled "Gateway-hub-based structural core"')
    print('*********************************')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    params = [['B', 1.5], ['Z', 1.5], ['P', 0.7]]
    for pp in params:
        defining_sc(pp)
        # significance_test_1()
        # significance_test_2()
