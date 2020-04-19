#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def calculate_intra_inter_connections_percentage():
    def _cal_dis(d_col):
        dis_all = pd.read_csv('../data/Other data/Distance_SR_GC_' + YEAR + '.csv')
        dict_dis = dict(zip(dis_all['Edge'], dis_all[d_col]))
        dict_module = dict(zip(Nodes['id'], Nodes['Community']))
        df_edges = Edges.copy()
        df_edges['source_comm'] = df_edges['source'].apply(dict_module.get)
        df_edges['target_comm'] = df_edges['target'].apply(dict_module.get)
        df_edges['Edge'] = df_edges['source'].astype(str) + '--' + df_edges['target'].astype(str)
        df_edges[d_col] = df_edges['Edge'].apply(dict_dis.get)

        list_res = []
        df_edges['edge_type'] = np.where(df_edges['target_comm'] == df_edges['source_comm'], "intra", "inter")

        '''------  <2.5  ------'''
        dis_index1 = df_edges[d_col] < 2500
        df_edges_short_dis1 = df_edges[dis_index1]
        count1 = len(df_edges_short_dis1)
        pr_1 = count1 / len(df_edges)
        s1 = pd.Series(((df_edges_short_dis1.groupby('edge_type')['source'].count() / count1) * pr_1), name='0')
        cc1 = round(s1, 4)
        list_res.append(cc1)

        '''------  2.5-5  ------'''
        df_edges_short_dis2 = df_edges.loc[(df_edges[d_col] >= 2500) & (df_edges[d_col] < 5000)]
        count2 = len(df_edges_short_dis2)
        pr_2 = count2 / len(df_edges)
        s2 = pd.Series(((df_edges_short_dis2.groupby('edge_type')['source'].count() / count2) * pr_2), name='1')
        cc2 = round(s2, 4)
        list_res.append(cc2)

        '''------  5-7.5  ------'''
        df_edges_short_dis3 = df_edges.loc[(df_edges[d_col] >= 5000) & (df_edges[d_col] < 7500)]
        count3 = len(df_edges_short_dis3)
        pr_3 = count3 / len(df_edges)
        s3 = pd.Series(((df_edges_short_dis3.groupby('edge_type')['source'].count() / count3) * pr_3), name='2')
        cc3 = round(s3, 4)
        list_res.append(cc3)

        """------  7.5-10  ------"""
        df_edges_short_dis4 = df_edges.loc[(df_edges[d_col] >= 7500) & (df_edges[d_col] < 10000)]
        count4 = len(df_edges_short_dis4)
        pr_4 = count4 / len(df_edges)
        s4 = pd.Series(((df_edges_short_dis4.groupby('edge_type')['source'].count() / count4) * pr_4), name='3')
        cc4 = round(s4, 4)
        list_res.append(cc4)

        '''------  >10  ------'''
        df_edges_short_dis5 = df_edges[df_edges[d_col] >= 10000]
        count5 = len(df_edges_short_dis5)
        pr_5 = count5 / len(df_edges)
        s5 = pd.Series(((df_edges_short_dis5.groupby('edge_type')['source'].count() / count5) * pr_5), name='4')
        cc5 = round(s5, 4)
        list_res.append(cc5)
        df_res = pd.DataFrame(list_res)
        dis_describe = df_edges.groupby('edge_type')[d_col].describe()
        return dis_describe, df_res

    d_cols = ['Distance(SR,unit:km)', 'Distance(GC,unit:km)']
    dict_dis = {}
    dict_res = {}
    for i, d_col in enumerate(d_cols):
        dict_dis[i], dict_res[i] = _cal_dis(d_col)

    return dict_res


def plot_result(data):
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(11, 5.5))
    N = 5
    ind = np.arange(N)
    xlabels = ['by real nautical distance', 'by great-circle distance']
    width = 0.35  # the width of the bars: can also be len(x) sequence
    for i in range(2):
        ax = axes[i]
        df = data[i]
        inter = df['inter'] * 100
        intra = df['intra'] * 100
        p1 = ax.bar(ind, inter, width, color='r')
        p2 = ax.bar(ind, intra, width, color='y', bottom=inter)

        ax.set_yticks(np.arange(0, 120, 20))
        font_dict = dict(fontproperties='Arial',  fontsize=22)
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontdict=font_dict)
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xticklabels([])
        ax.set_xticklabels([0, '<2.5', '2.5-5', '5-7.5', '7.5-10', '>10'], fontdict=font_dict)
        ax.set_xlabel(xlabels[i], family='Arial', fontsize=24,  labelpad=10)
        pltstyle.axes_style(ax)
        if i == 0:
            ax.legend((p1[0], p2[0]), (r"inter-module links", r"intra-module links"), fontsize=20)

    axes[0].set_title('(a)', loc='left', fontsize=24, pad=10, style='italic')
    axes[1].set_title('(b)', loc='left', fontsize=24, pad=10, style='italic')

    fig.suptitle(r'Geographical length (10$^3$ km)', family='Arial', fontsize=26,
                 x=0.5, y=-0.0)

    plt.tight_layout(w_pad=5)
    if SAVE_RESULT:
        save_path = os.path.join('output')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Supplementary Fig. 2 Proportional distribution of intra- and inter- module links in ' \
                   'different range of geographical length.png'
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
    print('This script reproduce the result figure titled "Supplementary Fig. 2 Proportional distribution of intra- and inter- module links in different range of geographical length"')
    print('*********************************')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()
    dict_res = calculate_intra_inter_connections_percentage()
    plot_result(dict_res)
