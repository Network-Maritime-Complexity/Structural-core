#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


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
        save_path = os.path.join('output', 'Structural_embeddedness_and_economic_performance_of_ports')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Table 1 Pearson correlation coefficients between network indicators and port capacity.csv'
        df_pearsonr.to_csv(save_path + '/' + filename, index=True, index_label='Network indicators')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
        print()


def startup():
    print()
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Subsection titled "Structural embeddedness and economic performance of ports"')
    print('Section titled "Results"')
    print('*********************************')
    print()
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()
    cal_pr(Nodes)
