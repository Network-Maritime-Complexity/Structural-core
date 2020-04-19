from configure import *


def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if height != 0:
            ax.text(rect.get_x()+rect.get_width()/2., 1*height, '%d' % int(height),
                     ha='center', va='bottom', fontsize=24)


def hist_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', direction='in', which='major', width=2.5, length=6, labelsize=20)
    ax.set_xlim(ax.get_xlim()[0] - 0.4, ax.get_xlim()[1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)


def plot_hist(dict_num_ports):

    plt.rcParams.update({'font.family': 'Arial'})
    n_rows = 1
    n_cols = 4
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8, 2.5))
    comms = list(dict_num_ports.keys())
    ix = 0
    for i in range(n_cols):
        ax = axes[i]
        y = dict_num_ports[comms[ix]]
        x = np.arange(1, len(y)+1, 1)
        rects = ax.bar(x, y, facecolor='k', alpha=0.7, width=0.4)
        hist_style(ax)
        ax.set_xlabel(r'sub-modules', fontsize=20, labelpad=2)
        ax.set_ylabel(r'# ports', fontsize=20, labelpad=3)
        ax.set_xticks(x)
        autolabel(rects, ax)
        ax.set_ylim([0, max(y)])
        ax.set_title('Module {}'.format(comms[ix]), fontsize=20, pad=30)
        if max(y) < 7:
            ax.set_yticks(np.arange(0, max(y)+1, 1))
        else:
            ax.set_yticks(np.arange(0, max(y) + 1, 2))
        ix += 1
    plt.tight_layout()
    if SAVE_RESULT:
        save_path = os.path.join('output', 'Supplementary note 6')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        filename = 'Supplementary Fig. 13 Results for the structural-core...submodular connector hubs (b).png'
        fig.savefig(save_path + '/' + filename, bbox_inches='tight')
        print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    else:
        plt.show()
    plt.close('all')


def startup(dict_num_ports):
    plot_hist(dict_num_ports)
