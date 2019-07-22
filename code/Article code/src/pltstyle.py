# -*- coding: utf-8 -*-


from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def axes_style(ax):
    """
    Axes style
    :param ax: axes to operate
    :return: None
    """
    plt.rcParams.update({'font.family': 'Arial'})

    ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=1.8, length=7,
                   pad=7, labelsize=20)
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor', width=1.5, length=4.5)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)


def anno_style(use_key):
    """
    Returns the line style dict.
    :param use_key: str, 'k' is for black solid line, 'r' is for red loosely dashed line
    :return: style_dict, dict, line style dict.
    """
    anno_dict = {
        'r': dict(arrowstyle="-", linestyle=get_linestyles('loosely dashed'),color='r',linewidth=2.5),
        'k': dict(arrowstyle="-", linestyle='-', color='k', linewidth=2),
                }

    style_dict = anno_dict.get(use_key)

    return style_dict


def get_linestyles(key):
    """
    Returns line style tuple
    :param key: str, line style name
    :return: linestyles[key]
    """
    linestyles = OrderedDict(
        [('solid',               (0,())),
         ('loosely dotted',      (0,(1,10))),
         ('dotted',              (0, (1, 5))),
         ('densely dotted',      (0, (1, 1))),

         ('loosely dashed',      (-5, (4, 4))),
         ('dashed',              (0, (5, 5))),
         ('densely dashed',      (0, (5, 1))),

         ('loosely dashdotted',  (0, (3, 10, 1, 10))),
         ('dashdotted',          (0, (3, 5, 1, 5))),
         ('densely dashdotted',  (0, (3, 1, 1, 1))),

         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    return linestyles[key]


def plot_glsn_basic(ax, xpoint, ypoint):
    plt.rcParams.update({'font.family': 'Arial', 'font.weight': 'medium'})
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.04))

    ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=1.8, length=7,
                   pad=5, labelsize=22)
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor', width=1.5, length=4.5)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    if xpoint:
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        plt.annotate("", xy=(xpoint, ymin),
                     xytext=(xpoint, ymax),
                     arrowprops=dict(arrowstyle='-', linestyle=get_linestyles('loosely dashed'), color='r', lw=3))
    if ypoint:
        xmin = ax.get_xlim()[0]
        xmax = ax.get_xlim()[1]
        plt.annotate("", xy=(xmin, ypoint),
                     xytext=(xmax, ypoint),
                     arrowprops=dict(arrowstyle='-', linestyle=get_linestyles('loosely dashed'), color='r', lw=3))


def plot_sub_basic(ax, xpoint, ypoint):
    ax.set_ylim(-0.05, 1.05)
    plt.rcParams.update({'font.family': 'Arial', 'font.weight': 'bold'})

    # ax.set_yticklabels('major', visible=False)
    # ax.set_yticks([])
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='major', width=1.8, length=7,
                   pad=4)
    ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor', width=1.5, length=4.5)

    plt.xticks(fontproperties='Arial', fontsize=16, weight='bold')
    plt.yticks(fontproperties='Arial', fontsize=16, weight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.spines['bottom'].set_linewidth(2.3)
    ax.spines['left'].set_linewidth(2.3)
    ax.spines['top'].set_linewidth(2.3)
    ax.spines['right'].set_linewidth(2.3)

    if xpoint:
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        plt.annotate("", xy=(xpoint, ymin),
                     xytext=(xpoint, ymax),
                     arrowprops=dict(arrowstyle='-', linestyle=get_linestyles('loosely dashed'), color='r', lw=3))
    if ypoint:
        xmin = ax.get_xlim()[0]
        xmax = ax.get_xlim()[1]
        plt.annotate("", xy=(xmin, ypoint),
                     xytext=(xmax, ypoint),
                     arrowprops=dict(arrowstyle='-', linestyle=get_linestyles('loosely dashed'), color='r', lw=3))

