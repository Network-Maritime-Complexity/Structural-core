#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


import sys
import os
import warnings
import shutil
import time
import random
import itertools
from src import pltstyle
import pandas as pd
import numpy as np
import powerlaw
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import community
from sklearn.metrics import r2_score
import seaborn as sns


YEAR = '2015'
DATASET = 'All'
METHOD = 'P'
SAVE_RESULT = True
dis_col = 'Distance(SR,unit:km)'
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

Edges = pd.read_csv('../data/GLSN data/Edges_' + YEAR + '_' + DATASET + '_' + METHOD + '.csv', header=None)
Edges.columns = ['source', 'target']
G = nx.from_pandas_edgelist(Edges, 'source', 'target', create_using=nx.Graph(), edge_attr=None)
Nodes = pd.read_csv('../data/Other data/Nodes_' + YEAR + '_' + DATASET + '_' + METHOD + '.csv')
