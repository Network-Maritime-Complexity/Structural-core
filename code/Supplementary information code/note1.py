#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def startup():
    print('*********************************')
    print("Location in the manuscript text: ")
    print('Section titled "Supplementary note 1: Statistical significance of the economic small-world-ness'
          ' of the GLSN"')
    print('*********************************')
    print()
    print('***************************RUN TIME WARNING***************************')
    print('It needs 9 minutes for corresponding experiments.')
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Output:')
    print()
    from src import effi_empirical
    p1 = mp.Process(target=effi_empirical.startup, args=('Distance(SR,unit:km)', ))
    p2 = mp.Process(target=effi_empirical.startup, args=('Distance(GC,unit:km)', ))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
