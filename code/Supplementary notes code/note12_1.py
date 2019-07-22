#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def startup():
    data_path = os.path.join('../', 'data', 'downloaded data files', 'note12_1')
    if os.path.exists(data_path):
        print('*********************************')
        print("Location in the manuscript text: ")
        print('Subsection titled "(1) Constraints of the number of shipping routes"')
        print('Section titled "Supplementary note 12: Influence of the constraints on the structural core of the GLSN"')
        print('*********************************')
        print()
        print('***************************RUN TIME WARNING***************************')
        print('It needs 2 days for 1000 iterations of the corresponding experiments.')
        print()
        print('---------------------------------------------------------------------------------------------------')
        print('Output:')
        print()
        print('**********************************************************************************************')
        print('Note: The number of iterations of the experiment: in your test, {}; in '
              'the manuscript, 1000.'.format(iters))
        print('**********************************************************************************************')
        print()
        from src import note12_1_Inc
        from src import note12_1_Rand
        p1 = mp.Process(target=note12_1_Inc.startup, args=(iters, ))
        p2 = mp.Process(target=note12_1_Rand.startup, args=(iters, ))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    else:
        print()
        print('Please download *downloaded data files.zip* file first!')
        sys.exit()
