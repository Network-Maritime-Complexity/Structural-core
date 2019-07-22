#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def startup():

    data_path = os.path.join('../', 'data', 'downloaded data files', 'note12_2')
    if os.path.exists(data_path):
        print('*********************************')
        print("Location in the manuscript text: ")
        print('Subsection titled "(2) Geographical constraints"')
        print('Section titled "Supplementary note 12: Influence of the constraints on the structural core of the GLSN"')
        print('*********************************')
        print()
        print('***************************RUN TIME WARNING***************************')
        print('It needs 3 days for 1000 iterations of the corresponding experiments.')
        print()
        print('---------------------------------------------------------------------------------------------------')
        print('Output:')
        print()
        print('**********************************************************************************************')
        print('Note: The number of iterations of the experiment: in your test, {}; in '
              'the manuscript, 1000.'.format(iters))
        print('**********************************************************************************************')
        print()
        from src import note12_2_method1
        from src import note12_2_method2
        from src import note12_2_method3
        p1 = mp.Process(target=note12_2_method1.startup, args=(iters, ))
        p2 = mp.Process(target=note12_2_method2.startup, args=(iters, ))
        p3 = mp.Process(target=note12_2_method3.startup, args=(iters, ))
        p1.start()
        p2.start()
        p3.start()
        p1.join()
        p2.join()
        p3.join()

        del_path = 'output/dis_process'
        if os.path.exists(del_path):
            shutil.rmtree(del_path)
        else:
            pass
    else:
        print()
        print('Please download *downloaded data files.zip* file first!')
        sys.exit()
