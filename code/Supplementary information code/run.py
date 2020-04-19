#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


import sys


def main(parts_of_the_manuscript):
    if 'Supplementary_Fig_2' in parts_of_the_manuscript:
        import Supplementary_Fig_2
        Supplementary_Fig_2.startup()
    if 'Supplementary_Fig_3' in parts_of_the_manuscript:
        import Supplementary_Fig_3
        Supplementary_Fig_3.startup()
    if 'Supplementary_Fig_6' in parts_of_the_manuscript:
        import Supplementary_Fig_6
        Supplementary_Fig_6.startup()
    if 'Supplementary_Fig_7' in parts_of_the_manuscript:
        import Supplementary_Fig_7
        Supplementary_Fig_7.startup()
    if 'note1' in parts_of_the_manuscript:
        import note1
        note1.startup()
    if 'note5' in parts_of_the_manuscript:
        import note5
        note5.startup()
    if 'note6' in parts_of_the_manuscript:
        import note6
        note6.startup()
    if 'note8' in parts_of_the_manuscript:
        import note8
        note8.startup()
    if 'note9' in parts_of_the_manuscript:
        import note9
        note9.startup()
    if 'note10_1' in parts_of_the_manuscript:
        import note10_1
        note10_1.startup()
    if 'note10_2' in parts_of_the_manuscript:
        import note10_2
        note10_2.startup()
    if 'note10_3' in parts_of_the_manuscript:
        import note10_3
        note10_3.startup()
    if 'note11' in parts_of_the_manuscript:
        import note11
        note11.startup()


if __name__ == "__main__":
    import configure
    import time

    all_args = ['Supplementary_Fig_2', 'Supplementary_Fig_3', 'Supplementary_Fig_6', 'Supplementary_Fig_7',
                'note1', 'note5', 'note6', 'note8', 'note9',
                'note10_1', 'note10_2', 'note10_3', 'note11']

    iterations = 1000
    parts_of_the_manuscript = all_args

    if len(sys.argv) == 1:
        iterations = 1000
        parts_of_the_manuscript = all_args

    if len(sys.argv) == 2:
        try:
            iterations = int(sys.argv[1])
            parts_of_the_manuscript = all_args
        except:
            if sys.argv[1] not in all_args:
                print('Please input the right args from: {}'.format(all_args))
                sys.exit()
            else:
                parts_of_the_manuscript = []
                iterations = 1000
                parts_of_the_manuscript.append(sys.argv[1])
    if len(sys.argv) > 2:
        try:
            iterations = int(sys.argv[1])
            parts_of_the_manuscript = sys.argv[2:]

        except:
            iterations = 1000
            parts_of_the_manuscript = sys.argv[1:]
            for part in parts_of_the_manuscript:
                if part not in all_args:
                    print('Please input the right args from: {}'.format(all_args))
                    sys.exit()

    iters_modules = ['Supplementary_Fig_6', 'Supplementary_Fig_7', 'note5', 'note6', 'note9',
                     'note10_1', 'note10_2', 'note10_3', 'note11']
    print()
    if len(list(set(parts_of_the_manuscript).intersection(set(iters_modules)))):
        print('iters: {}'.format(iterations))
    else:
        pass
    
    print('parts_of_the_manuscript: {}'.format(parts_of_the_manuscript))
    print()
    configure.iters = iterations
    s_time = time.time()
    main(parts_of_the_manuscript)

    print()
    print()
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Code performance: {:.0f}s.'.format(time.time() - s_time))
