#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


import sys


def main(parts_of_the_manuscript):
    if 'note2' in parts_of_the_manuscript:
        import note2
        note2.startup()
    if 'note3' in parts_of_the_manuscript:
        import note3
        note3.startup()
    if 'note4' in parts_of_the_manuscript:
        import note4
        note4.startup()
    if 'note5' in parts_of_the_manuscript:
        import note5
        note5.startup()
    if 'note10' in parts_of_the_manuscript:
        import note10
        note10.startup()
    if 'note11' in parts_of_the_manuscript:
        import note11
        note11.startup()
    if 'note12_1' in parts_of_the_manuscript:
        import note12_1
        note12_1.startup()
    if 'note12_2' in parts_of_the_manuscript:
        import note12_2
        note12_2.startup()
    if 'note12_3' in parts_of_the_manuscript:
        import note12_3
        note12_3.startup()
    if 'note13' in parts_of_the_manuscript:
        import note13
        note13.startup()
    if 'note14' in parts_of_the_manuscript:
        import note14
        note14.startup()


if __name__ == "__main__":
    import configure
    import time

    all_args = ['note2', 'note3', 'note4', 'note5', 'note10', 'note11',
                'note12_1', 'note12_2', 'note12_3', 'note13', 'note14']

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

    iters_modules = ['note4', 'note5', 'note10', 'note11', 'note12_1', 'note12_2', 'note12_3',
                     'note13']
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
