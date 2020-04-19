#! python3
# -*- coding: utf-8 -*-
"""

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


import sys


def main(parts_of_the_manuscript):
    if 'Basic_topological_properties_and_economic_small_world_ness' in parts_of_the_manuscript:
        import Basic_topological_properties_and_economic_small_world_ness
        Basic_topological_properties_and_economic_small_world_ness.startup()
    if 'Gateway_hub_structural_core' in parts_of_the_manuscript:
        import Gateway_hub_structural_core
        Gateway_hub_structural_core.startup()
    if 'Structural_embeddedness_and_economic_performance_of_ports' in parts_of_the_manuscript:
        import Structural_embeddedness_and_economic_performance_of_ports
        Structural_embeddedness_and_economic_performance_of_ports.startup()
    if 'Structural_core_and_international_trade' in parts_of_the_manuscript:
        import Structural_core_and_international_trade
        Structural_core_and_international_trade.startup()


if __name__ == "__main__":
    import time

    all_args = ['Basic_topological_properties_and_economic_small_world_ness',
                'Gateway_hub_structural_core',
                'Structural_embeddedness_and_economic_performance_of_ports',
                'Structural_core_and_international_trade']

    parts_of_the_manuscript = []
    if len(sys.argv) == 1:
        parts_of_the_manuscript = all_args

    if len(sys.argv) == 2:

        if sys.argv[1] not in all_args:
            print('Please input the right *parts_of_the_manuscript* from: {}'.format(all_args))
            sys.exit()
        else:
            parts_of_the_manuscript.append(sys.argv[1])
    if len(sys.argv) > 2:
        parts_of_the_manuscript = sys.argv[1:]
        for part in parts_of_the_manuscript:
            if part not in all_args:
                print('Please input the right args from: {}'.format(all_args))
                sys.exit()
            else:
                pass
    print()
    print('parts_of_the_manuscript: {}'.format(parts_of_the_manuscript))
    s_time = time.time()
    main(parts_of_the_manuscript)
    print()
    print()
    print()
    print('---------------------------------------------------------------------------------------------------')
    print('Code performance: {:.0f}s.'.format(time.time() - s_time))
