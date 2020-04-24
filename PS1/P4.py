import argparse
import numpy as np


def run_relative_value_iteration():
    raise NotImplementedError()

def run_policy_iteration():
    raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem Set #1. Problem #3')
    algo_list = ['RVI', 'PI']
    parser.add_argument('--algo_type', type=str,
        help="Specify an algorithm. Available algorithms are: %s"%algo_list
    )
    args = parser.parse_args()

    if args.algo_type == 'RVI':
        run_relative_value_iteration()
    elif args.algo_type == 'PI':
        run_policy_iteration()
    else:
        raise ValueError('Specify an algorithm. Available algorithms are: %s'%algo_list)