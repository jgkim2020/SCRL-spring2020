import argparse
from cvxopt import matrix, solvers
import numpy as np


def run_value_iteration()
    raise NotImplementedError()

def run_policy_iteration()
    raise NotImplementedError()

def run_linear_programming()
    raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem Set #1. Problem #3')
    algo_list = ['VI', 'PI', 'LP']
    parser.add_argument('--algo_type', type=str,
        help="Choose an algorithm. Available algorithms are: %s"%algo_list
    )
    args = parser.parse_args()

    if args.algo_type == 'VI':
        run_value_iteration()
    elif args.algo_type == 'PI':
        run_policy_iteration()
    elif args.algo_type == 'LP':
        run_linear_programming()
    else:
        raise ValueError('Undefined algorithm type')