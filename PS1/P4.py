import argparse
import numpy as np


def run_relative_value_iteration(mdp):
    # Initialize gain & bias
    bias = np.zeros_like(mdp['S']) # (s,)
    delta_bias = np.inf
    step = 0
    while delta_bias > 1e-10:
        old_bias = bias
        # Bellman optimality update for bias
        temp = mdp['r'] + np.einsum('ijk,i', mdp['P'], bias) # (s,a)
        bias = np.max(temp, axis=1) # T_h
        gain = bias[0] # T_h(s_bar)
        bias = bias - gain*np.ones_like(bias) # Th - Th(s_bar)
        pol = np.argmax(temp, axis=1)
        # Termination condition
        delta_bias = np.max(np.abs(bias - old_bias))
        step = step + 1
    print('value error %.10f @ RVI step %d'%(delta_bias, step))
    opt_gain = gain
    opt_bias = bias
    opt_pol = pol
    print('optimal gain function:', opt_gain)
    print('optimal bias function:', opt_bias)
    print('optimal policy:', opt_pol)

def run_policy_iteration(mdp):
    # Initialize policy & bias
    pol = np.zeros_like(mdp['S']) # (s,)
    pol = np.array([0,1])
    bias_pi = np.zeros_like(mdp['S']) # (s,)
    delta_pol = np.inf
    step = 0
    while delta_pol > 0:
        old_pol = pol
        # Policy evaluation (inexact evaluation)
        for _ in range(15):
            temp = mdp['r'] + np.einsum('ijk,i', mdp['P'], bias_pi) # (s,a)
            rhs = temp[np.arange(pol.shape[0]), pol] # (s,)
            gain_pi, bias_pi = rhs[0], rhs - rhs[0]*np.ones_like(rhs)
        # Policy improvement
        temp = mdp['r'] + np.einsum('ijk,i', mdp['P'], bias_pi) # (s,a)
        pol = np.argmax(temp, axis=1) # (s,)
        # Termination condition
        delta_pol = np.count_nonzero(pol - old_pol) 
        step = step + 1
    print('policy converged @ PI step %d'%(step))
    opt_gain = gain_pi
    opt_bias = bias_pi
    opt_pol = pol
    print('optimal gain function:', opt_gain)
    print('optimal bias function:', opt_bias)
    print('optimal policy:', opt_pol)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem Set #1. Problem #3')
    algo_list = ['RVI', 'PI']
    parser.add_argument('--algo_type', type=str,
        help="Specify an algorithm. Available algorithms are: %s"%algo_list
    )
    args = parser.parse_args()

    # Define MDP
    mdp = {
        'S': np.array([0,1]), # (s,)
        'A': np.array([0,1]), # (a,)
        'P': np.array([[[0.75,0.25],[0.75,0.25]], [[0.25,0.75],[0.25,0.75]]]).transpose(), # (a, s, s').transpose()
        'r': np.array([[-2,-0.5],[-1,-3]]) # (s, a)
    }
    # Optimize
    if args.algo_type == 'RVI':
        run_relative_value_iteration(mdp)
    elif args.algo_type == 'PI':
        run_policy_iteration(mdp)
    else:
        raise ValueError('Specify an algorithm (--algo_type). Available algorithms are: %s'%algo_list)