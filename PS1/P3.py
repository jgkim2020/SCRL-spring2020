import argparse
import copy
from cvxopt import matrix, solvers
import numpy as np
import random


def run_value_iteration(mdp):
    # Initialize value
    val = np.zeros_like(mdp['S']) # (s,)
    delta_val = np.inf
    step = 0
    while delta_val > 1e-12:
        old_val = val
        # Bellman optimality update
        return_sa = mdp['r'] + mdp['gamma']*np.einsum('ijk,i', mdp['P'], val) # (s,a)
        val = np.max(return_sa, axis=1) # (s,)
        pol = np.argmax(return_sa, axis=1) # (s,)
        # Termination condition
        delta_val = np.max(np.abs(val - old_val))
        step = step + 1
    print('value error %.10f @ VI step %d'%(delta_val, step))
    opt_val = val # (s,)
    opt_pol = pol # (s,)
    print('optimal value function:', opt_val)
    print('optimal policy:', opt_pol)

def run_policy_iteration(mdp):
    # Initialize policy and value
    pol = np.zeros_like(mdp['S']) # (s,)
    val_pi = np.zeros_like(mdp['S']) # (s,)
    delta_pol = np.inf
    step = 0
    while delta_pol > 0:
        old_pol = pol
        # Policy evaluation (inexact evaluation)
        for _ in range(100):
            return_sa_pi = mdp['r'] + mdp['gamma']*np.einsum('ijk,i', mdp['P'], val_pi) # (s,a)
            val_pi = return_sa_pi[np.arange(pol.shape[0]),pol] # (s)
        # Policy improvement
        return_sa_pi = mdp['r'] + mdp['gamma']*np.einsum('ijk,i', mdp['P'], val_pi) # (s,a)
        pol = np.argmax(return_sa_pi, axis=1) # (s,)
        # Termination condition
        delta_pol = np.count_nonzero(pol - old_pol) 
        step = step + 1
    print('policy converged @ PI step %d'%(step))
    opt_val = val_pi
    opt_pol = pol
    print('optimal value function:', opt_val)
    print('optimal policy:', opt_pol)

def run_linear_programming(mdp):
    '''
    minimize c^{T}*v
    subject to  -v(s) + gamma*Sigma_{s'}p(s'|s,a)v(s') <= -r(s,a)
    '''
    c = matrix([100.0]*mdp['S'].shape[0]) # (sx1)
    b = matrix(-mdp['r'].flatten()) # (sax1)
    A = []
    for s in range(mdp['S'].shape[0]):
        for a in range(mdp['A'].shape[0]):
            coeff = mdp['gamma']*mdp['P'][:,s,a]
            coeff[s] = coeff[s] - 1
            A.append(coeff)
    A = matrix(np.array(A)) # (saxs)
    sol = solvers.lp(c,A,b)
    opt_val = np.squeeze(sol['x']) # (s,)
    return_sa = mdp['r'] + mdp['gamma']*np.einsum('ijk,i', mdp['P'], opt_val) # (s,a)
    opt_pol = np.argmax(return_sa, axis=1) # (s,)
    print('optimal value function:', opt_val)
    print('optimal policy:', opt_pol)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem Set #1. Problem #3')
    algo_list = ['VI', 'PI', 'LP']
    parser.add_argument('--algo_type', type=str,
        help="Specify an algorithm. Available algorithms are: %s"%algo_list
    )
    args = parser.parse_args()

    # Define MDP
    mdp = {
        'S': np.array([0,1]), # (s,)
        'A': np.array([0,1]), # (a,)
        'P': np.array([[[0.75,0.25],[0.75,0.25]], [[0.25,0.75],[0.25,0.75]]]).transpose(), # (a, s, s').transpose()
        'r': np.array([[-2,-0.5],[-1,-3]]), # (s, a)
        'gamma': 0.9
    }
    # Optimize
    if args.algo_type == 'VI':
        run_value_iteration(mdp)
    elif args.algo_type == 'PI':
        run_policy_iteration(mdp)
    elif args.algo_type == 'LP':
        run_linear_programming(mdp)
    else:
        raise ValueError('Specify an algorithm (--algo_type). Available algorithms are: %s'%algo_list)