import argparse
from cvxopt import matrix, solvers 


def run_linear_program():
    '''
    minimize c^{T}x
    subject to  Ax <= b

    minimize 2*x_{1} + x_{2}
    subject to  -x_{1} + x_{2} <= 1
                x_{1} + x_{2} >= 2
                x_{2} >= 0
                x_{1} - 2*x_{2} <= 4
    '''
    # LP example from https://cvxopt.org/examples/tutorial/lp.html
    print('LP example:')
    A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ]) # (4x2)
    b = matrix([ 1.0, -2.0, 0.0, 4.0 ]) # (4x1)
    c = matrix([ 2.0, 1.0 ]) # (2x1)
    sol = solvers.lp(c,A,b)
    print(sol['x'])

def run_quadratic_program():
    '''
    minimize x^{T}Qx + p^{T}x
    subject to  Gx <= h
                Ax = b

    minimize 2*x_{1}^2 + x_{2}^2 + x_{1}*x_{2} + x_{1} + x_{2}
    subject to  x_{1} >= 0
                x_{2} >= 0
                x_{1} + x_{2} = 1
    '''
    # QP example from https://cvxopt.org/examples/tutorial/qp.html
    print('QP example:')
    Q = 2*matrix([ [2, .5], [.5, 1] ]) # (2x2)
    p = matrix([1.0, 1.0]) # (2x1)
    G = matrix([[-1.0,0.0],[0.0,-1.0]]) # (2x2)
    h = matrix([0.0,0.0]) # (2x1)
    A = matrix([1.0, 1.0], (1,2)) # (1x2)
    b = matrix(1.0) # (1x1)
    sol=solvers.qp(Q, p, G, h, A, b)
    print(sol['x'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVXOPT example')
    program_list = ['LP', 'QP']
    parser.add_argument('--program', type=str,
        help="Specify a program. Available programs are: %s"%program_list
    )
    args = parser.parse_args()

    if args.program == 'LP':
        run_linear_program()
    elif args.program == 'QP':
        run_quadratic_program()
    else:
        raise ValueError('Specify a program. Available programs are: %s'%program_list)