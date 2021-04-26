'''
    Various numerical constants/matrices/vectors to represent quantum tetrahedra
'''

import numpy as np

binformat = lambda n: '{0:0' + str(n) + 'b}'

Z_zero = [1.,0.]
Z_one = [0.,1.]

def kronlist(states):
    ''' given a list of matrices, [A, B, C, ...] perform the kronecker product across all of them 
        i.e. [I,I,X,I] is X_2
    '''
    state = np.kron(states[0],states[1])
    for s in states[2:]:
        state = np.kron(state,s)
    return state

def overlap(s1, s2):
    ''' take the squared overlap of two vectors '''
    return np.abs(np.dot(s1,s2))**2

def link_faces(f1, f2, dim):
    ''' entangle two tetrahedra faces with a Bell state 1/sqrt(2) |0_i 1_j> + |1_i 0_j>
    '''
    state = np.zeros(2**dim)
    state[2**f1-1] = 1/np.sqrt(2)
    state[2**f2-1] = 1/np.sqrt(2)
    return state
    

# |0> in the tetrahedron basis
L_zero = 1/2 * np.kron(np.kron(Z_zero,Z_one) - np.kron(Z_one,Z_zero),np.kron(Z_zero,Z_one) - np.kron(Z_one,Z_zero))

# |1> in the tetrahedron basis
L_one = 1/np.sqrt(3) * (
         kronlist([Z_one,Z_one,Z_zero,Z_zero]) + kronlist([Z_zero,Z_zero,Z_one,Z_one]) -
         1/2 * np.kron(np.kron(Z_zero,Z_one) + np.kron(Z_one,Z_zero),np.kron(Z_zero,Z_one) + np.kron(Z_one,Z_zero))
        )

# |+> 
L_plus = (L_zero + L_one)/np.sqrt(2)

# |-> 
L_minus = (L_zero - L_one)/np.sqrt(2)

# |left> (counter-clockwise), eigenstate of the V operator
L_left = (L_zero - 1j*L_one)/np.sqrt(2)

# |right> (clockwise), eigenstate of the V operator
L_right = (L_zero + 1j*L_one)/np.sqrt(2)

# true zero |0000>
T_zero = L_zero*0
T_zero[0] = 1

TetVec = lambda theta, phi: np.cos(theta/2)*L_zero + np.sin(theta/2)*np.exp(1j*phi)*L_one

