import numpy as np

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

# |0> in the tetrahedron basis
L_zero = 1/2 * np.kron(np.kron(Z_zero,Z_one) - np.kron(Z_one,Z_zero),np.kron(Z_zero,Z_one) - np.kron(Z_one,Z_zero))
#L_zero = L_zero/np.linalg.norm(L_zero)

# |1> in the tetrahedron basis
L_one = 1/np.sqrt(3) * (
         kronlist([Z_one,Z_one,Z_zero,Z_zero]) + kronlist([Z_zero,Z_zero,Z_one,Z_one]) -
         1/2 * np.kron(np.kron(Z_zero,Z_one) + np.kron(Z_one,Z_zero),np.kron(Z_zero,Z_one) + np.kron(Z_one,Z_zero))
        )

TetQubit = lambda theta, phi: np.cos(theta/2)*L_zero + np.sin(theta/2)*np.exp(1j*phi)*L_one

def main():
    print(np.kron(Z_zero, Z_one))

if __name__ == "__main__":
    main()