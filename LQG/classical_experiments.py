import numpy as np

def str_to_state(info):
    ''' take a bitstring+amplitude and transform it into a quantum state vector '''
    amplitude, bitstring = info
    state = np.zeros(2**len(bitstring))
    state[int(bitstring,2)] = amplitude
    return state

def create_state(superpos_list):
    ''' list of bitstring/amplitudes that are in superposition 
        normalize after
    '''
    quant_state = np.sum(str_to_state(state) for state in superpos_list)
    #return quant_state / np.linalg.norm(quant_state)
    return quant_state


S = create_state([(1/np.sqrt(2),'01'),(-1/np.sqrt(2),'10')])
TPLUS = str_to_state([1,'00'])
TMINUS = str_to_state([1,'11'])
TZERO = create_state([(1/np.sqrt(2),'01'),(1/np.sqrt(2),'10')])

IntertwinerZero = np.kron(S,S)
IntertwinerOne = 1/np.sqrt(3)*(np.kron(TPLUS,TMINUS) + np.kron(TMINUS,TPLUS) - np.kron(TZERO,TZERO))
