''' 
    Create experiments 
'''

import cirq
import circuits as tc 


class Experiment():
    ''' an experiment for quantum tetrahedra dynamics 

        qubits:     formatted with every 4 qubits representing one tetrahedra
        qubitmap:   a map of qubits to their row-major order
        circuit:    a circuit for the experiment
        pairs:      glued face indices
    '''

    def __init__(self, qubits, pairs, starting_states):
        ''' ctor. experiment
            qubits and pairs:   see above
            starting_states:    a list of quantum tetrahedra starting states 
        '''
        self.qubits = qubits
        self.qubitmap = qubit_map(qubits)
        self.pairs = pairs



def qubit_map(qubs):
    ''' map a list of qubits in a grid to their row majored order '''
    sortedqubs = sorted(qubs,key=lambda x: (x[0],x[1]))
    return {i: qubs.index(elem) for i,elem in enumerate(sortedqubs)}

