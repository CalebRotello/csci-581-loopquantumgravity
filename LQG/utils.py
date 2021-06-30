''' Created by Caleb Rotello, June 2021
    This file provides several utility functions for LQG experiments. The code is less focused on the scientific endeavor,
    and more on increasing the ease of writing the code for the scientific endeavor.
'''

import cirq

# take the size of the system as a parameter, then .format(state_as_int)
binformat = lambda qubit_count: '{0:0' + str(qubit_count) + 'b}'

def to_iswap(U):
    ''' circuit U converted to the iSWAP gateset 
        :param Circuit U:
    '''
    circuit = cirq.Circuit(U)
    try:
        cirq.google.ConvertToSqrtIswapGates().optimize_circuit(circuit)
    except:
        cirq.google.ConvertToSycamoreGates().optimize_circuit(circuit)
        cirq.google.ConvertToSqrtIswapGates().optimize_circuit(circuit)
    return cirq.google.optimized_for_sycamore(circuit)

def parallelize(circuits):
    ''' Take a list of circuits and combine seperate moments in order to parallelize
        :param list(Circuit) or Circuit circuits: 
    '''
    if type(circuits) != list:
        circuits = [circuits]
    circuit = cirq.Circuit()
    for i,_ in enumerate(circuits[0]):
        m = []
        for t in circuits:
            m.append(t[i])
        circuit.append(cirq.Moment(m))
    return circuit

# shorthand for GridQubit
qubit = lambda a, b: cirq.GridQubit(a,b) 

def remap_qubits(qubits):
    ''' Map a list of qubits in a grid to their row major order '''
    sorted_qubits = sorted(qubits, key=lambda x: (x.row, x.col))
    return {i: qubits.index(elem) for i,elem in enumerate(sorted_qubits)}

def serialize_qubits(qubits):
    ''' We save qubits as a list of their coordinates
        :param list(Qubit) qubits: A list of qubits used in an experiment
    '''
    if type(qubits[0]) == cirq.LineQubit:
        return list(range(len(qubits)))
    elif type(qubits[0]) == cirq.GridQubit:
        return [(q.row, q.col) for q in qubits]


def remap(hist,qubit_map):
    ''' Take a histogram of results and qubit map and remap the results
        :param dict(str,int) hist: A dict of the state_int to occurences of that measured state. The keys are strings, because JSON
        :param dict(int,int) qubit_map: A dict of the measured index of a qubit to the desired index of the qubit
    '''
    newhist = {}
    for key, value in hist.items():
        measured_state = binformat(len(qubit_map)).format(int(key))
        new_state = list(measured_state)
        for start,end in qubit_map.items():
            new_state[end] = measured_state[int(start)]
        newhist[int("".join(str(i) for i in new_state),2)] = value 
    return newhist

def filter_even(stateint,nqubs=4):
    ''' Given state_int , accept if even number of ones '''
    if binformat(nqubs).format(stateint).count('1') % 2 == 0:
        return True
    return False