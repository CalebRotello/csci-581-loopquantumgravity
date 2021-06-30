import cirq

binformat = lambda qubits: '{0:0' + str(qubits) + 'b}'

def to_iswap(U):
    ''' circuit U converted to the iSWAP gateset '''
    circuit = cirq.Circuit(U)
    try:
        cirq.google.ConvertToSqrtIswapGates().optimize_circuit(circuit)
    except:
        cirq.google.ConvertToSycamoreGates().optimize_circuit(circuit)
        cirq.google.ConvertToSqrtIswapGates().optimize_circuit(circuit)
    return cirq.google.optimized_for_sycamore(circuit)

def parallelize(circuits):
    ''' take a list of circuits and combine seperate moments '''
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

qubit_cords = lambda qub: (qub.row, qub.col)

def remap_qubits(qubits):
    ''' map a list of qubits in a grid to their row major order '''
    sorted_qubits = sorted(qubits, key=lambda x: (x.row, x.col))
    return {i: qubits.index(elem) for i,elem in enumerate(sorted_qubits)}

def serialize_qubits(qubits):
    if type(qubits[0]) == cirq.LineQubit:
        return list(range(len(qubits)))
    elif type(qubits[0]) == cirq.GridQubit:
        return [(q.row, q.col) for q in qubits]


def remap(hist,qubit_map):
    ''' take a histogram of results and qubit map '''
    newhist = {}
    for key, value in hist.items():
        measured_state = binformat(len(qubit_map)).format(int(key))
        new_state = list(measured_state)
        for start,end in qubit_map.items():
            new_state[end] = measured_state[int(start)]
        newhist[int("".join(str(i) for i in new_state),2)] = value 
    return newhist

def filter_even(stateint,nqubs=4):
    ''' given int format state, accept if even number of ones '''
    if binformat(nqubs).format(stateint).count('1') % 2 == 0:
        return True
    return False