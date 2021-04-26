'''
    Circuits to prepare the tetrahedron on a QC
'''
import cirq
import numpy as np
import tetrahedra_num as tn

project_name ='HA_CR_AMPLITUDE_LQG'

binformat = lambda qubits: '{0:0' + str(qubits) + 'b}'

# A|1010> + A|0101> c2
A = lambda theta, phi: 1/np.sqrt(2)*(np.cos(theta/2)-1/np.sqrt(3)*np.exp(1j*phi)*np.sin(theta/2))
# B|0110> - B|1001> c3
B = lambda theta, phi: -1/np.sqrt(2)*(np.cos(theta/2)+1/np.sqrt(3)*np.exp(1j*phi)*np.sin(theta/2))
# C|0011> + C|1100> c1
C = lambda theta, phi: np.sqrt(2/3)*np.exp(1j*phi)*np.sin(theta/2)

dist = lambda a,b: np.sqrt(np.abs(a)**2 + np.abs(b)**2)
UUnitary = lambda theta,phi: [[C(theta,phi), dist(A(theta,phi),B(theta,phi))], 
                              [-dist(A(theta,phi),B(theta,phi)), np.conj(C(theta,phi))]]
VUnitary = lambda theta,phi: [[-A(theta,phi)/dist(A(theta,phi),B(theta,phi)),np.conj(B(theta,phi))/dist(A(theta,phi),B(theta,phi))], 
                              [-B(theta,phi)/dist(A(theta,phi),B(theta,phi)),-np.conj(A(theta,phi))/dist(A(theta,phi),B(theta,phi))]]

class UGate(cirq.Gate):
    def __init__(self, theta, phi):
        super(UGate, self)
        self.theta = theta
        self.phi = phi
    
    def _num_qubits_(self):
        return 1
    
    def _unitary_(self):
        return np.array(UUnitary(self.theta, self.phi))

    def _circuit_diagram_info_(self, args):
        return f"U({self.theta:.2f},{self.phi:.2f})"

class CVGate(cirq.TwoQubitGate):
    def __init__(self, theta, phi):
        super(CVGate, self)
        self.theta = theta 
        self.phi = phi
    
    def _unitary_(self):
        mtx = [[1.,0.,0.,0.],[0.,1.,0.,0.]]
        v = VUnitary(self.theta,self.phi)
        a = [0.,0.]
        a.extend(v[0])
        mtx.append(a)
        a = [0.,0.]
        a.extend(v[1])
        mtx.append(a)
        return np.array(mtx)

    def _circuit_diagram_info_(self, args):
        return "CV", f"V({self.theta:.2f},{self.phi:.2f})"

def TetStatePrep(qubits, angles, gateset=True, chain=False):
    ''' turn 4 qubits into a quantum tetrahedra with a circuit '''
    assert(len(qubits)==4)
    theta = angles[0]
    phi = angles[1]
    U = UGate(theta,phi).on(qubits[1])
    CV = CVGate(theta,phi).on(qubits[1],qubits[2])
    circuit = cirq.Circuit(
     cirq.H(qubits[0]),
     U,
     CV,
     cirq.X(qubits[2]))
    if chain:
        circuit.append([
            cirq.CNOT(qubits[1],qubits[2]),
            cirq.CNOT(qubits[2],qubits[3]),
            cirq.CNOT(qubits[1],qubits[2]),
            cirq.CNOT(qubits[2],qubits[3]),
            cirq.CNOT(qubits[0],qubits[1]),
            cirq.CNOT(qubits[1],qubits[2]),
            cirq.CNOT(qubits[2],qubits[3]),
        ])
    else:
        circuit.append([
            cirq.CNOT(qubits[0],qubits[1]),
            cirq.CNOT(qubits[2],qubits[3]),
            cirq.CNOT(qubits[1],qubits[2]),
            cirq.CNOT(qubits[0],qubits[3]),
        ])

    if gateset:
        circuit = to_gateset(circuit)
        circuit = cirq.google.optimized_for_sycamore(circuit)

    return circuit



def to_gateset(U):
    ''' U circuit converted to the iswap gates
    '''
    circuit = cirq.Circuit(U)
    try:
        cirq.google.ConvertToSqrtIswapGates().optimize_circuit(circuit)
    except:
        cirq.google.ConvertToSycamoreGates().optimize_circuit(circuit)
        cirq.google.ConvertToSqrtIswapGates().optimize_circuit(circuit)
    return cirq.google.optimized_for_sycamore(circuit)

def apply(gate, qubits, pairs):
    ''' apply one 2-qubit gate to a series of qubits '''
    for p in pairs:
        yield gate(qubits[p[0]],qubits[p[1]])

bloch = {
    'zero':(0,0), 
    'one':(np.pi,0),
    'left':(np.pi/2,np.pi/2),
    'right':(np.pi/2,-np.pi/2),
    'plus':(np.pi/2,0),
    'minus':(-np.pi/2,0)
}

LeftTet  = lambda qubits, gateset=True, chain=False: TetStatePrep(qubits, bloch['left'],  gateset=gateset,chain=chain)
RightTet = lambda qubits, gateset=True, chain=False: TetStatePrep(qubits, bloch['right'], gateset=gateset,chain=chain)
PlusTet  = lambda qubits, gateset=True, chain=False: TetStatePrep(qubits, bloch['plus'],  gateset=gateset,chain=chain)
MinusTet = lambda qubits, gateset=True, chain=False: TetStatePrep(qubits, bloch['minus'], gateset=gateset,chain=chain)
OneTet   = lambda qubits, gateset=True, chain=False: TetStatePrep(qubits, bloch['one'],      gateset=gateset,chain=chain)

def ZeroTet(qubits, gateset=True):
    assert(len(qubits)==4)
    circuit = cirq.Circuit(
     cirq.X.on_each(qubits),
     cirq.H.on_each(qubits[::2]),
     cirq.CNOT(qubits[0],qubits[1]),
     cirq.CNOT(qubits[2],qubits[3]),
    )
    if gateset:
        circuit = to_gateset(circuit)
    yield circuit


def final_state(circuit,ket=False):
    ''' return the final state vector of a circuit 
        for testing
    '''
    sim = cirq.Simulator()
    C = cirq.Circuit(circuit)
    if not ket:
        experiment = sim.simulate(C).final_state_vector
    else:
        experiment = sim.simulate(C)
    return experiment


def entangle_tets(pairs, exp=1, gateset=True):
    ''' given a list of pairs (source->sink) 
    '''
    paired = set() # make sure we don't double-pair
    cnot = cirq.CNotPowGate(exponent=exp)
    circuit = cirq.Circuit()
    for p in pairs:
        for itm in p:
            assert(itm not in paired)
            paired.add(itm)
        circuit.append([
            cnot.on(p[0],p[1]),
            cirq.H(p[0]),
            cirq.X.on_each(p[0],p[1]),
        ])
    if gateset:
        circuit = to_gateset(circuit)
    return circuit

def noisify(circuit,x):
    ''' take a circuit and add depolarizing noise 
        for testing
    '''
    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(x))
    noisy_circuit = cirq.Circuit()
    sysqubits = sorted(circuit.all_qubits())
    for moment in circuit:
        noisy_circuit.append(noise.noisy_moment(moment, sysqubits))
    return noisy_circuit

def measurement_circuit(qubits):
    return cirq.Circuit(cirq.measure(*qubits, key='z')) 

def parallel_init(tets):
    circuit = cirq.Circuit()
    for i,_ in enumerate(tets[0]):
        m = []
        for t in tets:
            m.append(t[i])
        circuit.append(cirq.Moment(m))
    return circuit
    
