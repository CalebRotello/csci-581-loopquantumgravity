'''
    Circuits to prepare the tetrahedron on a QC
'''
import cirq
import numpy as np
import LQG.tetrahedra_num as tn

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

def TetStatePrep(qubits, angles):#theta, phi):
    ''' turn 4 qubits into a quantum tetrahedra with a circuit '''
    assert(len(qubits)==4)
    theta = angles[0]
    phi = angles[1]
    U = UGate(theta,phi)
    CV = CVGate(theta,phi)
    yield [
     cirq.H(qubits[0]),
     U.on(qubits[1]),
     CV.on(qubits[1],qubits[2]),
     cirq.X(qubits[2]),
     cirq.CNOT(qubits[0],qubits[1]),
     cirq.CNOT(qubits[2],qubits[3]),
     cirq.CNOT(qubits[1],qubits[2]),
     cirq.CNOT(qubits[0],qubits[3]),
    ]


#def TetStatePrep(qubits, angles):
#    ''' different parameters to simplify it '''
#    return TetStatePrep(qubits, angles[0], angles[1])

def HermTet(qubits, angles):
    assert(len(qubits)==4)
    theta = angles[0]
    phi = angles[1]
    U = UGate(theta,phi)
    CV = CVGate(theta,phi)
    yield [
     cirq.CNOT(qubits[0],qubits[1]),
     cirq.CNOT(qubits[2],qubits[3]),
     cirq.CNOT(qubits[1],qubits[2]),
     #cirq.CNOT(qubits[1],qubits[3]),
     cirq.CNOT(qubits[0],qubits[3]),
     cirq.X(qubits[2]),
     cirq.H(qubits[0])
    ]

def apply(gate, qubits, pairs):
    for p in pairs:
        yield gate(qubits[p[0]],qubits[p[1]])

LeftTet = lambda qubits: TetStatePrep(qubits, tn.bloch['left'])
RightTet = lambda qubits: TetStatePrep(qubits, tn.bloch['right'])
PlusTet = lambda qubits: TetStatePrep(qubits, tn.bloch['plus'])
MinusTet = lambda qubits: TetStatePrep(qubits, tn.bloch['minus'])
OneTet = lambda qubits: TetStatePrep(qubits, tn.bloch['one'])

def ZeroTet(qubits):
    assert(len(qubits)==4)
    yield [
     cirq.X.on_each(qubits),
     cirq.H.on_each(qubits[::2]),
     cirq.CNOT(qubits[0],qubits[1]),
     cirq.CNOT(qubits[2],qubits[3])
    ]


def final_state(circuit,ket=False):
    ''' return the final state vector of a circuit '''
    sim = cirq.Simulator()
    C = cirq.Circuit(circuit)
    if not ket:
        experiment = sim.simulate(C).final_state_vector
    else:
        experiment = sim.simulate(C)
    return experiment


def entangle_tets(qubits, pairs):
    ''' given a list of pairs (source->sink) 
    '''
    paired = set() # make sure we don't double-pair
    for p in pairs:
        for itm in p:
            assert(itm not in paired)
            paired.add(itm)
        yield cirq.CNOT(qubits[p[0]],qubits[p[1]])
        yield cirq.H(qubits[p[0]])
        yield cirq.X.on_each((qubits[p[0]],qubits[p[1]]))

def hist_to_wavefn(hist,N,samples):
    ''' transform a histogram into the wavefunction
    '''
    wavefn = np.zeros(2**N)
    for key,item in hist.items():
        wavefn[key] = item/samples
    return wavefn

#def to_iswap_gateset(circuit):


