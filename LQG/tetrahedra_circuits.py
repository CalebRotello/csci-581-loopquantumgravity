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

def TetStatePrep(qubits, angles, gateset=True):
    ''' turn 4 qubits into a quantum tetrahedra with a circuit '''
    assert(len(qubits)==4)
    theta = angles[0]
    phi = angles[1]
    U = UGate(theta,phi).on(qubits[1])
    #if gateset:
    #    U = to_gateset(U)
    CV = CVGate(theta,phi).on(qubits[1],qubits[2])
    #if gateset:
    #    CV = gateset_CV(CV,qubits[1],qubits[2])
    circuit =  cirq.Circuit([
     cirq.H(qubits[0]),
     U,
     CV,
     cirq.X(qubits[2]),
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
#
#def gateset_CV(CV, q1, q2):
#    ''' CV gate needs to be converted to the google used gates
#    '''
#    circuit = cirq.Circuit(CV)
#    oplist = cirq.google.ConvertToSycamoreGates().convert(circuit)
#    circuit = cirq.Circuit(oplist)
#    k1 = circuit[1]
#    k2 = circuit[-1:]
#    circuit = cirq.Circuit(k1, cirq.CNOT(q1,q2), k2)
#    cirq.google.ConvertToSqrtIswapGates().optimize_circuit(circuit)
#    yield circuit

def apply(gate, qubits, pairs):
    ''' apply one 2-qubit gate to a series of qubits '''
    for p in pairs:
        yield gate(qubits[p[0]],qubits[p[1]])


LeftTet  = lambda qubits, gateset=True: TetStatePrep(qubits, tn.bloch['left'],  gateset=gateset)
RightTet = lambda qubits, gateset=True: TetStatePrep(qubits, tn.bloch['right'], gateset=gateset)
PlusTet  = lambda qubits, gateset=True: TetStatePrep(qubits, tn.bloch['plus'],  gateset=gateset)
MinusTet = lambda qubits, gateset=True: TetStatePrep(qubits, tn.bloch['minus'], gateset=gateset)
OneTet   = lambda qubits, gateset=True: TetStatePrep(qubits, tn.bloch['one'],   gateset=gateset)

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
    ''' return the final state vector of a circuit '''
    sim = cirq.Simulator()
    C = cirq.Circuit(circuit)
    if not ket:
        experiment = sim.simulate(C).final_state_vector
    else:
        experiment = sim.simulate(C)
    return experiment


def entangle_tets(qubits, pairs, exp=1):
    ''' given a list of pairs (source->sink) 
    '''
    paired = set() # make sure we don't double-pair
    cnot = cirq.CNotPowGate(exponent=exp)
    for p in pairs:
        for itm in p:
            assert(itm not in paired)
            paired.add(itm)
        yield cnot.on(qubits[p[0]],qubits[p[1]])
        yield cirq.H(qubits[p[0]])
        yield cirq.X.on_each((qubits[p[0]],qubits[p[1]]))

#def hist_to_wavefn(hist,N,samples):
#    ''' transform a histogram into the wavefunction
#    '''
#    wavefn = np.zeros(2**N)
#    for key,item in hist.items():
#        wavefn[key] = item/samples
#    return wavefn


def noisify(circuit,x):
    ''' take a circuit and add depolarizing noise
    '''
    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(x))
    noisy_circuit = cirq.Circuit()
    sysqubits = sorted(circuit.all_qubits())
    for moment in circuit:
        noisy_circuit.append(noise.noisy_moment(moment, sysqubits))
    return noisy_circuit


def post_select_even(hist,N):
    ''' post select data from a histogram knowing the states should have an even 
        number of 1's 
    '''
    newhist = {}
    newsamples = 0
    for key,item in hist.items():
        s = binformat(N).format(key)
        if s.count('1') % 2 == 0:
            newhist[key] = item
            newsamples += item
    return newhist, newsamples