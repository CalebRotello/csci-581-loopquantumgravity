'''
    Circuits to prepare the tetrahedron on a QC
'''
import cirq
import numpy as np

# A|1010> + A|0101> c2
A = lambda theta, phi: 1/np.sqrt(2)*(np.cos(theta/2)-1/np.sqrt(3)*np.exp(1j*phi)*np.sin(theta/2))
# B|0110> - B|1001> c3
B = lambda theta, phi: -1/np.sqrt(2)*(np.cos(theta/2)+1/np.sqrt(3)*np.exp(1j*phi)*np.sin(theta/2))
# C|0011> + C|1100> c1
C = lambda theta, phi: np.sqrt(2/3)*np.exp(1j*phi)*np.sin(theta/2)

class UGate(cirq.Gate):
    def __init__(self, theta, phi):
        super(UGate, self)
        self.theta = theta
        self.phi = phi
    
    def _num_qubits_(self):
        return 1
    
    def _unitary_(self):
        ant = np.sqrt(np.abs(A(self.theta,self.phi))**2 + np.abs(B(self.theta,self.phi))**2)
        dag = C(self.theta,self.phi)
        return np.array([
            [dag, ant],
            [-ant, np.conj(dag)]    
        ])

    def _circuit_diagram_info_(self, args):
        return f"U({self.theta:.2f},{self.phi:.2f})"

class CVGate(cirq.TwoQubitGate):
    def __init__(self, theta, phi):
        super(CVGate, self)
        self.theta = theta 
        self.phi = phi
    
    def _unitary_(self):
        dist = np.sqrt(np.abs(A(self.theta,self.phi))**2 + np.abs(B(self.theta,self.phi))**2)
        diag = A(self.theta,self.phi)/dist
        ant = B(self.theta,self.phi)/dist
        return np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., -diag, np.conj(ant)],
            [0., 0., -ant, -np.conj(diag)]
        ])

    def _circuit_diagram_info_(self, args):
        return "CV", f"V({self.theta:.2f},{self.phi:.2f})"

def TetStatePrep(qubits, theta, phi):
    ''' turn 4 qubits into a quantum tetrahedra with a circuit '''
    assert(len(qubits)==4)
    U = UGate(theta,phi)
    CV = CVGate(theta,phi)
    yield [cirq.H(qubits[0]),
     U.on(qubits[1]),
     CV.on(qubits[1],qubits[2]),
     cirq.X(qubits[2]),
     cirq.CNOT(qubits[0],qubits[1]),
     cirq.CNOT(qubits[2],qubits[3]),
     cirq.CNOT(qubits[0],qubits[3]),
     cirq.CNOT(qubits[1],qubits[2]),
    ]


LeftTet = lambda qubits: TetStatePrep(qubits, np.pi/2,np.pi/2)
RightTet = lambda qubits: TetStatePrep(qubits, np.pi/2,-np.pi/2)
PlusTet = lambda qubits: TetStatePrep(qubits, np.pi/2,0)
MinusTet = lambda qubits: TetStatePrep(qubits, -np.pi/2,0)
ZeroTet = lambda qubits: TetStatePrep(qubits, 0,0)
OneTet = lambda qubits: TetStatePrep(qubits, np.pi,0)


def final_state(circuit):
    ''' return the final state vector of a circuit '''
    sim = cirq.Simulator()
    C = cirq.Circuit(circuit)
    experiment = sim.simulate(C).final_state_vector
    return experiment