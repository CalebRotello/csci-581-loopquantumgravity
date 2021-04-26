''' Spin-foam amplitudes of loop quantum gravity
    project by Hakan Ayaz and Caleb Rotello
'''
import cirq
import json
import numpy as np
import pickle
import os

''' change run-specific data here 
    the simulating happens in run_circuit(), ll 369
'''
processor = ''
project_id = '' 
sim_mode = 'engine'
project_name ='HA_CR_AMPLITUDE_LQG'




''' All tetrahedra circuits code below 
    goto bottom for main()
'''

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
    circuit =  cirq.Circuit(
     cirq.H(qubits[0]),
     U,
     CV)
    if not chain:
        circuit.append([
         cirq.X(qubits[2]),
         cirq.CNOT(qubits[0],qubits[1]),
         cirq.CNOT(qubits[2],qubits[3]),
         cirq.CNOT(qubits[1],qubits[2]),
         cirq.CNOT(qubits[0],qubits[3]),
        ])
    else:
        circuit.append([
         cirq.X(qubits[2]),
         cirq.CNOT(qubits[1],qubits[2]),
         cirq.CNOT(qubits[2],qubits[3]),
         cirq.CNOT(qubits[1],qubits[2]),
         cirq.CNOT(qubits[2],qubits[3]),
         cirq.CNOT(qubits[0],qubits[1]),
         cirq.CNOT(qubits[1],qubits[2]),
         cirq.CNOT(qubits[2],qubits[3]),
        ])

    if gateset:
        circuit = to_gateset(circuit)

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

def apply(gate, pairs):
    ''' apply one 2-qubit gate to a series of qubits '''
    for p in pairs:
        yield gate(p[0],p[1])

bloch = {
    'zero':(0,0), 
    'one':(np.pi,0),
    'left':(np.pi/2,np.pi/2),
    'right':(np.pi/2,-np.pi/2),
    'plus':(np.pi/2,0),
    'minus':(-np.pi/2,0)
}

LeftTet  = lambda qubits, gateset=True: TetStatePrep(qubits, bloch['left'],  gateset=gateset)
RightTet = lambda qubits, gateset=True: TetStatePrep(qubits, bloch['right'], gateset=gateset)
PlusTet  = lambda qubits, gateset=True: TetStatePrep(qubits, bloch['plus'],  gateset=gateset)
MinusTet = lambda qubits, gateset=True: TetStatePrep(qubits, bloch['minus'], gateset=gateset)
OneTet   = lambda qubits, gateset=True, chain=False: TetStatePrep(qubits, bloch['one'],   gateset=gateset, chain=chain)

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

#def entangle_tets(qubits, pairs, exp=1):
def entangle_tets(qubits, exp=1, gateset=True):
    ''' given a list of pairs (source->sink) 
    '''
    paired = set() # make sure we don't double-pair
    cnot = cirq.CNotPowGate(exponent=exp)
    c = cirq.Circuit()
    for p in qubits:
        for itm in p:
            assert(itm not in paired)
            paired.add(itm)
        c.append([
            cnot.on(p[0],p[1]),
            cirq.H(p[0]),
            cirq.X.on_each((p[0],p[1]))
        ])

    if gateset:
        c = to_gateset(c)
    return c

def inv_epr_state(q1, q2, gateset=True):
    ''' return <00| + <11| '''
    circuit = cirq.Circuit(cirq.CNOT(q1,q2),cirq.H(q1))
    if gateset:
        circuit = to_gateset(circuit)
    return circuit

def measurement_circuit(indices):
    relevant_qubits = []
    if type(indices[0]) == tuple:
        relevant_qubits = [qb(a[0],a[1]) for a in indices]
    else:
        relevant_qubits = indices
    return cirq.Circuit(cirq.measure(*relevant_qubits, key='z'))

def parallel_init(tets):
    circuit = cirq.Circuit()
    for i,_ in enumerate(tets[0]):
        m = []
        for t in tets:
            m.append(t[i])
        circuit.append(cirq.Moment(m))
    return circuit


''' submission '''
qb = lambda a,b: cirq.GridQubit(a,b)
SYC = cirq.google.devices.Sycamore
to_iswap = cirq.google.ConvertToSqrtIswapGates()

def main():
    results_dict = {}

    # zero and one state preparation & fidelity (1000) 
    qubs = [qb(4,7),qb(3,7),qb(3,6),qb(4,6)]
    # 1000
    zero_fidelity_circuit = cirq.Circuit(ZeroTet(qubs),device=SYC)
    result = run_circuit(zero_fidelity_circuit + measurement_circuit(qubs), 1000)
    results_dict['zero_fidelity'] = result.histogram(key='z')

    # 1000
    one_fidelity_circuit = cirq.Circuit(OneTet(qubs),device=SYC)
    result = run_circuit(one_fidelity_circuit + measurement_circuit(qubs), 1000)
    results_dict['one_fidelity'] = result.histogram(key='z')
    
    # single tet zero & one (10000)
    pairs = [(qubs[1],qubs[2]),(qubs[0],qubs[3])]
    # 5000
    zero_transition = cirq.Circuit(ZeroTet(qubs),entangle_tets(pairs),device=SYC)
    for i  in range(5):
        result = run_circuit(zero_transition + measurement_circuit(qubs), 1000)
        results_dict['zero_transition_{}'.format(i)] = result.histogram(key='z')

    # 5000
    one_transition = cirq.Circuit(OneTet(qubs),entangle_tets(pairs),device=SYC)
    for i in range(5):
        result = run_circuit(one_transition + measurement_circuit(qubs), 1000)
        results_dict['one_transition_{}'.format(i)] = result.histogram(key='z')

    # 12_000

    # dipole 
    zerotwo_state = lambda qubs: parallel_init([cirq.Circuit(ZeroTet(qubs[i*4:i*4+4]) for i in range(2))])
    onetwo_state = lambda qubs: parallel_init([cirq.Circuit(OneTet(qubs[i*4:i*4+4]) for i in range(2))])
    # |00> 04152637
    qubs = [qb(3,6),qb(3,5),qb(4,7),qb(4,8),
            qb(4,6),qb(4,5),qb(5,7),qb(5,8)]
    pairs = [(qubs[0],qubs[4]),(qubs[1],qubs[5]),(qubs[2],qubs[6]),(qubs[3],qubs[7])]
    # 100_000
    zerotwo_transition = cirq.Circuit(zerotwo_state(qubs),entangle_tets(pairs),device=SYC)
    for i in range(10):
        result = run_circuit(zerotwo_transition + measurement_circuit(qubs), 10_000)
        results_dict['zero_two_transition_04152637_{}'.format(i)] = result.histogram(key='z')

    # |11> 04152637
    qubs = [qb(4,7),qb(4,6),qb(3,6),qb(3,7),
            qb(4,9),qb(4,8),qb(3,8),qb(3,9)]
    # 100_000
    onetwo_transition = cirq.Circuit(onetwo_state(qubs),device=SYC)
    append_circuit = cirq.Circuit(
        cirq.Moment([cirq.SWAP(qubs[5],qubs[0]),cirq.SWAP(qubs[3],qubs[6])]),
        entangle_tets([(qubs[3],qubs[2]),(qubs[1],qubs[0]),(qubs[6],qubs[7]),(qubs[5],qubs[4])])
    )
    to_iswap.optimize_circuit(append_circuit)
    onetwo_transition.append(append_circuit)
    for i in range(10):
        result = run_circuit(onetwo_transition + measurement_circuit(qubs), 10_000)
        results_dict['one_two_transition_04152637_{}'.format(2)] = result.histogram(key='z')


    # |00> 04172536
    qubs = [qb(3,5),qb(3,6),qb(5,6),qb(5,7),
            qb(4,5),qb(5,5),qb(4,7),qb(4,6)] 
    pairs = [(qubs[0],qubs[4]),(qubs[1],qubs[7]),(qubs[2],qubs[5]),(qubs[3],qubs[6])]
    # 100_000
    zerotwo_transition = cirq.Circuit(zerotwo_state(qubs),entangle_tets(pairs),device=SYC)
    for i in range(10):
        result = run_circuit(zerotwo_transition + measurement_circuit(qubs), 10_000)
        results_dict['zero_two_transition_04172536_{}'.format(i)] = result.histogram(key='z')

    # |11> with this network 07152634
    qubs = [qb(4,7),qb(4,6),qb(3,6),qb(3,7),
            qb(4,9),qb(4,8),qb(3,8),qb(3,9)]
    # 100_000
    onetwo_transition = cirq.Circuit(onetwo_state(qubs),device=SYC)
    append_circuit = cirq.Circuit([
        cirq.SWAP(qubs[0],qubs[3]),
        cirq.Moment(cirq.SWAP(qubs[3],qubs[6]),cirq.SWAP(qubs[0],qubs[5])),
        entangle_tets([(qubs[2],qubs[3]),(qubs[1],qubs[0]),(qubs[6],qubs[7]),(qubs[5],qubs[4])])
    ])
    to_iswap.optimize_circuit(append_circuit)
    onetwo_transition.append(append_circuit)
    # 0 -> 2, 3 -> 1, 2 -> 3, 1 -> 0, 4 <-> 5, 7 <-> 6
    for i in range(10):
        result = run_circuit(onetwo_transition + measurement_circuit(qubs), 10_000)
        results_dict['onetwo_transition_07152634_{}'.format(i)] = result.histogram(key='z')

    # 412_000

    # 4-simplex 
    qubs = [qb(3,7),qb(4,7),qb(2,4),qb(1,4),
            qb(1,5),qb(1,6),qb(3,5),qb(3,6),
            qb(2,6),qb(2,5),qb(4,4),qb(4,3),
            qb(3,3),qb(3,4),qb(4,6),qb(5,6),
            qb(5,5),qb(5,4),qb(1,7),qb(2,7)]
    pairs = [(qubs[19],qubs[0]),(qubs[18],qubs[5]),(qubs[14],qubs[1]),(qubs[6],qubs[13]),
            (qubs[2],qubs[9]),(qubs[10],qubs[17]),(qubs[3],qubs[4]),(qubs[7],qubs[8]),
            (qubs[11],qubs[12]),(qubs[15],qubs[16])]
    ## 1_500_000
    zerofive_state = lambda qubs: parallel_init([cirq.Circuit(ZeroTet(qubs[i*4:i*4+4])) for i in range(5)])
    zerofive_transition = cirq.Circuit(zerofive_state(qubs),entangle_tets(pairs),device=SYC)
    for i in range(15):
        result = run_circuit(zerofive_transition + measurement_circuit(qubs), 100_000)
        results_dict['zero_five_transition_{}'.format(i)] = result.histogram(key='z')

    # 1_912_000

    # maybe worth tying gluing two simplices?
    qubs = [qb(3,5),qb(4,5),qb(0,6),qb(0,5),
            qb(1,5),qb(1,4),qb(2,8),qb(2,7),
            qb(1,7),qb(1,6),qb(4,8),qb(4,9),
            qb(3,9),qb(3,8),qb(4,6),qb(3,6),
            qb(3,7),qb(4,7),qb(2,4),qb(3,4),
            
            qb(5,4),qb(4,4),qb(7,6),qb(6,6),
            qb(6,7),qb(5,7),qb(8,3),qb(8,4),
            qb(8,5),qb(7,5),qb(5,2),qb(6,2),
            qb(6,3),qb(7,3),qb(4,3),qb(3,3),
            qb(3,2),qb(4,2),qb(5,6),qb(5,5)]
    pairs = [(qubs[19],qubs[0]),(qubs[18],qubs[5]),
             (qubs[14],qubs[1]),(qubs[6],qubs[13]),
             (qubs[2],qubs[9]),(qubs[10],qubs[17]),
             (qubs[3],qubs[4]),(qubs[7],qubs[8]),
             (qubs[11],qubs[12]),(qubs[15],qubs[16]),

             (qubs[39],qubs[20]),(qubs[38],qubs[25]),
             (qubs[34],qubs[21]),(qubs[26],qubs[33]),
             (qubs[22],qubs[29]),(qubs[30],qubs[37]),
             (qubs[23],qubs[24]),(qubs[27],qubs[28]),
             (qubs[31],qubs[32]),(qubs[35],qubs[36]),
             ]

    # 1_500_000
    zeroten_state = lambda qubs: parallel_init([cirq.Circuit(ZeroTet(qubs[i*4:i*4+4])) for i in range(10)])
    zeroten_transition = cirq.Circuit(zeroten_state(qubs),entangle_tets(pairs),device=SYC)
    # entangle 38,17, 39,16, 37,19, 36,18 07 16 24 35
    append_circuit = cirq.Circuit(
        cirq.Moment([cirq.SWAP(qubs[36],qubs[35]),cirq.SWAP(qubs[37],qubs[34]),cirq.SWAP(qubs[19],qubs[21]),
                     cirq.SWAP(qubs[16],qubs[15]),cirq.SWAP(qubs[38],qubs[25])]), 
                     # 36 <-> 35, 37 <-> 34, 19 <-> 21, 16 <-> 15, 38 <-> 25
        cirq.Moment([cirq.SWAP(qubs[15],qubs[14]),cirq.SWAP(qubs[39],qubs[38]),cirq.SWAP(qubs[18],qubs[19])]),
                     # 15 <-> 14, 39 <-> 38, 18 <-> 19
        apply(inv_epr_state, 
                          [(qubs[35],qubs[19]),(qubs[34],qubs[21]),(qubs[17],qubs[25]),(qubs[14],qubs[38])])
    )
    to_iswap.optimize_circuit(append_circuit)
    zeroten_transition.append(append_circuit)
    for i in range(15):
        result = run_circuit(zeroten_transition + measurement_circuit(qubs), 100_000)
        results_dict['zero_ten_transition_{}'.format(i)] = result.histogram(key='z')

    # 3_412_000

    # serialization
    root_dir = os.getcwd()
    top_dir = project_name
    dir_path = os.path.join(root_dir, top_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(dir_path + '/lqg_ha_cr_measurements.json','w') as f:
        json.dump(results_dict,f)
        

def run_circuit(circuit: cirq.Circuit, reps: int) -> cirq.TrialResult:
    if sim_mode == 'engine':
        engine = cirq.google.Engine(project_id=project_id)
        print("Uploading program and scheduling job on Quantum Engine...\n")
        results = engine.run(
            program=circuit,
            repetitions=reps,
            processor_ids=[SYC],
            gate_set=cirq.google.SQRT_ISWAP_GATESET
        )
    else:
        results = cirq.Simulator().run(circuit, repetitions=reps)
    return results

if __name__ == '__main__':
    main()