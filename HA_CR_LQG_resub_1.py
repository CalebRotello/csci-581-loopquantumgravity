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
sim_mode = ''
project_name = 'HA_CR_AMPLITUDE_LQG_sub2'




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
OneTet   = lambda qubits, gateset=True, chain=False: TetStatePrep(qubits, bloch['one'],   gateset=gateset,chain=chain)

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

def final_state_simulation(circuit,runs):
    results = cirq.Simulator().run(cirq.Circuit(circuit),repetitions=runs)    
    return results


def entangle_tets(pairs, gateset=True):
    ''' given a list of pairs (source->sink) 
    '''
    paired = set() # make sure we don't double-pair
    #cnot = cirq.CNotPowGate(exponent=exp)
    circuit = cirq.Circuit()
    for p in pairs:
        for itm in p:
            assert(itm not in paired)
            paired.add(itm)
        circuit.append([
            #cnot.on(p[0],p[1]),
            cirq.CNOT(p[0],p[1]),
            cirq.H(p[0]),
            #cirq.X.on_each(p[0],p[1]),
            cirq.X(p[1])
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

def parallelize(circuits):
    circuit = cirq.Circuit()
    for i,_ in enumerate(circuits[0]):
        m = []
        for t in circuits:
            m.append(t[i])
        circuit.append(cirq.Moment(m))
    return circuit
    



''' submission '''
qb = lambda a,b: cirq.GridQubit(a,b)
SYC = cirq.google.devices.Sycamore
to_iswap = cirq.google.ConvertToSqrtIswapGates()

def main():
    # serialization
    root_dir = os.getcwd()
    top_dir = project_name
    dir_path = os.path.join(root_dir, top_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    #with open(dir_path + '/lqg_ha_cr_test_measurements.json','w') as f:
    #    json.dump(results_dict,f)

    # zero and one state preparation & fidelity (1000) 
    qubs = [qb(0,5),qb(0,6),qb(1,6),qb(1,5),
            qb(4,4),qb(4,5),qb(5,5),qb(5,4),
            qb(2,7),qb(2,8),qb(3,8),qb(3,7)
           ]
    
    # single tet zero & one (10000)
    pairs = [(qubs[1],qubs[2]),(qubs[0],qubs[3]),
             (qubs[5],qubs[6]),(qubs[4],qubs[7]),
             (qubs[9],qubs[10],(qubs[8],qubs[11]))
            ]

    ''' Zero Monopole Transition 
    '''
    # 5*1024
    zero_transition = cirq.Circuit(ZeroTet(qubs[:4]),entangle_tets(pairs[:4]),device=SYC)
    zero_transition = cirq.Circuit(ZeroTet(qubs[:4]),
                                   ZeroTet(qubs[4:8]),
                                   ZeroTet(qubs[8:]),
                                   entangle_tets(pairs),
                                   device=SYC)
    print('Running zero monopole')
    zero_monopole_dict = {}
    for i in range(5):
        result = run_circuit(zero_transition + measurement_circuit(qubs), 1024)
        zero_monopole_dict['zero_transition_1024_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/zero_monopole.json', 'w') as f:
        json.dump(zero_monopole_dict,f)
    print('Done zero monopole')

    ''' One Monopole Transition
    '''
    # 5*1024
    one_transition = cirq.Circuit([OneTet(qubs[i*4:i*4+4]) for i in range(3)],entangle_tets(pairs),device=SYC)
    print('Running one monoppole')
    one_monopole_dict = {}
    for i in range(5):
        result = run_circuit(one_transition + measurement_circuit(qubs), 1024)
        one_monopole_dict['one_transition_1024_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/one_monopole.json', 'w') as f:
        json.dump(one_monopole_dict,f)
    print('Done one monopole')

    # used 10240


    ''' Zero Dipole transition 
    '''
    zerotwo_state = lambda qubs: parallelize([cirq.Circuit(ZeroTet(qubs[i*4:i*4+4]) for i in range(2))])
    #onetwo_state = lambda qubs: parallelize([cirq.Circuit(OneTet(qubs[i*4:i*4+4]) for i in range(2))])
    # |00> 04152637
    qubs = [qb(2,7),qb(2,8),qb(0,5),qb(0,6),
            qb(3,7),qb(3,8),qb(1,5),qb(1,6),

            qb(2,4),qb(2,5),qb(4,4),qb(4,5),
            qb(3,4),qb(3,5),qb(5,4),qb(5,5),

            qb(6,4),qb(6,5),qb(4,7),qb(4,8),
            qb(7,4),qb(7,5),qb(5,7),qb(5,8),
    ]

    pairs = [(qubs[0],qubs[4]),(qubs[1],qubs[5]),(qubs[2],qubs[6]),(qubs[3],qubs[7]),
             (qubs[8],qubs[12]),(qubs[9],qubs[13]),(qubs[10],qubs[14]),(qubs[11],qubs[15]),
             (qubs[16],qubs[20]),(qubs[17],qubs[21]),(qubs[18],qubs[22]),(qubs[19],qubs[23])
    ]
    # 10_240
    zerotwo_transition = cirq.Circuit(zerotwo_state(qubs),entangle_tets(pairs),device=SYC)
    print('Running zero dipole')
    zero_dipole_dict = {}
    for i in range(10): 
        result = run_circuit(zerotwo_transition + measurement_circuit(qubs), 1_024) 
        zero_dipole_dict['zero_two_transition_04152637_10240_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/zero_dipole_04152637.json','w') as f:
        json.dump(zero_dipole_dict, f)
    print('Done zero dipole')


    # |00> 04172536
    qubs = [qb(3,5),qb(3,6),qb(5,6),qb(5,7),
            qb(4,5),qb(5,5),qb(4,7),qb(4,6),
            qb(0,5),qb(0,6),qb(2,6),qb(2,7),
            qb(1,5),qb(2,5),qb(1,7),qb(1,6),
            qb(7,5),qb(8,5),qb(7,3),qb(6,3),
            qb(7,4),qb(8,3),qb(6,4),qb(8,4),
    ]

    pairs = [(qubs[0],qubs[4]),(qubs[1],qubs[7]),(qubs[2],qubs[5]),(qubs[3],qubs[6]),
             (qubs[8],qubs[12]),(qubs[9],qubs[15]),(qubs[10],qubs[13]),(qubs[11],qubs[14]),
             (qubs[16],qubs[20]),(qubs[17],qubs[23]),(qubs[18],qubs[21]),(qubs[19],qubs[22])
    ]
    # 10_240
    zerotwo_transition = cirq.Circuit(zerotwo_state(qubs),entangle_tets(pairs),device=SYC)
    print('Running zero dipole 04172536')
    zero_dipole_dict = {}
    for i in range(10):
        result = run_circuit(zerotwo_transition + measurement_circuit(qubs), 1_024)
        zero_dipole_dict['zero_two_transition_04172536_10240_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/zero_dipole_04172536.json','w') as f:
        json.dump(zero_dipole_dict,f)
    print('Done zero dipole')


    ''' 4 Simplex
    '''
    # 4 simplex
    zerofive_state = lambda qubs: parallelize([cirq.Circuit(ZeroTet(qubs[i*4:i*4+4])) for i in range(5)])

    ''' 2 rings
    '''
    # 4-simplex 2 rings
    qubs = [
        qb(1,7),qb(2,7),qb(4,7),qb(5,7),
        qb(3,7),qb(3,6),qb(6,7),qb(6,6),
        qb(3,5),qb(3,4),qb(6,5),qb(6,4),
        qb(2,4),qb(2,5),qb(5,4),qb(5,5),
        qb(1,5),qb(1,6),qb(4,5),qb(4,6)
    ]
    pairs = [
        (qubs[1],qubs[4]),(qubs[5],qubs[8]),(qubs[9],qubs[12]),(qubs[13],qubs[16]),
        (qubs[17],qubs[0]),(qubs[3],qubs[6]),(qubs[7],qubs[10]),(qubs[11],qubs[14]),
        (qubs[15],qubs[18]),(qubs[19],qubs[2])
    ]
    zerofive_transition = cirq.Circuit(zerofive_state(qubs),entangle_tets(pairs),device=SYC)
    print('Running 4-simplex, 2 entanglement rings 1')
    zero_simplex_dict = {}
    for i in range(7):
        result = run_circuit(zerofive_transition + measurement_circuit(qubs), 102_400)
        zero_simplex_dict['zero_4simplex_2rings_102400_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/zero_4simplex_2rings_1.json','w') as f:
        json.dump(zero_simplex_dict,f)
    print('Done 4-simplex, 2 rings')

    # 2 rings (2)
    qubs = [
        qb(5,3),qb(5,4),qb(5,6),qb(5,7),
        qb(5,5),qb(4,5),qb(5,8),qb(4,8),
        qb(3,5),qb(2,5),qb(3,8),qb(2,8),
        qb(2,4),qb(3,4),qb(2,7),qb(3,7),
        qb(3,3),qb(4,3),qb(3,6),qb(4,6)
    ]
    pairs = [
        (qubs[1],qubs[4]),(qubs[5],qubs[8]),(qubs[9],qubs[12]),(qubs[13],qubs[16]),
        (qubs[17],qubs[0]),(qubs[3],qubs[6]),(qubs[7],qubs[10]),(qubs[11],qubs[14]),
        (qubs[15],qubs[18]),(qubs[19],qubs[2])
    ]
    zerofive_transition = cirq.Circuit(zerofive_state(qubs),entangle_tets(pairs),device=SYC)
    print('Running 4-simplex, 2 entanglement rings 2')
    zero_simplex_dict = {}
    for i in range(8):
        result = run_circuit(zerofive_transition + measurement_circuit(qubs), 102_400)
        zero_simplex_dict['zero_4simplex_2rings_102400_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/zero_4simplex_2rings_2.json','w') as f:
        json.dump(zero_simplex_dict,f)
    print('Done 4-simplex, 2 rings')



    # 4-simplex, 3 rings (1)
    qubs = [
        qb(1,7),qb(2,7),qb(2,5),qb(3,5),
        qb(3,4),qb(2,4),qb(5,4),qb(5,5),
        qb(1,4),qb(1,5),qb(5,6),qb(5,7),
        qb(3,7),qb(3,6),qb(4,7),qb(4,6),
        qb(2,6),qb(1,6),qb(4,5),qb(4,4)
    ]
    pairs = [
        (qubs[13],qubs[16]),(qubs[17],qubs[0]),(qubs[12],qubs[1]),(qubs[3],qubs[4]),(qubs[5],qubs[8]),
        (qubs[2],qubs[9]),(qubs[7],qubs[10]),(qubs[11],qubs[14]),(qubs[15],qubs[18]),(qubs[19],qubs[6])
    ]
    zerofive_transition = cirq.Circuit(zerofive_state(qubs),entangle_tets(pairs),device=SYC)
    print('Running 4-simplex, 3 entanglement rings 1')
    zero_simplex_dict = {}
    for i in range(7):
        result = run_circuit(zerofive_transition + measurement_circuit(qubs), 102_400)
        zero_simplex_dict['zero_4simplex_3rings_102400_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/zero_4simplex_3rings_1.json','w') as f:
        json.dump(zero_simplex_dict,f)
    print('Done 4-simplex, 3 entanglement rings')

    # 4-simplex, 3 rings (2)
    qubs = [
        qb(5,4),qb(5,5),qb(3,5),qb(3,6),
        qb(2,6),qb(2,5),qb(2,8),qb(3,8),
        qb(2,4),qb(3,4),qb(4,8),qb(5,8),
        qb(5,6),qb(4,6),qb(5,7),qb(4,7),
        qb(4,5),qb(4,4),qb(3,7),qb(2,7)
    ]
    pairs = [
        (qubs[13],qubs[16]),(qubs[17],qubs[0]),(qubs[12],qubs[1]),(qubs[3],qubs[4]),(qubs[5],qubs[8]),
        (qubs[2],qubs[9]),(qubs[7],qubs[10]),(qubs[11],qubs[14]),(qubs[15],qubs[18]),(qubs[19],qubs[6])
    ]
    zerofive_transition = cirq.Circuit(zerofive_state(qubs),entangle_tets(pairs),device=SYC)
    print('Running 4-simplex, 3 entanglement rings 2')
    zero_simplex_dict = {}
    for i in range(8):
        result = run_circuit(zerofive_transition + measurement_circuit(qubs), 102_400)
        zero_simplex_dict['zero_4simplex_3rings_102400_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/zero_4simplex_3rings_2.json','w') as f:
        json.dump(zero_simplex_dict,f)
    print('Done 4-simplex, 3 entanglement rings')


    # 4-simplex 1 ring IN PARALLEL
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

    zerofive_transition = cirq.Circuit(zerofive_state(qubs),entangle_tets(pairs),device=SYC)
    print('Running 4-simplex, 1 ring')
    zero_simplex_dict = {}
    for i in range(10):
        result = run_circuit(zerofive_transition + measurement_circuit(qubs), 102_400)
        zero_simplex_dict['zero_4simplex_1rings_102400_{}'.format(i)] = result.histogram(key='z')
    with open(dir_path + '/zero_4simplex_1rings.json','w') as f:
        json.dump(zero_simplex_dict, f)
    print('Done 4-simplex, 1 ring')

    #zerofive_transition = cirq.Circuit(zerofive_state(qubs),entangle_tets(pairs),device=SYC)
    ##print(cirq.Simulator().simulate(zerofive_transition).final_state_vector[0])
    #for i in range(15):
    #    result = run_circuit(zerofive_transition + measurement_circuit(qubs), 100_000)
    #    results_dict['zero_five_transition_{}'.format(i)] = result.histogram(key='z')

    # 1_912_000

    # maybe worth trying gluing two simplices?
    #qubs = [qb(3,5),qb(4,5),qb(0,6),qb(0,5),
    #        qb(1,5),qb(1,4),qb(2,8),qb(2,7),
    #        qb(1,7),qb(1,6),qb(4,8),qb(4,9),
    #        qb(3,9),qb(3,8),qb(4,6),qb(3,6),
    #        qb(3,7),qb(4,7),qb(2,4),qb(3,4),
    #        
    #        qb(5,4),qb(4,4),qb(7,6),qb(6,6),
    #        qb(6,7),qb(5,7),qb(8,3),qb(8,4),
    #        qb(8,5),qb(7,5),qb(5,2),qb(6,2),
    #        qb(6,3),qb(7,3),qb(4,3),qb(3,3),
    #        qb(3,2),qb(4,2),qb(5,6),qb(5,5)]
    #pairs = [(qubs[19],qubs[0]),(qubs[18],qubs[5]),
    #         (qubs[14],qubs[1]),(qubs[6],qubs[13]),
    #         (qubs[2],qubs[9]),(qubs[10],qubs[17]),
    #         (qubs[3],qubs[4]),(qubs[7],qubs[8]),
    #         (qubs[11],qubs[12]),(qubs[15],qubs[16]),

    #         (qubs[39],qubs[20]),(qubs[38],qubs[25]),
    #         (qubs[34],qubs[21]),(qubs[26],qubs[33]),
    #         (qubs[22],qubs[29]),(qubs[30],qubs[37]),
    #         (qubs[23],qubs[24]),(qubs[27],qubs[28]),
    #         (qubs[31],qubs[32]),(qubs[35],qubs[36]),
    #         ]

    # 1_500_000
    #zeroten_state = lambda qubs: parallelize([cirq.Circuit(ZeroTet(qubs[i*4:i*4+4])) for i in range(10)])
    #zeroten_transition = cirq.Circuit(zeroten_state(qubs),entangle_tets(pairs),device=SYC)
    ## entangle 38,17, 39,16, 37,19, 36,18 07 16 24 35
    #append_circuit = cirq.Circuit(
    #    cirq.Moment([cirq.SWAP(qubs[36],qubs[35]),cirq.SWAP(qubs[37],qubs[34]),cirq.SWAP(qubs[19],qubs[21]),
    #                 cirq.SWAP(qubs[16],qubs[15]),cirq.SWAP(qubs[38],qubs[25])]), 
    #                 # 36 <-> 35, 37 <-> 34, 19 <-> 21, 16 <-> 15, 38 <-> 25
    #    cirq.Moment([cirq.SWAP(qubs[15],qubs[14]),cirq.SWAP(qubs[39],qubs[38]),cirq.SWAP(qubs[18],qubs[19])]),
    #                 # 15 <-> 14, 39 <-> 38, 18 <-> 19
    #    apply(inv_epr_state, 
    #                      [(qubs[35],qubs[19]),(qubs[34],qubs[21]),(qubs[17],qubs[25]),(qubs[14],qubs[38])])
    #)
    #to_iswap.optimize_circuit(append_circuit)
    #zeroten_transition.append(append_circuit)
    #for i in range(15):
    #    result = run_circuit(zeroten_transition + measurement_circuit(qubs), 100_000)
    #    results_dict['zero_ten_transition_{}'.format(i)] = result.histogram(key='z')

    # 3_412_000

        

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