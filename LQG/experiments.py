
import cirq
import numpy as np
import inspect
#import pickle
import pprint
import json
import os
import LQG.utils as utils
from cirq.contrib.svg import SVGCircuit
from enum import Enum

SYCAMORE = cirq.google.devices.Sycamore
PROJECT_ID = 'HA_CR_LQG'
DATA_DIR = 'HA_CR_LQG_numerical_tests'

class Experiment:
    def __init__(self, intertwiner_states, qubits, entagling_pairs, invert_mask=None, on_hardware=True):
        '''Ctor.
            :param list(state fn) intertwiner_states: List of starting states of the system or single starting state (convert to list)
            :param list(Qubit) qubits: List of qubits
            :param list(int) entangling_pairs: List of states to entangle. List of tuples or even-sized list (convert to list of tuples)
            :param list(bool) invert_mask: List of bool values indicating if a bit should be flipped at the end of the circuit
            :param int experiments: Number of experiments
            :param int samples: Number of samples per experiment
            :param bool on_hardware: Convert circuit to iSWAP hardware

            create the initialization of each intertwiner qubit state with the qubits listed
            qubits will be a list of GridQubit, where the each qubits index is its place in the intertwiners
            intertwiners consume qubits as (0,1,2,3),(4,5,6,7), ...
            entangle given pairs. this function may have to be done outside of the class, because of SWAPs
        '''
        # convert to list of start states
        if inspect.isfunction(intertwiner_states):
            intertwiner_states = [intertwiner_states]

        num_states = len(intertwiner_states)
        assert(len(qubits) / 4 == num_states)
        assert(len(qubits) % 4 == 0)

        # convert to list of tuples
        if type(entagling_pairs[0]) is int:
            entagling_pairs = [(entagling_pairs[i*2], entagling_pairs[i*2+1]) for _,i in enumerate(entagling_pairs[::2])]
        entagling_pairs = np.array(entagling_pairs)
        assert(len(entagling_pairs) * 2 == len(qubits))

        # create the circuit
        if invert_mask is None:
            invert_mask = [False]*len(qubits)
        self.invert_mask = invert_mask

        states = [intertwiner_states[i](qubits[i*4:i*4+4]) for i in range(num_states)]
        self.circuit = cirq.Circuit(states, 
                                    inverse_circuit(entagling_pairs, qubits, invert_mask),
                                    )
        self.circuit = utils.parallelize(self.circuit)

        mapped_invert_mask = self.invert_mask
        if on_hardware:
            self.circuit = utils.to_iswap(self.circuit)
            self.circuit = cirq.Circuit(self.circuit, device=SYCAMORE)
            self.qubit_map = utils.remap_qubits(qubits)
            mapped_invert_mask = [self.invert_mask[item] for _,item in self.qubit_map.items()]

        self.circuit.append(cirq.measure(*qubits,invert_mask=mapped_invert_mask,key='z'))

        self.on_hardware = on_hardware
        self.qubits = qubits
        self.entangling_pairs = entagling_pairs


    def svg(self):
        return SVGCircuit(self.circuit)

    def sim(self, target_engine, experiments=10, samples=1024):
        ''' run experiment '''
        print('Simulating {} experiments with {} samples each'.format(experiments, samples))
        if target_engine == SYCAMORE:
            if not self.on_hardware:
                print("WARNING circuit submitted to Sycamore hardware was not compiled for the gateset")
                print("Converting to iSWAP gateset now...")
                utils.to_iswap(self.circuit)
            engine = cirq.google.Engine(project_id=PROJECT_ID) 
        self.experiment_results = []
        for experiment in range(experiments):
            if target_engine == SYCAMORE:
                print('Running experiment {} of {} on QPU...'.format(experiment,experiments),end='')
                results = engine.run(program=self.circuit,
                                     repetitions=samples,
                                     processor_ids=[SYCAMORE],
                                     gate_set=cirq.google.SQRT_ISWAP_GATESET,
                                    )
                print("Done")
            else:
                print("Running experiment {} of {} on classical simulator...".format(experiment,experiments),end='')
                results = cirq.Simulator().run(self.circuit,repetitions=samples)
                print("Done")
            self.experiment_results.append(results.histogram(key='z'))
        print("Finished.")

    def save(self, name, description=None):
        ''' save the experiment to a file, named after the (hopefully unique) name of the experiment '''
        root_dir = os.getcwd()
        top_dir = DATA_DIR
        dir_path = os.path.join(root_dir,top_dir) 
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = '{}.json'.format(name)
        print('Saving {}'.format(filename))
        with open(dir_path + '/' + filename, 'w') as f:
            data = {'description': description,
                    'qubits' : utils.serialize_qubits(self.qubits),
                    'qubit_map' : self.qubit_map if self.on_hardware else None,
                    'pairs' : self.entangling_pairs.tolist(),
                    'remapped' : False,
                    'results' : self.experiment_results,
                    'invert_mask' : self.invert_mask
                   }
            json.dump(data,f)
            

class Result:
    ''' store the data from the experiment plus circuit information useful for calculations '''
    def __init__(self, experiment_name):
        with open(DATA_DIR+'/'+experiment_name+'.json', 'r') as f:
            results_json = json.load(f)
        self.experiment_name = experiment_name
        self.description = results_json['description']
        self.qubits = results_json['qubits']
        self.qubit_map = results_json['qubit_map']
        self.pairs = results_json['pairs']
        self.histograms = results_json['results']  
        self.invert_mask = results_json['invert_mask']
        if not results_json['remapped'] and self.qubit_map is not None:
            self.histograms = [utils.remap(hist,self.qubit_map) for hist in self.histograms]
        elif self.qubit_map is None:
            self.histograms = [{int(key):value for key,value in hist.items()} for hist in self.histograms]

    def info(self):
        print(self.experiment_name)
        print(self.description)
        print('qubits: ',self.qubits)
        print('pairs: ',self.pairs)
        print('invert_mask: ',self.invert_mask)


    def filter(self,filter_fn):
        histograms = []
        removed = []
        for hist in self.histograms:
            newhist = {}
            removed.append(0)
            for key,item in hist.items():
                if filter_fn(key,len(self.qubits)):
                    newhist[key] = item
                else:
                    removed[-1]+=1
            histograms.append(newhist)
        self.histograms = histograms 
        print("Filter removed the following: {}".format(removed))
        

    def report(self,state=0,expected=None):
        self._avg_()
        print('---------------------------------------')
        print('REPORT {}:'.format(self.experiment_name))
        print('Amplitudes of |{}>: (amplitude, sample#)'.format(state))
        pprint.pprint([self._amplitude_(state,hist) for hist in self.histograms])
        avgamp = self._amplitude_(state,self.avg_histogram)
        print('Avgerage: {}'.format(avgamp))
        if expected is not None:
            print("%error: {}".format( np.abs((avgamp[0] - expected) / expected)))
        print('---------------------------------------')

    def _amplitude_(self,state_int,hist):
        total_runs = sum([value for _,value in hist.items()])
        return hist[state_int]/total_runs, total_runs

    def _avg_(self):
        self.avg_histogram = {}
        for hist in self.histograms:
            for key,value in hist.items():
                if key not in self.avg_histogram:
                    self.avg_histogram[key] = 0
                self.avg_histogram[key] += value




''' algorithm circuit generating functions '''

def inverse_circuit(pairs,qubits,invert_mask):
    ''' create an inverse(ish) circuit given a list of pairs (source->sink) and list of states to flip '''
    paired = set() # make sure we dont double pair
    circuit = cirq.Circuit()
    for p in pairs:
        for face in p:
            assert(face not in paired)
            paired.add(face)
        circuit.append(
            cirq.CNOT(qubits[p[0]],qubits[p[1]])
        )
    circuit.append(cirq.H.on_each([qubits[p[0]] for p in pairs]))
    for i,bit in enumerate(invert_mask):
        if not bit:
            circuit.append(cirq.X(qubits[i]))
    return circuit


def zero(qubits):
    assert(len(qubits) == 4)
    return cirq.Circuit(
        cirq.X.on_each(qubits),
        cirq.H.on_each(qubits[::2]),
        cirq.CNOT(qubits[0],qubits[1]),
        cirq.CNOT(qubits[2],qubits[3]),
    )

def one(qubits):
    assert(len(qubits) == 4)
    return cirq.Circuit(
        cirq.H(qubits[0]),
        UGate(np.pi,0).on(qubits[1]),
        cirq.H(qubits[2]).controlled_by(qubits[1]),
        cirq.X(qubits[2]),
        cirq.CNOT(qubits[1],qubits[3]),
        cirq.CNOT(qubits[0],qubits[1]),
        cirq.CNOT(qubits[1],qubits[2]),
        cirq.CNOT(qubits[2],qubits[3]),
    )


''' circuit information and gates '''

c_one = lambda theta,phi: np.sqrt(2/3)*np.exp(1j*phi)*np.sin(theta/2)
c_two = lambda theta,phi: np.sqrt(1/2)*(np.cos(theta/2) - np.sqrt(1/3)*np.exp(1j*phi)*np.sin(theta/2))
c_three = lambda theta,phi: np.sqrt(1/2)*(-np.cos(theta/2) - np.sqrt(1/3)*np.exp(1j*phi)*np.sin(theta/2))

distance = lambda a,b: np.sqrt(np.abs(a)**2 + np.abs(b)**2)

def U(theta,phi):
    c1 = c_one(theta,phi)
    c2 = c_two(theta,phi)
    c3 = c_three(theta,phi)
    return [[c1, distance(c2,c3)], 
            [-1*distance(c2,c3), np.conj(c1)]]

def V(theta,phi):
    c1 = c_one(theta,phi)
    c2 = c_two(theta,phi)
    c3 = c_three(theta,phi)
    return [[-1*c2/distance(c2,c3), np.conj(c3)/distance(c2,c3)],
            [-1*c3/distance(c2,c3), np.conj(c2)/distance(c2,c3)]]

class UGate(cirq.Gate):
    def __init__(self, theta, phi):
        super(UGate, self)
        self.theta = theta
        self.phi = phi

    def _num_qubits_(self):
        return 1
    
    def _unitary_(self):
        return np.array(U(self.theta,self.phi))

    def _circuit_diagram_info_(self,args):
        return f"U({self.theta:.2f},{self.phi:.2f})"

class CVGate(cirq.TwoQubitGate):
    def __init__(self, theta, phi):
        super(CVGate, self)
        self.theta = theta
        self.phi = phi

    def _unitary_(self):
        mtx = [[1.,0.,0.,0.],[0.,1.,0.,0.]]
        v = V(self.theta,self.phi)
        a = [0.,0.]
        a.extend(v[0])
        mtx.append(a)
        a = [0.,0.]
        a.extend(v[1])
        mtx.append(a)
        return np.array(mtx)

    def _circuit_diagram_info_(self, args):
        return "CV", f"V({self.theta:.2f},{self.phi:.2f})"