'''
    Post process the histogram results 
'''

import numpy as np
import fnmatch
import circuits as tc 
import numerical as tn
import json
from matplotlib import pyplot




class Sample():
    ''' Take a histogram returned by the experiment, deconstruct it into common useful forms
        hist: histogram
        wavefn: absolute value of the wavefunction
        nqubs: number of qubits in the system
    '''
    def __init__(self,hist,nqubs=4,qubitmap=None):
        if qubitmap is not None:
            # qubits were swapped, so we must swap back
            newhist = {}
            for key,value in hist.items():
                state = tc.binformat(nqubs).format(key)
                newstate = list(state)
                for start,end in qubitmap.items():
                    newstate[end] = state[start]
                newhist[int("".join(str(i) for i in newstate),2)] = value
            hist = newhist
        self.hist = hist
        self.nqubs = nqubs
        self.samples = 0
        for _,value in hist.items():
            self.samples += value
        try:
            self.amplitude = hist[0] / self.samples
        except:
            self.amplitude = 0

    def split(self,n):
        ''' split the samples into N seperate experiments '''
        size_subsys = int(self.nqubs/n)
        hists = [{} for _ in range(n)]
        for key,value in self.hist.items():
            measure_state = tc.binformat(self.nqubs).format(key)
            substates = [measure_state[size_subsys*i:size_subsys*i+size_subsys] for i in range(n)]
            for i,state in enumerate(substates):
                intstate = int(state,2)
                if intstate not in hists[i].keys():
                    hists[i][intstate] = 0
                hists[i][intstate] += value
        return [Sample(h,nqubs=size_subsys) for h in hists]

    def mask(self,flips):
        ''' take a list of qubits to flip '''
        newhist = {}
        for key,value in self.hist.items():
            state = list(tc.binformat(self.nqubs).format(key))
            for flip in flips:
                state[flip] = str(int(not int(state[flip])))
            newhist[int("".join(str(i) for i in state),2)] = value
        self.hist = newhist

    def post_select(self,keepfn):
        ''' provide a function which returns true when a state can be kept 
        '''
        newhist = {}
        for key,item in self.hist.items():
            if keepfn(key,self.nqubs):
                newhist[key] = item
        self.__init__(newhist,self.nqubs)

    def probability(self,i):
        try:
            return self.hist[i]/self.samples
        except:
            return 0


''' Meta
'''
def load(fname):
    ''' take a json filename, load it
        convert the keys to integers
    '''
    results_load = json.load(open(fname,'r'))
    results = {}
    sample_names = []
    for run,hist in results_load.items():
        sample_names.append(run)
        results[run] = {}
        for key,value in hist.items():
            results[run][int(key)] = value
    return results, sample_names



''' Post Selection Strategies
'''
def keepfn_L_zero(stateint,nqubs=4):
    ''' given a state in integer representation, check if it is valid 
    '''
    if tc.binformat(nqubs).format(stateint) in ('1001','0101','1010','0110'):
        return True
    return False

def keepfn_even(stateint,nqubs=4):
    if tc.binformat(nqubs).format(stateint).count('1') % 2 == 0:
        return True
    return False

def keepfn_paired(stateint,nqubs=4):
    state = tc.binformat(nqubs).format(stateint)
    for i in np.arange(0,nqubs,2):
        if state[i] != state[i+1]:
            return False
    return True

def keepfn_even_tets(stateint,nqubs=4):
    state = tc.binformat(nqubs).format(stateint)
    for i in range(int(nqubs/4)):
        if not keepfn_even(int(state[i*4:i*4+4],2),nqubs=4):
            return False
    return True



''' Runner '''
def process_results(results,qubitmap,n_qubits,mask,n_split=1,keepfn=None):
    ''' get the Sample of one type of experiment and return a amplitude '''
    amps = []
    for _,result in results.items():
        sample = Sample(result,n_qubits*n_split,qubitmap)
        sample.mask(mask)
        samples = sample.split(n_split)
        for _,s in enumerate(samples):
            if keepfn is not None:
                s.post_select(keepfn)
            amps.append(s.amplitude)
    return amps 



