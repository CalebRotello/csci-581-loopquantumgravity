

import numpy as np
import fnmatch
import tetrahedra_circuits as tc 
import tetrahedra_num as tn
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
            0
        self.hist = hist
        self.nqubs = nqubs
        self.samples = 0
        for _,value in hist.items():
            self.samples += value
        try:
            self.amplitude = hist[0] / self.samples
        except:
            self.amplitude = 0

    def state_subset(self,start,end,verbose=True):
        states = {}
        print('subsets of {} to {}'.format(start, end))
        for key,value in self.hist.items():
            bstr = tc.binformat(self.nqubs).format(key)
            s = []
            for i,st in enumerate(start):
                s.append(bstr[st:end[i]])
            try:
                states[''.join(s)] += value
            except:
                states[''.join(s)] = value
        if verbose:
            for s,value in states.items():
                print(s, value)
            print()
        return states

    def post_select(self,keepfn):
        ''' provide a function which returns true when a state can be kept 
        '''
        newhist = {}
        for key,item in self.hist.items():
            if keepfn(key):
                newhist[key] = item
        self.__init__(newhist,self.nqubs)

    def probability(self,i):
        try:
            return self.hist[i]/self.samples
        except:
            return 0




class SampleWavefunction(Sample):
    ''' A more specific type of sample, where the histogram is used to reconstruct 
        the wavefunction
    '''
    def __init__(self,hist,nqubs):
        super().__init__(hist,nqubs)        
        self.wavefn = hist_to_wavefn(hist,nqubs)

    def error(self,expected):
        ''' overlap of abs value wavefn and abs value expected wavefn 
        '''
        self.sqoverlap = tn.overlap(self.wavefn,np.abs(expected))
        return self.sqoverlap




class Experiment():
    ''' A collection of samples for one specific model
    '''
    def __init__(self,results,re,qnum,qubitmap=[]):
        sample_names = fnmatch.filter(results.keys(),re)
        self.sample_table = {s: Sample(results[s],qnum,qubitmap) for s in sample_names}
        self.sample_count = len(sample_names)

    def avg(self):
        return np.sum([sample.amplitude for sample in self.sample_table.values()])/len(self.sample_count)

    def post_select(self,keepfn):
        ''' provide a function which returns true when a state can be kept 
        '''
        for name in self.sample_table.keys():
            self.sample_table[name].post_select(keepfn)

    def amplitude(self):
        alist = [s.amplitude for s in self.sample_table.values()]
        return alist, np.sum(alist)/len(alist)

    def probability(self,i):
        plist = []
        for hist in self.sample_table.values():
            plist.append(hist.probability(i))
        return plist,np.sum(plist)/len(plist)

def split_state(stateint,nqubs):
    state = tc.binformat(nqubs).format(stateint)
    substates = [state[i*4:i*4+4] for i in range(int(nqubs/4))]
    return substates




''' Meta
'''

def glob_samples(results,re,qnum):
    sample_names = fnmatch.filter(results.keys(),re)
    return Sample(histogram_average([results[s] for s in sample_names]),qnum)

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




''' Histogram manipulations
'''

def histogram_average(histlist):
    ''' take a list of histograms and return the histogram of their average
    '''
    avg_hist = {}
    samples = 0
    for hist in histlist:
        for key,value in hist.items():
            samples += value
            try:
                avg_hist[key] += value
            except:
                avg_hist[key] = value
    return avg_hist

def hist_to_wavefn(hist,N):
    ''' given a histogram, get the absolute value of its wavefunction 
    '''
    wavefn = np.zeros(2**N)
    runs = 0
    for key,value in hist.items():
        wavefn[key] = value
        runs += value
    return wavefn/np.linalg.norm(wavefn)

def plot_wavefn(plt,wavefn,N,label):
    strhist = {}
    for i,val in enumerate(wavefn):
        strhist[bin(i)] = val
    plt.bar(*zip(*strhist.items()),label=label)
    return plt

def plot_hist(plt,hist,N,label):
    strhist = {}
    total = 0
    for key,value in hist.items():
        b = tc.binformat(N).format(key)
        strhist[b] = value
        total+= value
    plt.bar(*zip(*strhist.items()),label=label)
    return plt


''' Post Selection
'''

def post_select_even(hist,N):
    ''' post select data from a histogram knowing the states should have an even 
        number of 1's 
    '''
    newhist = {}
    newsamples = 0
    for key,item in hist.items():
        s = tc.binformat(N).format(key)
        if s.count('1') % 2 == 0:
            newhist[key] = item
            newsamples += item
    return newhist, newsamples

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
