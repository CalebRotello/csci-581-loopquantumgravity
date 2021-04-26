import numpy as np
from LQG import tetrahedra_circuits as tc 



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


