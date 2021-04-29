#import LQG
#import LQG.tetrahedra_num as tn
#import LQG.tetrahedra_circuits as tc
import tetrahedra_circuits as tc
import tetrahedra_num as tn
import cirq
import numpy as np

digs = 5
rnd = lambda x,y: round(x,ndigits=y)

def test_init_vec(theta,phi):
    Q = cirq.LineQubit.range(4)
    C = cirq.Circuit(tc.TetStatePrep(Q,(theta,phi),gateset=False))
    experiment = tc.final_state(C)
    print(f'Testing theta,phi {theta:.4f}, {phi:.4f} -> sqoverlap', end=' ')
    sqoverlap = tn.sqoverlap(tn.TetVec(theta,phi),experiment)
    print(f'{sqoverlap:.4f}',end=', ')
    try:
        assert(round(sqoverlap,ndigits=digs)==1.)
        print('passed!')
    except:
        print('failed! :(')

def test_polarizations():
    # test the constant-defined states
    try:
        assert(tn.sqoverlap(tn.L_zero, np.real(tn.TetVec(0,0)))==1.)
        assert(round(tn.sqoverlap(tn.L_one, tn.TetVec(np.pi,0)),ndigits=digs)==1.)
        assert(round(tn.sqoverlap(tn.L_plus, tn.TetVec(np.pi/2,0)),ndigits=digs)==1.)
        assert(round(tn.sqoverlap(tn.L_minus, tn.TetVec(-np.pi/2,0)),ndigits=digs)==1.)
        assert(round(tn.sqoverlap(tn.L_left, tn.TetVec(np.pi/2,np.pi/2)),ndigits=digs)==1.)
        assert(round(tn.sqoverlap(tn.L_right, tn.TetVec(np.pi/2,-np.pi/2)),ndigits=digs)==1.)
        print('passed!')
    except:
        0

def test_circuit_polarizations():
    # test the constant-defined state circuits
    qubits = cirq.LineQubit.range(4)
    assert(round(tn.sqoverlap(tn.L_zero,  tc.final_state(tc.ZeroTet(qubits,gateset=False))),ndigits=digs)==1.)
    assert(round(tn.sqoverlap(tn.L_one,   tc.final_state(tc.OneTet(qubits,gateset=False))),ndigits=digs)==1.)
    assert(round(tn.sqoverlap(tn.L_plus,  tc.final_state(tc.PlusTet(qubits,gateset=False))),ndigits=digs)==1.)
    assert(round(tn.sqoverlap(tn.L_minus, tc.final_state(tc.MinusTet(qubits,gateset=False))),ndigits=digs)==1.)
    assert(round(tn.sqoverlap(tn.L_left,  tc.final_state(tc.LeftTet(qubits,gateset=False))),ndigits=digs)==1.)
    assert(round(tn.sqoverlap(tn.L_right, tc.final_state(tc.RightTet(qubits,gateset=False))),ndigits=digs)==1.)
    print('passed!')

def test_amplitude(circuit, expected, digs=12):
    fvec = tc.final_state(circuit)
    while digs > 0:
        zero = rnd(abs(fvec[0])**2,digs)
        if zero == expected:
            break
        digs-=1
    if digs <= 3:
        print('expected: {}, got P = {}\namplitude = {}'.format(expected,fvec[0],abs(fvec[0])**2))
    else:
        print('{} digits, passed!'.format(digs))
    return digs

def main():
    qb = lambda x: cirq.LineQubit(x)

    # state preparation
    # test the circuit generation for arbitrary theta,phi
    print('Test the initializing vector function')
    for theta in [0.,np.pi,np.pi/2,np.pi/4]:
        for phi in [0.,np.pi/3,np.pi/2]:
            test_init_vec(theta,phi)
    # test that the constant defined intertwiner states are correct
    print('Test the numerical polarized intertwiner states')
    test_polarizations()  
    # are the polarized tetrahedra correct 
    print('Test the polarized intertwiner qubit circuit')
    test_circuit_polarizations()

    # monopole spins
    qubs = [qb(i) for i in range(4)]
    # |0>
    print('\nTest monopole \n|0>')
    circuit = cirq.Circuit(
        tc.ZeroTet(qubs,False),
        tc.entangle_tets([(qubs[0],qubs[3]),(qubs[1],qubs[2])])
    )
    test_amplitude(circuit,.25) 
    # |1>
    print('|1>')
    c = cirq.Circuit(
        tc.OneTet(qubs,False),
        tc.entangle_tets([(qubs[0],qubs[3]),(qubs[1],qubs[2])])
    )
    test_amplitude(c,.75)

    # dipole spins
    qubs = [qb(i) for i in range(8)]
    # |00>
    print('\nTest dipole 04152637\n|00>')
    c = cirq.Circuit(
        tc.ZeroTet(qubs[:4],False),
        tc.ZeroTet(qubs[4:],False),
        tc.entangle_tets([(qubs[0],qubs[4]),(qubs[1],qubs[5]),(qubs[2],qubs[6]),(qubs[3],qubs[7])])
    )
    test_amplitude(c,.0625)
    print('|11>')
    c = cirq.Circuit(
        tc.OneTet(qubs[:4],False),
        tc.OneTet(qubs[4:],False),
        tc.entangle_tets([(qubs[0],qubs[4]),(qubs[1],qubs[5]),(qubs[2],qubs[6]),(qubs[3],qubs[7])])
    )
    test_amplitude(c,.0625)
    # |00>
    print('\nTest dipole 07152634\n|00>')
    c = cirq.Circuit(
        tc.ZeroTet(qubs[:4],False),
        tc.ZeroTet(qubs[4:],False),
        tc.entangle_tets([(qubs[0],qubs[7]),(qubs[1],qubs[5]),(qubs[2],qubs[6]),(qubs[3],qubs[4])])
    )
    test_amplitude(c,1/8**2)
    print('|11>')
    c = cirq.Circuit(
        tc.OneTet(qubs[:4],False),
        tc.OneTet(qubs[4:],False),
        tc.entangle_tets([(qubs[0],qubs[7]),(qubs[1],qubs[5]),(qubs[2],qubs[6]),(qubs[3],qubs[4])])
    )
    test_amplitude(c,1/64)

    # 4-simplex
    # one paper says to expect 1/8^2 (or 1/2**8), the circuit gets 1/2**9
    qubs = [qb(i) for i in range(20)]
    # |00000>
    print('\nTest 4-simplex 19,0 18,5 14,1 6,13 2,9 10,17 3,4 7,8 11,12 15,16\n|00000>')
    c = cirq.Circuit([tc.ZeroTet(qubs[i*4:i*4+4],False) for i in range(5)])
    c.append(tc.entangle_tets([
        (qubs[19],qubs[0]),(qubs[18],qubs[5]),(qubs[14],qubs[1]),(qubs[6],qubs[13]),
        (qubs[2],qubs[9]),(qubs[10],qubs[17]),(qubs[3],qubs[4]),(qubs[7],qubs[8]),
        (qubs[11],qubs[12]),(qubs[15],qubs[16])
    ]))
    test_amplitude(c,1/2**8)
    print('|11111>')
    c = cirq.Circuit([tc.OneTet(qubs[i*4:i*4+4],False) for i in range(5)])
    c.append(tc.entangle_tets([
        (qubs[19],qubs[0]),(qubs[18],qubs[5]),(qubs[14],qubs[1]),(qubs[6],qubs[13]),
        (qubs[2],qubs[9]),(qubs[10],qubs[17]),(qubs[3],qubs[4]),(qubs[7],qubs[8]),
        (qubs[11],qubs[12]),(qubs[15],qubs[16])
    ]))
    test_amplitude(c,1/2**8)

if __name__ == '__main__':
    main()