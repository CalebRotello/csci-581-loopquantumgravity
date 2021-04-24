import LQG.tetrahedra_num as tn
import LQG.tetrahedra_circuits as tc
import cirq
import numpy as np

def test_init_vec(theta,phi):
    Q = cirq.LineQubit.range(4)
    C = cirq.Circuit(tc.TetStatePrep(Q,(theta,phi)))
    experiment = tc.final_state(C)
    print(f'Testing theta,phi {theta:.4f}, {phi:.4f} -> overlap', end=' ')
    overlap = tn.overlap(tn.TetVec(theta,phi),experiment)
    print(f'{overlap:.4f}',end=', ')
    try:
        assert(round(overlap,ndigits=6)==1.)
        print('passed!')
    except:
        print('failed! :(')

def test_polarizations():
    # test the constant-defined states
    try:
        assert(tn.overlap(tn.L_zero, np.real(tn.TetVec(0,0)))==1.)
        a = tn.bloch['one']
        assert(round(tn.overlap(tn.L_one, tn.TetVec(a[0],a[1])),ndigits=6)==1.)
        assert(round(tn.overlap(tn.L_plus, tn.TetVec(np.pi/2,0)),ndigits=6)==1.)
        assert(round(tn.overlap(tn.L_minus, tn.TetVec(-np.pi/2,0)),ndigits=6)==1.)
        assert(round(tn.overlap(tn.L_left, tn.TetVec(np.pi/2,np.pi/2)),ndigits=6)==1.)
        assert(round(tn.overlap(tn.L_right, tn.TetVec(np.pi/2,-np.pi/2)),ndigits=6)==1.)
        print('passed!')
    except:
        0

def test_circuit_polarizations():
    # test the constant-defined state circuits
    qubits = cirq.LineQubit.range(4)
    assert(round(tn.overlap(tn.L_zero,  tc.final_state(tc.ZeroTet(qubits))),ndigits=6)==1.)
    assert(round(tn.overlap(tn.L_one,   tc.final_state(tc.OneTet(qubits))),ndigits=6)==1.)
    assert(round(tn.overlap(tn.L_plus,  tc.final_state(tc.PlusTet(qubits))),ndigits=6)==1.)
    assert(round(tn.overlap(tn.L_minus, tc.final_state(tc.MinusTet(qubits))),ndigits=6)==1.)
    assert(round(tn.overlap(tn.L_left,  tc.final_state(tc.LeftTet(qubits))),ndigits=6)==1.)
    assert(round(tn.overlap(tn.L_right, tc.final_state(tc.RightTet(qubits))),ndigits=6)==1.)
    print('passed!')
