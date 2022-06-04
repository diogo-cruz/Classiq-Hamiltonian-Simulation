import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
from qiskit.aqua.operators.list_ops.summed_op import SummedOp

def get_openfermion_str(pauli_term, reverse=False):
    
    cirq_term = []
    
    for i, op in enumerate(list(pauli_term)):
        if op == 'I':
            continue
        
        cirq_term.append(op + str(9-i if not reverse else i))
        
    new_pauli_term = ' '.join(cirq_term)
    
    return new_pauli_term

def ops_commute(op1, op2):
    sign = 1
    for pauli_1, pauli_2 in zip(list(op1), list(op2)):
        if pauli_1=='I' or pauli_2=='I' or pauli_1==pauli_2:
            continue
        sign *= -1
    
    return True if sign==1 else False

def ops_do_not_overlap(op_1, op_2):
    qbs = []
    for i, (p1, p2) in enumerate(zip(list(op_1), list(op_2))):
        if p1!='I' and p2!='I':
            return False
    return True

def match_weight(op_1, op_2):
    weight = 0.
    for p1, p2 in zip(list(op_1), list(op_2)):
        if p1=='I' and p2=='I':
            continue
        elif p1=='I' or p2=='I':
            weight += 1.
        elif p1 == p2:
            weight += 0.
        else:
            weight += 2.
    return weight

# From https://stackoverflow.com/a/10824420
def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            yield from flatten(i)
        else:
            yield i

def convert_to_qiskit(H_corr, n_qubits):
    """
    Converts QubitHamiltonian from Cirq/Openfermion into SummedOp from 
    Qiskit.

    Parameters
    ----------
    H_corr : QubitOperator
        Spin Hamiltonian.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    SummedOp
        `H_corr` Hamiltonian in SummedOp form. Note that Qiskit is big 
        endian, so the first qubit has the highest index, while the last 
        qubit has the first index, unlike OpenFermion.
    """

    oplist = []
    id = ['I'] * n_qubits
    for op, coeff in H_corr.terms.items():
        op_str = id.copy()
        for ind, pauli_op in list(op):
            op_str[ind] = pauli_op
        op_str = ''.join(reversed(op_str))
        pauli = PauliOp(Pauli(op_str), coeff=coeff)
        oplist.append(pauli)    

    H = SummedOp(oplist)

    return H
    
def nthperm(l, n):
    l = list(l)

    indices = []
    for i in range(1, 1+len(l)):
        indices.append(n % i)
        n //= i
    indices.reverse()

    perm = []
    for index in indices:
        # Using pop is kind of inefficient. We could probably avoid it.
        perm.append(l.pop(index))
    return tuple(perm)

def compare_with_exact(U_exp, U_exact, global_phase = -1.0709274663656798, circuit_phase = np.pi/2, get_result=False):
    # Note: the default circuit phase is specific to our submitted solution.
    total_phase = global_phase + circuit_phase
    diff_matrix = np.exp(1j*total_phase) * U_exp - U_exact
    error = np.linalg.norm(diff_matrix, ord=2)
    
    total_U = np.exp(-1j*total_phase) * U_exp.T.conj() @ U_exact
    corr_phase = np.mean(np.angle(np.diag(total_U)))
    corr_error = np.linalg.norm(np.exp(-1j*corr_phase) * total_U - np.eye(2**10), ord=2)
    print("Error: {}, Corr: {}, Phase: {}.".format(error, corr_error, corr_phase))
    
    if get_result:
        return error, corr_error