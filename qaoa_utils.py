import qiskit
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, transpile

# Hamiltonian layer
def zz_term(qc, q1, q2, gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1,q2)

def get_cost_circuit(qc, G, gamma):
    N = G.number_of_nodes()
    for i, j in G.edges():
        zz_term(qc, i, j, gamma)
    return qc

# def get_cost_circuit(qc, G, gamma):
#     N = G.number_of_nodes()
#     for i, j, w in G.edges(data=True):
#         zz_term(qc, i, j, w["weight"]*gamma)
#     return qc

# Mixer layer
def x_term(qc, q1, beta):
    qc.rx(2*beta, q1)

def get_mixer_circuit(qc, G, beta):
    N = G.number_of_nodes()
    for n in G.nodes():
        x_term(qc, n, beta)
    return qc

# Combine two layers and get the QAOA circuit
def get_qaoa_circuit(G, gamma, beta, sv=False):
    p = len(beta) # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
#     qc = QuantumCircuit(N)
    # get the initial state
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        qc = get_cost_circuit(qc,G,gamma[i])
        qc = get_mixer_circuit(qc,G,beta[i])
    # finally, do not forget to measure the result!
    qc.barrier(range(N))
    if sv is False:
        qc.measure(range(N), range(N))
    return qc

def measure_circuit(qc, n_trials=1024,seed=None,sv=False):
    """Get the output from circuit, either measured samples or full state vector"""
    if sv is False:
        backend = Aer.get_backend("qasm_simulator")
        qc = transpile(qc, backend)
        job = execute(qc, backend, shots=n_trials, seed_simulator=seed)
        result = job.result()
        bitstrings = invert_counts(result.get_counts())
        return bitstrings
    else:
        backend = Aer.get_backend("statevector_simulator")
        result = execute(qc, backend).result()
        state = result.get_statevector()
        return get_adjusted_state(state)

# define the objective function
def invert_counts(counts):
    return {k[::-1]:v for k, v in counts.items()}

def state_num2str(basis_state_as_num, nqubits):
    return "{0:b}".format(basis_state_as_num).zfill(nqubits)

def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)

def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)

def state_to_ampl_counts(vec, eps=1e-15):
    """
    Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = "0{}b".format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real**2 + val.imag**2 > eps:
            counts[format(kk, str_format)] = val
    return counts

def get_adjusted_state(state):
    """
    Convert qubit ordering invert for state vector
    https://github.com/rsln-s/QAOA_tutorial/blob/main/Hands-on.ipynb
    """
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    nqubits = int(nqubits)

    adjusted_state = np.zeros(2**nqubits, dtype=complex)
    for basis_state in range(2**nqubits):
        adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
    return adjusted_state

def maxcut_obj(x,G):
    ''' calculate energy from one bitstring '''
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut

# def maxcut_obj(x,G):
#     ''' Weighted MaxCut. calculate energy from one bitstring '''
#     cut = 0
#     for i, j, w in G.edges(data=True):
#         if x[i] != x[j]:
#             # the edge is cut
#             cut -= 1*w["weight"]
#     return cut
            
def compute_maxcut_energy(counts, G):
    ''' calculate energy from samples '''
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy / total_counts

def compute_maxcut_energy_sv(counts, G):
    """Compute energy expectation from full state vector"""
    expectation_value = 0
    # convert state vector to dictionary
    counts = state_to_ampl_counts(counts)
    for (meas, wf) in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        expectation_value += obj_for_meas * (np.abs(wf) ** 2) 
    return expectation_value

from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error
from qiskit_aer.noise import depolarizing_error
def cutomize_noise_model(p_gate1=0.01):
    
    # error_gate1 = pauli_error([('X',p_gate1), ('Y',p_gate1), ('Z',p_gate1), ('I', 1 - 3*p_gate1)])
    error_gate1 = depolarizing_error(p_gate1, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_gate1, ["u"])
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])
    return noise_model

def customize_measure_circuit(qc, error_p, n_trials=1024):
    """Get the output from circuit, either measured samples or full state vector"""
    noise_model=cutomize_noise_model(p_gate1=error_p)
    sim = AerSimulator(noise_model=noise_model)
    qc = transpile(qc, sim, basis_gates= ['u', 'cx']) #
    result_noise = sim.run(qc,shots=n_trials).result()
    counts_noise = result_noise.get_counts(0) 
    bitstrings = invert_counts(counts_noise)
    return bitstrings

def get_black_box_objective(G,p,n_trials=1024,seed=None,repeat=1, sv=True):
    def f(theta):
        # first half is betas, second half is gammas
        gamma = theta[:p]
        beta = theta[p:]
        ## define the qaoa circuit
        qc = get_qaoa_circuit(G, gamma, beta, sv=sv)
        
        energy = 0
        for i in range(repeat):
            counts = measure_circuit(qc, n_trials=n_trials,seed=seed, sv=sv)
            if sv is False:
                temp_energy = compute_maxcut_energy(counts, G)
            else:
                temp_energy = compute_maxcut_energy_sv(counts, G)
                
            energy += temp_energy
            
        energy = energy/repeat 
        return energy
    return f

def get_black_box_objective_customized(G,p,error_para,n_trials=1024,repeat=1):
    p_gate1 = error_para[0] 
    
    def f(theta):
        # first half is betas, second half is gammas, last parameter for noise
        gamma = theta[:p]
        beta = theta[p:]
        ## define the qaoa circuit
        qc = get_qaoa_circuit(G, gamma, beta)
        
        energy = 0
        for i in range(repeat):
            counts = customize_measure_circuit(qc, p_gate1, n_trials=n_trials) 
            temp_energy = compute_maxcut_energy(counts, G)
            energy += temp_energy
        energy = energy/repeat 
        return energy        
    return f


def hilbert_iter(n_qubits):
    """
    An iterator over all 2**n_qubits bitstrings in a given hilbert space basis.
    """ 
    for n in range(2**n_qubits):
        yield np.fromiter(map(int, np.binary_repr(n, width=n_qubits)), dtype=np.bool_, count=n_qubits)
        
def best_cut(G):
    from operator import itemgetter
    n_node = len(G._node)
    all_config  = 1*np.array(list(hilbert_iter(n_node)), dtype=np.bool_)
    best_cut, best_solution = min([(maxcut_obj(x,G),x) for x in all_config], key=itemgetter(0))
    print(f"Best string: {best_solution} with cut: {-best_cut}")
    return best_solution, -best_cut
