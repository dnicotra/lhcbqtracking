import os
import sys
import q_event_model as em
import toymodel_3d as toy
import regularized_hamiltonian_3d as ham
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from qiskit.algorithms.linear_solvers import hhl
import qiskit
from qiskit.quantum_info import Statevector, Operator
import qiskit.circuit

N_MODULES = 3
N_TRACKS = 2

LX = 2
LY = 2
SPACING = 1


def generate_hamiltonian(event):
    params = {
        'alpha': 0.0,
        'delta': 1.0,
        'eps': 1e-9,
        'gamma': 2
    }
    return ham.generate_hamiltonian(event, params)


def upscale_pow2(A, b):
    m = A.shape[0]
    d = int(2 ** np.ceil(np.log2(m)) - m)
    if d > 0:
        A_tilde = np.block([[A, np.zeros((m, d), dtype=np.float64)],
                            [np.zeros((d, m), dtype=np.float64), np.eye(d, dtype=np.float64)]])
        b_tilde = np.block([b, b[:d]])
        return A_tilde, b_tilde
    else:
        return A, b


def initialize_qiskit(b):
    solver = hhl.HHL(epsilon=0.1)
    b_circuit = qiskit.QuantumCircuit(qiskit.QuantumRegister(int(np.log2(len(b)))), name="init")
    for i in range(int(np.log2(len(b)))):
        b_circuit.h(i)

    return solver, b_circuit


def circuit(raw_hits):
    # raw_hits is a list of tuples (x,y,z,module_id) 
    
    # Generate toy sequence
    #detector = toy.generate_simple_detector(N_MODULES, LX, LY, SPACING)
    #event = toy.generate_event(detector, N_TRACKS, theta_max=np.pi / 50, seed=1)

    # Set hits to C++ hits
    print("Creating hits....")
    hits = []
    module_dict = dict()
    for hit_id, raw_hit in enumerate(raw_hits):
        x,y,z = raw_hit[:3]
        module_id = raw_hit[3]
        
        hit = em.hit(hit_id, x,y,z,module_id, -1)
        
        if module_id in module_dict:
            module_dict[module_id].append(hit)
        else:
            module_dict[module_id] = list().append(hit)

    
    modules = []
    for module_id, module_hits in module_dict.items():
        modules.append(em.module(module_id, -1, -1, -1, module_hits))
    
    # Setting the event
    event = em.event(modules, None, hits)
    
    print("Creating unitary")
    result = create_unitary(event)
    # print(type(result), len(result), result(0))
    return result


def create_unitary(event, return_all=False):
    A, b, components, segments = generate_hamiltonian(event)

    A, b = upscale_pow2(A, b)

    solver, b_circuit = initialize_qiskit(b)

    circ = solver.construct_circuit(A, b_circuit, neg_vals=False)
    unitary = Operator(circ).data

    if return_all:
        return unitary, solver, A, b, segments, b_circuit

    return unitary


def show_solution(event, solution_segments):
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = plt.axes(projection='3d')
    event.display(ax, show_tracks=False)

    for segment in solution_segments:
        segment.display(ax)

    # ax.view_init(vertical_axis='y')
    fig.set_tight_layout(True)
    ax.axis('off')

    ax.set_title(f"Solution")
    plt.show()


def main():
    detector = toy.generate_simple_detector(N_MODULES, LX, LY, SPACING)
    event = toy.generate_event(detector, N_TRACKS, theta_max=np.pi / 50, seed=1)

    unitary, solver, A, b, segments, b_circuit = create_unitary(event, return_all=True)

    backend = qiskit.Aer.get_backend("aer_simulator_statevector")
    result = solver.solve(A, b_circuit)
    sv = Statevector(result.state)
    # %%
    solution_norm = result.euclidean_norm
    post_select_qubit = int(np.log2(len(sv.data))) - 1
    solution_len = len(b)
    base = 1 << post_select_qubit
    solution_vector = sv.data[base: base + solution_len].real
    solution_vector = solution_vector / np.linalg.norm(solution_vector) * solution_norm * np.linalg.norm(b)
    solution_segments = [seg for pseudosol, seg in zip(solution_vector, segments) if pseudosol > 0.45]

    # show_solution(event, solution_segments)


if __name__ == "__main__":
    main()
