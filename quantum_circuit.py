import os
import sys
import q_event_model as em
import toymodel_3d as toy
import regularized_hamiltonian_3d as ham
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import random

from qiskit.algorithms.linear_solvers import hhl
import qiskit
from qiskit.quantum_info import Statevector, Operator
import qiskit.circuit

from collections import namedtuple, defaultdict

## FIXING THE SEED ##
random.seed(1)

Hit = namedtuple("Hit", ["x", "y", "z", "hit_id", "mod_id"])
def parse(raw_hits):
        
    hits = [Hit(*o) for o in raw_hits]
    
    
    module2hitids = defaultdict(set)
    hitid2hit = {}
    for hit in hits:
        module2hitids[hit.mod_id].add(hit.hit_id)
        hitid2hit[hit.hit_id] = hit

    return hits, module2hitids, hitid2hit

def map_track_hit(hitid2hit, tracks, modules = [5, 6, 7]):
    nice_subset_track_list = []
    for track in tracks:

        subset_track = []
        for hit_id in track:
            hit = hitid2hit.get(hit_id, None)
            if hit is None:
                # This is not a real hit, it was created by Monte Carlo method
                continue

            # This is a real hit from here on
            if hit.mod_id not in modules:
                continue
            
            subset_track.append(hit)
        
        if len(subset_track) == len(modules) and sorted([hit.mod_id for hit in subset_track]) == modules:
            nice_subset_track_list.append(subset_track)

    return nice_subset_track_list

def filtering(hitid2hit, mcps, modules = [6,7,8], n_particles = 2):
        nice_tracks = map_track_hit(hitid2hit, mcps, modules)
        return list(itertools.chain.from_iterable(random.sample(nice_tracks, k=n_particles)))


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

def circuit(raw_hits, mcps):
    # raw_hits is a list of tuples (x,y,z,hit_id,module_id) 
    
    # Generate toy sequence
    #detector = toy.generate_simple_detector(N_MODULES, LX, LY, SPACING)
    #event = toy.generate_event(detector, N_TRACKS, theta_max=np.pi / 50, seed=1)

    # Set hits to C++ hits
    
    _, _, hitid2hit = parse(raw_hits)
    
    filtered_hits = filtering(hitid2hit, mcps, modules=[6,7,8], n_particles=2)
    
    
    print("Creating Event....")
    hits = []
    module_dict = defaultdict(list)
    for hit_id, picked_hit in enumerate(filtered_hits):
        
        hit = em.hit(hit_id, picked_hit.x,picked_hit.y,picked_hit.z,picked_hit.mod_id, -1)
        
        module_dict[picked_hit.mod_id].append(hit)

    
    modules = []
    for module_id, module_hits in module_dict.items():
        modules.append(em.module(module_id, -1, -1, -1, module_hits))
    
    # Setting the event
    event = em.event(modules, None, hits)
    
    print("Creating unitary")
    result, lenb = create_unitary(event)
    # print(type(result), len(result), result(0))
    return result, lenb


def create_unitary(event, return_all=False):
    A, b, components, segments = generate_hamiltonian(event)

    A, b = upscale_pow2(A, b)

    solver, b_circuit = initialize_qiskit(b)

    circ = solver.construct_circuit(A, b_circuit, neg_vals=False)
    unitary = Operator(circ).data

    if return_all:
        return unitary, solver, A, b, segments, b_circuit

    return unitary, len(b)


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
