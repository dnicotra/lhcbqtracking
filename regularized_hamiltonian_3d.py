import numpy as np
import itertools
import q_event_model as em
import copy

def generate_hamiltonian(event: em.event, params:dict):
    alpha = params.get('alpha')
    delta =  params.get('delta')
    eps = params.get('eps')
    gamma = params.get('gamma')

    # First generate the segments from subsequent detectors

    modules = copy.deepcopy(event.modules)
    modules.sort(key= lambda a: a.z)

    segments = []
    for idx in range(len(modules) - 1):
        from_hits = modules[idx].hits
        to_hits = modules[idx + 1].hits

        for from_hit, to_hit in itertools.product(from_hits,to_hits):
            segments.append(em.segment(from_hit,to_hit))

    N = len(segments)
    A = np.zeros((N, N))
    A_ang = np.zeros((N, N))
    A_bif = np.zeros((N, N))
    A_reg = np.eye(N)*(-delta)
    A_gamma = np.eye(N)*(-gamma)
    b = np.ones(N)*delta

    for (i,seg_i),(j,seg_j) in itertools.product(zip(range(N),segments),repeat=2):
        hit_from_i = seg_i.from_hit
        hit_from_j = seg_j.from_hit
        hit_to_i = seg_i.to_hit
        hit_to_j = seg_j.to_hit
        if i != j:
            if (hit_from_i == hit_to_j) or (hit_from_j == hit_to_i):
                vect_i = seg_i.to_vect()
                vect_j = seg_j.to_vect()
                cosine = np.dot(vect_i,vect_j)/(np.linalg.norm(vect_i)*np.linalg.norm(vect_j))
                
                if np.abs(cosine-1) < eps:
                    A_ang[i,j] += 1
            
            if (hit_from_i == hit_from_j) and (hit_to_i != hit_to_j):
                A_bif[i,j] += -alpha
            if (hit_from_i != hit_from_j) and (hit_to_i == hit_to_j):
                A_bif[i,j] += -alpha
    
    A = -1*(A_ang + A_bif + A_reg + A_gamma)
    components = {
        'A_ang': -A_ang,
        'A_bif': -A_bif,
        'A_reg': -A_reg,
        'A_gamma': -A_gamma
    }

    return A, b, components, segments

