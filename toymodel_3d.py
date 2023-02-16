import numpy as np
import dataclasses as dc
import itertools
import q_event_model as em

default_geometry = [
    {'module_id': 0, 'z': 1.0, 'lx': 1.0, 'ly': 1.0},
    {'module_id': 1, 'z': 2.0, 'lx': 1.0, 'ly': 1.0},
    {'module_id': 2, 'z': 3.0, 'lx': 1.0, 'ly': 1.0},
    {'module_id': 3, 'z': 4.0, 'lx': 1.0, 'ly': 1.0},
]

def generate_simple_detector(n_modules, lx, ly, spacing):
    return [{'module_id': i, 'z': spacing*(i+1), 'lx': lx, 'ly':ly} for i in range(n_modules)]


def generate_event(geometry, N_tracks, primary_vertex_iter=itertools.repeat((0.0,0.0,0.0)),phi_min = 0, phi_max = 2*np.pi, theta_min = 0, theta_max=np.pi/10,seed=None):

    hit_id_counter = itertools.count()
    rng = np.random.default_rng(seed)
    mc_tracks = []

    hits_per_module = [[] for _ in geometry]
    hits_per_track = []
    for track_id in range(N_tracks):
        pvx, pvy, pvz = next(primary_vertex_iter)
        phi = rng.uniform(phi_min, phi_max)
        cos_theta = rng.uniform(np.cos(theta_max),np.cos(theta_min))
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)
        mc_tracks.append({
            'track_id': track_id,
            'pv': (pvx, pvy, pvz),
            'phi': phi,
            'theta': theta
        })

        vx = sin_theta*np.cos(phi)
        vy = sin_theta*np.sin(phi)
        vz = cos_theta

        track_hits = []
        for idx, module in enumerate(geometry):
            zm = module['z']
            lx = module['lx']
            ly = module['ly']
            t = (zm - pvz)/vz
            x_hit = pvx + vx*t
            y_hit = pvy + vy*t

            if np.abs(x_hit) < lx/2 and np.abs(y_hit) < ly/2:
                hit = em.hit(next(hit_id_counter), x_hit, y_hit, zm, module['module_id'], track_id)
                hits_per_module[idx].append(hit)
                track_hits.append(hit)
        hits_per_track.append(track_hits)

    modules = [em.module(modgeom['module_id'], modgeom['z'], modgeom['lx'], modgeom['ly'], hits_per_module[idx]) for idx, modgeom in enumerate(geometry)]
    tracks = []

    for idx, mc_track in enumerate(mc_tracks):
        tracks.append(em.track(mc_track['track_id'], mc_track, hits_per_track[idx]))
    global_hits = [hit for sublist in hits_per_module for hit in sublist]

    event = em.event(modules, tracks, global_hits)
    return event

        
                



    



