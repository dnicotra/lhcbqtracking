import itertools
import pickle
import random
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt

Hit = namedtuple("Hit", ["x", "y", "z", "hit_id", "mod_id"])
def load_dump(path):
    """
    Dump has the following format

    (hits, tracks)

    Where hit is formatted as:
     (x, y, z, hit_id, module_id)

    And track is formatted as
     list(hit_id)
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
        
    hits = [Hit(*o) for o in obj[0]]
    
    mcps = obj[1]
    
    
    module2hitids = defaultdict(set)
    hitid2hit = {}
    for hit in hits:
        module2hitids[hit.mod_id].add(hit.hit_id)
        hitid2hit[hit.hit_id] = hit

    return hits, module2hitids, hitid2hit, mcps

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
        
    
if __name__=="__main__":
    
    hits, module2hitids, hitid2hit, mcps  = load_dump("data.dump")

    
    filtered = filtering(hitid2hit, mcps)
    print(len(filtered))
    exit(0)
    print(filtering(hits, module2hitids, hitid2hit, mcps))
    
    
    
    
        