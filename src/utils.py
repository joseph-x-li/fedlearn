import copy
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def weight_dist(w1, w2):
    """
    Calculates the squared L2 distance between weights w1 and w2 (produced by .state_dict())
    Used in FedSEM
    """
    totaldist = 0
    for k in w1.keys():
        totaldist += ((w1[k] - w2[k]) ** 2).sum()


    return totaldist

def _flatten(source, args):
    """
    Flatten a state_dict into a 1D tensor.
    If we are doing weight sharing, this will only flatten the FC layers.
    """
    if args.cfl_wsharing:
        return torch.cat([v.flatten() for v in source.values()[4:]]).detach()
    return torch.cat([value.flatten() for value in source.values()]).detach()

def subtract_weights(minuends, subtrahend, args):
    """
    Used in CFL
    minuends: list of state_dicts
    subtrahend: state_dict of cluster center
    return flattened weights on CPU as a vector
    """
    result = []
    subtrahend = _flatten(subtrahend, args)
    for dct in minuends:
        minuend = _flatten(dct, args)
        result.append(minuend - subtrahend)
    
    return result

def pairwise_cossim(state_vecs):
    """
    Used in CFL
    Computes all pairwise cosine similiarities given a list of state_vecs.
    Returns a 2D numpy array.
    """
    angles = torch.zeros([len(state_vecs), len(state_vecs)])
    for i, t1 in enumerate(state_vecs):
        for j, t2 in enumerate(state_vecs):
            angles[i,j] = torch.sum(t1 * t2) / (torch.norm(t1) * torch.norm(t2) + 1e-12)

    return angles.numpy()

def compute_max_update_norm(state_vecs):
    """
    state_vecs: contianing weight updates for a cluster.
    return the magnitude of the one with largest L2 norm.
    """
    norms = [torch.norm(vec).item() for vec in state_vecs]
    print(f"average of norms: {sum(norms) / len(norms)}")
    return max(norms)

def compute_mean_update_norm(state_vecs):
    """
    state_vecs contianing weight updates for a cluster.
    return the L2 norm of the average weight update.
    """
    base = copy.deepcopy(state_vecs[0])
    for vec in state_vecs[1:]:
        base += vec
    return torch.norm(base / len(state_vecs)).item()

def cluster_clients(similarities):
    # since precomputed, it clusters based on distance. 
    # Thus, we negate cossim since close points have large cossim, which is the opposite of what we want.
    clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-similarities)
    c1_idx = np.argwhere(clustering.labels_ == 0).flatten() 
    c2_idx = np.argwhere(clustering.labels_ == 1).flatten() 
    return c1_idx, c2_idx

class Accumulator:
    def __init__(self):
        self.sharedw_dict = None
        self.shared_n = 0
        self.sharedw_keys = ['0.weight', '0.bias', '3.weight', '3.bias']

    def add(self, state_dicts):
        if self.sharedw_dict is None:
            self.sharedw_dict = {k : copy.deepcopy(state_dicts[0][k]) for k in self.sharedw_keys}
            for dct in state_dicts[1:]:
                for k in self.sharedw_keys:
                    self.sharedw_dict[k] += dct[k]
        else:
            for dct in state_dicts:
                for k in self.sharedw_keys:
                    self.sharedw_dict[k] += dct[k]
        
        self.shared_n += len(state_dicts)

    def write(self, models):
        """
        1) Average accumulated weights together
        2) write them into the state_dicts of the models
        """
        averaged = {k: self.sharedw_dict[k] / self.shared_n for k in self.sharedw_dict}
        for model in models:
            sd = model.state_dict()
            for k in self.sharedw_keys:
                sd[k] = copy.deepcopy(self.sharedw_dict[k])
            model.load_state_dict(sd)

    def flush(self):
        self.sharedw_dict = None
        self.shared_n = 0