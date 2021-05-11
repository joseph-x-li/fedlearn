import copy
import  torch

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