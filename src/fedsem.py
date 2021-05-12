import torch
import copy
import random
import time
import numpy as np
from tqdm import tqdm
from collections import Counter

from arguments import args_parser
from update import LocalUpdate, test_inference
from data import loadfemnist_raw
from models import femnistmodel
from utils import average_weights, weight_dist

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# IID  # python3.6 fedsem.py --clusters 4 --epochs 50 --cpr 3 --local_ep 100 --local_bs 10 --lr 0.004 --iid 1 --dataset_size small
# NIID # python3.6 fedsem.py --clusters 4 --epochs 1000  --local_ep 1 --local_bs 10 --lr 0.004 --iid 0 --dataset_size small


def train(args, global_model, raw_data_train, raw_data_test):
    start_time = time.time()
    user_list = list(raw_data_train[2].keys())
    user_weights = [None for _ in range(len(user_list))]
    user_assignments = [i % args.clusters for i in range(len(user_list))]

    # global_model.to(device)
    # global_weights = global_model.state_dict()
    global_models = [copy.deepcopy(global_model) for _ in range(args.clusters)]
    for m in global_models: m.to(device)

    # if args.frac == -1:
    #     m = args.cpr
    #     if m > len(user_list):
    #         raise ValueError(f"Clients Per Round: {args.cpr} is greater than number of users: {len(user_list)}")
    # else:
    #     m = max(int(args.frac * len(user_list)), 1)
    # print(f"Training {m} users each round")

    train_loss, train_accuracy = [], []
    for epoch in range(args.epochs):
        print(f"Global Training Round: {epoch + 1}/{args.epochs}")
        local_losses = []
        for modelidx, cluster_model in tqdm(enumerate(global_models)):
            local_weights = []
            for useridx, (user, user_assign) in enumerate(zip(user_list, user_assignments)):
                if user_assign == modelidx:
                    local_model = LocalUpdate(args=args, raw_data=raw_data_train, user=user)
                    w, loss = local_model.update_weights(copy.deepcopy(cluster_model))
                    local_weights.append(w)
                    local_losses.append(loss)
                    user_weights[useridx] = w
            if local_weights:
                cluster_model.load_state_dict(average_weights(local_weights))

        train_loss.append(sum(local_losses) / len(local_losses))

        # sampled_users = random.sample(user_list, m)
        # for user in tqdm(sampled_users):
        # FedSEM cluster reassignment step
        print(f"Calculating User Assignments")
        dists = np.zeros((len(user_list), len(global_models)))
        for cidx, cluster_model in enumerate(global_models):
            for ridx, user_weight in enumerate(user_weights):
                dists[ridx, cidx] = weight_dist(user_weight, cluster_model.state_dict())

        user_assignments = list(np.argmin(dists, axis=1))
        print("Cluster: number of clients in that cluster index")
        print(Counter(user_assignments))
        print(f"")

        # Calculate avg training accuracy over all users at every epoch
        test_acc, test_loss = [], []
        for modelidx, cluster_model in enumerate(global_models):
            local_weights = []
            for user, user_assign in zip(user_list, user_assignments):
                if modelidx == user_assign:
                    local_model = LocalUpdate(args=args, raw_data=raw_data_test, user=user)
                    acc, loss = local_model.inference(model=cluster_model)
                    test_acc.append(acc)
                    test_loss.append(loss)

        train_accuracy.append(sum(test_acc) / len(test_acc))
        wandb.log({"Train Loss": train_loss[-1], "Test Accuracy": (100 * train_accuracy[-1])})
        print(
            f"Train Loss: {train_loss[-1]:.4f}\t Test Accuracy: {(100 * train_accuracy[-1]):.2f}%"
        )

    print(f"Results after {args.epochs} global rounds of training:")
    print("Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print(f"Total Run Time: {(time.time() - start_time):0.4f}")

if __name__ == "__main__":
    args = args_parser()
    if args.iid:
        wandb.init(project="FedSEMIID")
    else:
        wandb.init(project="FedSEMNIID")
    
    data_base_path = ""
    data_base_path += "iid_" if args.iid else "niid_"
    data_base_path += "s" if args.dataset_size == "small" else "l"

    global_model = femnistmodel()
    data_train = loadfemnist_raw(data_base_path + "/train")
    data_test = loadfemnist_raw(data_base_path + "/test")
    model = train(args, global_model, data_train, data_test)