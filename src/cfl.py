import torch
import copy
import random
import time
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

from arguments import args_parser
from update import LocalUpdate, test_inference
from data import loadfemnist_raw
from models import femnistmodel
from utils import average_weights, subtract_weights, pairwise_cossim,\
compute_mean_update_norm, compute_max_update_norm, cluster_clients, Accumulator

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# NIID WSH
# python3.6 cfl.py --epochs 500 --cpr 3 --local_ep 50 --local_bs 10 --lr 0.004 --iid 0 --dataset_size small --cfl_e1 0.225 --cfl_e2 0.9 --cfl_split_every 7 --cfl_min_size 20 --cfl_local_epochs 1 --cfl_wsharing 1
# NIID NWSH
# python3.6 cfl.py --epochs 500 --cpr 3 --local_ep 50 --local_bs 10 --lr 0.004 --iid 0 --dataset_size small --cfl_e1 0.225 --cfl_e2 0.9 --cfl_split_every 7 --cfl_min_size 20 --cfl_local_epochs 1 --cfl_wsharing 0

# ./preprocess.sh -s niid --sf 0.05 -k 100 -t sample --smplseed 1549786595 --spltseed 1549786796
# ./preprocess.sh -s iid --sf 0.05 -k 100 -t sample --smplseed 1549786595 --spltseed 1549786796


def train(args, global_model, raw_data_train, raw_data_test):
    start_time = time.time()
    user_list = list(raw_data_train[2].keys())[:100]
    nusers = len(user_list)
    cluster_models = [copy.deepcopy(global_model)]
    del global_model
    cluster_models[0].to(device)
    cluster_assignments = [user_list.copy()] # all users assigned to single cluster_model in beginning

    if args.cfl_wsharing:
        shaccumulator = Accumulator()

    if args.frac == -1:
        m = args.cpr
        if m > nusers:
            raise ValueError(f"Clients Per Round: {args.cpr} is greater than number of users: {nusers}")
    else:
        m = max(int(args.frac * nusers), 1)
    print(f"Training {m} users each round")
    print(f"Trying to split after every {args.cfl_split_every} rounds")

    train_loss, train_accuracy = [], []
    for epoch in range(args.epochs):
        # CFL
        if (epoch + 1) % args.cfl_split_every == 0: 
            all_losses = []
            new_cluster_models, new_cluster_assignments = [], []
            for cidx, (cluster_model, assignments) in enumerate(
                tzip(cluster_models, cluster_assignments, desc="Try to split each cluster")
            ):
                # First, train all models in cluster
                local_weights = []
                for user in tqdm(assignments, desc="Train ALL users in the cluster", leave=False):
                    local_model = LocalUpdate(args=args, raw_data=raw_data_train, user=user)
                    w, loss = local_model.update_weights(
                        copy.deepcopy(cluster_model), local_ep_override=args.cfl_local_epochs
                    )
                    local_weights.append(copy.deepcopy(w))
                    all_losses.append(loss)

                # record shared weights so far
                if args.cfl_wsharing:
                    shaccumulator.add(local_weights)

                weight_updates = subtract_weights(local_weights, cluster_model.state_dict(), args)
                similarities = pairwise_cossim(weight_updates)

                max_norm = compute_max_update_norm(weight_updates)
                mean_norm = compute_mean_update_norm(weight_updates)

                # wandb.log({"mean_norm / eps1": mean_norm, "max_norm / eps2": max_norm}, commit=False)
                split = mean_norm < args.cfl_e1 and max_norm > args.cfl_e2 and len(assignments) > args.cfl_min_size
                print(f"CIDX: {cidx}[{len(assignments)}] elem")
                print(f"mean_norm: {(mean_norm):.4f}; max_norm: {(max_norm):.4f}")
                print(f"split? {split}")
                if split:
                    c1, c2 = cluster_clients(similarities)
                    assignments1 = [assignments[i] for i in c1]
                    assignments2 = [assignments[i] for i in c2]
                    new_cluster_assignments += [assignments1, assignments2]
                    print(f"Cluster[{cidx}][{len(assignments)}] -> ({len(assignments1)}, {len(assignments2)})")
                    
                    local_weights1 = [local_weights[i] for i in c1]
                    local_weights2 = [local_weights[i] for i in c2]

                    cluster_model.load_state_dict(average_weights(local_weights1))
                    new_cluster_models.append(cluster_model)

                    cluster_model2 = copy.deepcopy(cluster_model)
                    cluster_model2.load_state_dict(average_weights(local_weights2))
                    new_cluster_models.append(cluster_model2)

                else:
                    cluster_model.load_state_dict(average_weights(local_weights))
                    new_cluster_models.append(cluster_model)
                    new_cluster_assignments.append(assignments)


            # Write everything
            cluster_models = new_cluster_models
            if args.cfl_wsharing:
                shaccumulator.write(cluster_models)
                shaccumulator.flush()
            cluster_assignments = new_cluster_assignments
            train_loss.append(sum(all_losses) / len(all_losses))


        # Regular FedAvg
        else: 
            all_losses = []

            # Do FedAvg for each cluster
            for cluster_model, assignments in tzip(cluster_models, cluster_assignments, desc="Train each cluster through FedAvg"):
                if args.sample_dist == "uniform":
                    sampled_users = random.sample(assignments, m)
                else:
                    xs = np.linspace(-args.sigm_domain, args.sigm_domain, len(assignments))
                    sigmdist = 1/(1 + np.exp(-xs))
                    sampled_users = np.random.choice(assignments, m, p=sigmdist / sigmdist.sum())

                local_weights = []
                for user in tqdm(sampled_users, desc="Training Selected Users", leave=False):
                    local_model = LocalUpdate(args=args, raw_data=raw_data_train, user=user)
                    w, loss = local_model.update_weights(copy.deepcopy(cluster_model))
                    local_weights.append(copy.deepcopy(w))
                    all_losses.append(loss)


                # update global and shared weights
                if args.cfl_wsharing:
                    shaccumulator.add(local_weights)
                new_cluster_weights = average_weights(local_weights)
                cluster_model.load_state_dict(new_cluster_weights)

            if args.cfl_wsharing:
                shaccumulator.write(cluster_models)
                shaccumulator.flush()
            train_loss.append(sum(all_losses) / len(all_losses))

        # Calculate avg training accuracy over all users at every epoch
        # regardless if it was a CFL step or not
        test_acc, test_loss = [], []
        for cluster_model, assignments in zip(cluster_models, cluster_assignments):
            for user in assignments:
                local_model = LocalUpdate(args=args, raw_data=raw_data_test, user=user)
                acc, loss = local_model.inference(model=cluster_model)
                test_acc.append(acc)
                test_loss.append(loss)
        train_accuracy.append(sum(test_acc) / len(test_acc))

        wandb.log({
            "Train Loss": train_loss[-1], 
            "Test Accuracy": (100 * train_accuracy[-1]), 
            "Clusters": len(cluster_models)
        })
        print(
            f"Train Loss: {train_loss[-1]:.4f}\t Test Accuracy: {(100 * train_accuracy[-1]):.2f}%"
        )

    print(f"Results after {args.epochs} global rounds of training:")
    print("Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print(f"Total Run Time: {(time.time() - start_time):0.4f}")


if __name__ == "__main__":
    args = args_parser()
    if args.iid:
        wandb.init(project="CFLIID")
    else:
        wandb.init(project="CFLNIID")
    
    data_base_path = ""
    data_base_path += "iid_" if args.iid else "niid_"
    data_base_path += "s" if args.dataset_size == "small" else "l"

    global_model = femnistmodel()
    data_train = loadfemnist_raw(data_base_path + "/train")
    data_test = loadfemnist_raw(data_base_path + "/test")
    model = train(args, global_model, data_train, data_test)
