import torch
import copy
import random
import time
from tqdm import tqdm

from arguments import args_parser
from update import LocalUpdate, test_inference
from data import loadfemnist_raw
from models import femnistmodel
from utils import average_weights

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# IID  # python3.6 fedavg.py --epochs 500 --cpr 3 --local_ep 100 --local_bs 10 --lr 0.004 --iid 1 --dataset_size small
# NIID # python3.6 fedavg.py --epochs 500 --cpr 3 --local_ep 100 --local_bs 10 --lr 0.004 --iid 0 --dataset_size small
# ./preprocess.sh -s niid --sf 0.05 -k 100 -t sample --smplseed 1549786595 --spltseed 1549786796
# ./preprocess.sh -s iid --sf 0.05 -k 100 -t sample --smplseed 1549786595 --spltseed 1549786796


def train(args, global_model, raw_data_train, raw_data_test):
    start_time = time.time()
    user_list = list(raw_data_train[2].keys())
    global_model.to(device)
    global_weights = global_model.state_dict()

    if args.frac == -1:
        m = args.cpr
        if m > len(user_list):
            raise ValueError(f"Clients Per Round: {args.cpr} is greater than number of users: {len(user_list)}")
    else:
        m = max(int(args.frac * len(user_list)), 1)
    print(f"Training {m} users each round")

    train_loss, train_accuracy = [], []
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        print(f"Global Training Round: {epoch + 1}/{args.epochs}")
        sampled_users = random.sample(user_list, m)
        for user in tqdm(sampled_users):
            local_model = LocalUpdate(args=args, raw_data=raw_data_train, user=user)
            w, loss = local_model.update_weights(copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        train_loss.append(sum(local_losses) / len(local_losses))

        # Calculate avg training accuracy over all users at every epoch
        test_acc, test_loss = [], []
        for user in user_list:
            local_model = LocalUpdate(args=args, raw_data=raw_data_test, user=user)
            acc, loss = local_model.inference(model=global_model)
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


def test(args, model):
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print("Test Accuracy: {:.2f}%".format(100 * test_acc))
    pass


if __name__ == "__main__":
    args = args_parser()
    if args.iid:
        wandb.init(project="FedAvgIID")
    else:
        wandb.init(project="FedAvgNIID")
    
    data_base_path = ""
    data_base_path += "iid_" if args.iid else "niid_"
    data_base_path += "s" if args.dataset_size == "small" else "l"

    global_model = femnistmodel()
    data_train = loadfemnist_raw(data_base_path + "/train")
    data_test = loadfemnist_raw(data_base_path + "/test")
    model = train(args, global_model, data_train, data_test)
    # test(args)
