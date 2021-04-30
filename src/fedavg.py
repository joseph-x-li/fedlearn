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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# p fedavg.py --epochs 5 --frac 0.1 --local_ep 6 --local_bs 32 

def train(args, global_model, raw_data_train, raw_data_test):
    start_time = time.time()
    user_list = list(raw_data_train[2].keys())
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    print(global_model)


    m = max(int(args.frac * len(user_list)), 1)
    print(f"Training {m} users each round")

    train_loss, train_accuracy = [], []
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        print(f'Global Training Round : {epoch+1}\n')

        global_model.train()
        
        sampled_users = random.sample(user_list, m)

        for user in sampled_users:
            local_model = LocalUpdate(args=args, raw_data=raw_data_train, user=user)
            w, loss = local_model.update_weights(copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for user in user_list:
            local_model = LocalUpdate(args=args, raw_data=raw_data_test, user=user)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)

        train_accuracy.append(sum(list_acc)/len(list_acc))

        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f'Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

def test(args, model):
    pass
# raw_data_test = loadfemnist_raw("niid_small/test")
if __name__ == "__main__":
    args = args_parser()
    global_model = femnistmodel()
    raw_data_train = loadfemnist_raw("niid_s/train")
    raw_data_train = loadfemnist_raw("niid_s/test")
    model = train(args, global_model)
    # test(args)