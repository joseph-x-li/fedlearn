import json
import os
import torch
from torch.utils.data import Dataset

def loadfemnist_raw(dir):
    dir = "/home/ubuntu/Github/fedlearn/data/femnist/" + dir
    data_files = [x for x in os.listdir(dir) if ".json" in x]

    datas = []
    n_users = n_samples = 0
    users = dict() # user name -> idx in datas

    for idx, file in enumerate(data_files):
        with open(os.path.join(dir, file), 'r') as f:
            content = f.read()
        datadict = json.loads(content)
        datas.append(datadict)
        for user in datadict['users']:
            users[user] = idx
            n_users += 1
        for smp in datadict['num_samples']: 
            n_samples += smp

    print(f"Loaded raw data from: {dir}")
    print(f"{n_samples} samples")
    print(f"{n_users} users")

    return n_users, n_samples, users, datas

class LeafFEMNISTDataset(Dataset):
    def __init__(self, raw_data, user):
        n_users, n_samples, users, datas = raw_data
        datasidx = users[user]
        blockdata = datas[datasidx]
        self.data = blockdata['user_data'][user]
        self.len = len(self.data['x'])
        # for idx, u in enumerate(blockdata['users']):
        #     if u == user:
        #         self.len = blockdata['num_samples'][idx]
        #         break

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        xdata = torch.tensor(self.data['x'][idx])
        ydata = torch.tensor(self.data['y'][idx])
        return xdata.reshape((1, 28, 28)), ydata

if __name__ == "__main__":
    _, _, users, _ = raw_data = loadfemnist_raw("iid_s/train")
    user = list(users.keys())[0]
    dataset = LeafFEMNISTDataset(raw_data, user)
    import pdb; pdb.set_trace()