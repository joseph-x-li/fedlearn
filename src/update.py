import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import LeafFEMNISTDataset

def test_inference(args, model, raw_data_test):
    """ Returns a summary??? of test accuracy and loss.
    """
    _, n_samples, users, _ = raw_data_test
    all_users = list(users.keys())

    loss, total, correct = 0.0, 0.0, 0.0
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    for user in all_users:
        testloader = DataLoader(LeafFEMNISTDataset(raw_data_test, user), batch_size=128, shuffle=False)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)

                # Inference
                logits = model(images)
                batch_loss = criterion(logits, labels)
                loss += batch_loss.item()

                # Prediction
                _, indicies = torch.max(logits, 1)
                indicies = indicies.view(-1) # same as flatten
                correct += torch.sum(torch.eq(indicies, labels)).item()
                total += len(labels)

    assert total == n_samples
    accuracy = correct / total
    return accuracy, loss / total

class LocalUpdate:
    def __init__(self, args, raw_data, user):
        self.args = args
        self.trainloader = DataLoader(
            LeafFEMNISTDataset(raw_data, user), 
            batch_size=self.args.local_bs, 
            shuffle=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def update_weights(self, model):
        # Set mode to train model
        model.train()
        epoch_loss = []

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = model(images)
                batch_loss = self.criterion(logits, labels)
                loss += batch_loss.item()

                # Prediction
                _, indicies = torch.max(logits, 1)
                indicies = indicies.view(-1) # same as flatten
                correct += torch.sum(torch.eq(indicies, labels)).item()
                total += len(labels)

        accuracy = correct / total
        return accuracy, loss / total