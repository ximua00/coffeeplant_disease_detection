from torchvision.models import alexnet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import os

from dataloader import get_dataloaders
from utils import make_directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net():
    def __init__(self):
        self.model = nn.Sequential(
            alexnet(pretrained=True), nn.Linear(1000, 6))
        self.model[0].features.require_grad = False
        self.model = self.model.to(device)


class Model():
    def __init__(self, learning_rate=1e-3, batch_size=32, scheduler_stepsize=20):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.scheduler_stepsize = scheduler_stepsize

        self.dataloaders = get_dataloaders(self.batch_size)
        self.model = Net().model.to(device)

    def train(self, n_epochs):
        self.config_train()
        for epoch in range(n_epochs):
            train_loss, train_accuracy = self.train_epoch(self.dataloaders["train"])
            eval_loss, eval_accuracy = self.eval(self.dataloaders["eval"])
            test_loss, test_accuracy = self.eval(self.dataloaders["eval"])
            self.scheduler.step()
            print("TRAIN: Loss {} Accuracy {}".format(train_loss, train_accuracy))
            print("EVAL: Loss {} Accuracy {}".format(eval_loss, eval_accuracy))
            print("TEST: Loss {} Accuracy {}".format(test_loss, test_accuracy))

    def config_train(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([{'params': self.model[1].parameters(), 'lr': 1e-3},
                                     {'params': self.model[0].classifier.parameters(), 'lr': 1e-3}], lr=0.0)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.1)

    def train_epoch(self, train_dataloader):
        cum_loss = 0.0
        accurates = 0
        for image, target in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            accurates += self.evaluate_accuracy(output, target)
            cum_loss += loss.item()

        return cum_loss/len(train_dataloader), accurates/len(train_dataloader.dataset)

    @torch.no_grad()
    def eval(self, dataloader):
        cum_loss = 0.0
        accurates = 0
        for image, target in dataloader:
            output = self.model(image)
            loss = self.criterion(output, target)
            cum_loss += loss.item()
            accurates += self.evaluate_accuracy(output, target)

        return cum_loss/len(dataloader), accurates/len(dataloader.dataset)

    def evaluate_accuracy(self, output, target):
        predicted = torch.argmax(output, 1)
        correct = (predicted == target).sum().item()
        return correct

    def save_model(self, model_name):
        path = make_directory("../models")
        torch.save(self.model.state_dict(),
                   os.path.join(path, model_name + ".pt"))


if __name__ == "__main__":
    model = Model()
    model.train(20)
    model.save_model("test")
