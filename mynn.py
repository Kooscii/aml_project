import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
# from torchvision import utils
import torch.legacy.nn
from scipy.io import arff
import xml.etree.ElementTree as ET
from random import random as rand

using_cuda = True

class myDataset():
    def __init__(self, _name):
        name = 'datasets/'+_name+'/'+_name
        # get the number of K
        labels = ET.parse(name+'.xml')
        K = sum(1 for _ in labels.getroot())

        # generate train loader
        data, meta = arff.loadarff(name+'-train.arff')
        trnName = []
        trnFeature = []
        trnLabel = []
        for x in data:
            x = list(x)
            if _name == 'genbase':
                trnName.append(x[0])
                trnFeature.append(list(map(lambda x: 1 if x ==b'YES' else 0, x[1:-K])))
            elif _name == 'scene':
                trnFeature.append(x[0:-K])
            elif _name == 'yeast':
                trnFeature.append(x[0:-K])
            trnLabel.append(list(map(int, x[-K:])))
        trnFeature = np.array(trnFeature)
        trnLabel = np.array(trnLabel)
        # trnLabel = np.array(list(map(assign_label, trnLabel)))
        trnFeature = th.from_numpy(trnFeature).float()
        trnLabel = th.from_numpy(trnLabel).float()
        self.trnSet = TensorDataset(trnFeature, trnLabel)
        # self.trnLoader_unshuf = DataLoader(self.trnSet, batch_size=1, shuffle=False)

        # generate test loader
        data, meta = arff.loadarff(name+'-test.arff')
        tstName = []
        tstFeature = []
        tstLabel = []
        for x in data:
            x = list(x)
            if _name == 'genbase':
                tstName.append(x[0])
                tstFeature.append(list(map(lambda x: 1 if x == b'YES' else 0, x[1:-K])))
            elif _name == 'scene':
                tstFeature.append(x[0:-K])
            elif _name == 'yeast':
                tstFeature.append(x[0:-K])
            tstLabel.append(list(map(int, x[-K:])))
        tstFeature = np.array(tstFeature)
        tstLabel = np.array(tstLabel)
        tmp = tstLabel
        # tstLabel = np.array(list(map(assign_label, tstLabel)))
        # multi = 0
        # for i in tstLabel:
        #     multi += 1 if np.sum(i)>1 else 0
        #     print(i)
        # print(1-multi/len(tstLabel)) 
        # print(multi)
        tstFeature = th.from_numpy(tstFeature).float()
        tstLabel = th.from_numpy(tstLabel).float()
        self.tstSet = TensorDataset(tstFeature, tstLabel)
        # self.tstLoader_unshuf = DataLoader(self.tstSet, batch_size=1, shuffle=False)

    def getTrnLoader(self, shuf=False):
            return DataLoader(self.trnSet, batch_size=1, shuffle=shuf)

    def getTstLoader(self, shuf=False):
            return DataLoader(self.tstSet, batch_size=1, shuffle=shuf)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # # if database == 'genbase':
        # self.l1 = nn.Linear(1185, 1200)
        # self.l2 = nn.Linear(1200, 200)
        # # self.l3 = nn.Linear(400, 50)
        # self.l4 = nn.Linear(200, 2)
        # # elif database == 'scene':
        # self.l1 = nn.Linear(294, 200)
        # self.l2 = nn.Linear(200, 50)
        # # self.l3 = nn.Linear(400, 50)
        # self.l4 = nn.Linear(50, 2)
        # # elif database == 'yeast':
        self.l1 = nn.Linear(103, 50)
        self.l2 = nn.Linear(50, 2)
        # self.l3 = nn.Linear(400, 50)
        self.l4 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        x = self.l4(x)
        if not self.training:
            x = F.sigmoid(x)
        return x


class Classifier(object):
    def __init__(self):
        self.model = Net()
        self.loss = nn.MultiLabelSoftMarginLoss()
        self.optimizer = optim.Adadelta(self.model.parameters())
        if using_cuda and torch.cuda.is_available():
            self.model.cuda()
            self.loss.cuda()

    # make prediction
    def __call__(self, data):
        self.model.eval()
        if using_cuda and torch.cuda.is_available():
            data = data.cuda()
        data = Variable(data)

        out = self.model(data)
        pred = np.where(out.data.cpu().numpy() > 0.5, 1, 0)[0]
        # r = rand() * 3
        # if r < 1:
        #     rand_pred = [1, 1]
        # elif r > 2:
        #     rand_pred = [0, 1]
        # else:
        #     rand_pred = [1, 0]
        return pred

    def train(self, data, label):
        label = th.FloatTensor([label])

        self.model.train()

        if using_cuda and torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss(output, label)
        loss.backward()
        self.optimizer.step()

        # pred = np.where(out.data.cpu().numpy() > 0.5, 1, 0)
        out = output.data.cpu().view(-1).numpy()
        label = label.data.cpu().view(-1).numpy()
        # print('output:', out, ' label: ', label)





