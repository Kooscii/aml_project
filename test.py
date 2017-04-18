class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(1185, 1200)
        self.l2 = nn.Linear(1200, 200)
        # self.l3 = nn.Linear(400, 50)
        self.l4 = nn.Linear(200, 2)
        self.loss = nn.MultiLabelSoftMarginLoss()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        x = self.l4(x)
        if not self.training:
            x = F.sigmoid(x)
        return x




def __init__(self, _labl):
        print('new tree:')
        self.lt = None
        self.rt = None
        self.isleaf = False
        leftK = np.power(2, int(np.log2(_labl.size-1)))
        rightK = _labl.size - leftK
        self.K = np.array([leftK, rightK])
        print('K: ', self.K)
        # assign label manually for now
        leftL = np.arange(0, self.K[0]) + _labl[0]
        rightL = np.arange(self.K[0], _labl.size) + _labl[0]
        self.L = [leftL, rightL]
        print('L: ', self.L)
        self.model = Net()
        self.model.cuda()
        self.optimizer = optim.Adadelta(self.model.parameters())


    def train(self, epoch, loader):
        self.model.train()
        cnt = 0
        for batch_idx, (data, target) in enumerate(loader):
            # reform target
            t = target.numpy()[:,np.array([self.L[0],self.L[1]])]
            if np.any(t) == 0:
                continue
            else:
                cnt += 1  
            target = th.from_numpy(np.any(t, 2)*1).float()

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.model.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(loader.dataset),
                        100. * batch_idx / len(loader), loss.data[0]))


    def test(self, epoch, loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        cnt = 0
        for data, target in tstLoader:
            tmp = target.numpy()
            # reform target
            t = target.numpy()[:,np.array([self.L[0],self.L[1]])]
            if np.any(t) == 0:
                continue
            else:
                cnt += 1
            target = th.from_numpy(np.any(t, 2)*1).float()

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            loss = self.model.loss(output, target)
            test_loss += loss.data[0]

            out = np.where(output.data.cpu().numpy()>0.5,1,0)
            tar = np.where(target.data.cpu().numpy()==1,1,0)
            correct += 1 if np.sum(out!=tar)==0 else 0

            if not np.sum(out!=tar)==0:
                print('wrong >>>')
                print(tmp)
                print(output.data.cpu().numpy())
                print(tar)

            # elif np.sum(tar)>1:
            #     print(output.data.cpu().numpy())
            #     print(tar)
            # pred = output.data.max(1)[1]
            # correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(tstLoader)
        print('Epoch: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, test_loss, correct, cnt,
            100. * correct / cnt))


        # model = Net()
# model.cuda()
# optimizer = optim.Adadelta(model.parameters())




# for epoch in range(1, 100 + 1):
#     trnLoader = DataLoader(trnSet, batch_size=1, shuffle=True)
#     train(epoch)
#     test(epoch)

root = Tree(np.arange(0,4))
root.lt = Tree(root.L[0])
root.rt = Tree(root.L[1])

for i in range(1, 5+1):
    trnLoader = DataLoader(trnSet, batch_size=1, shuffle=True)
    root.train(i, trnLoader)
    root.test(i, tstLoader)

for i in range(1, 5+1):
    trnLoader = DataLoader(trnSet, batch_size=1, shuffle=True)
    root.lt.train(i, trnLoader)
    root.lt.test(i, tstLoader)

for i in range(1, 5+1):
    trnLoader = DataLoader(trnSet, batch_size=1, shuffle=True)
    root.rt.train(i, trnLoader)
    root.rt.test(i, tstLoader)

correct = 0
for data, target in tstLoader:
    tmp = target
    t = target.numpy()[:,np.array([root.L[0],root.L[1]])]
    target = th.from_numpy(np.any(t, 2)*1).float()

    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)

    output = root.model(data)
    out_root = np.where(output.data.cpu().numpy()>0.5,1,0)

    out_lt = np.array([[0, 0]])
    if out_root[0][0]:
        output = root.lt.model(data)
        out_lt = np.where(output.data.cpu().numpy()>0.5,1,0)

    out_rt = np.array([[0, 0]])
    if out_root[0][1]:
        output = root.rt.model(data)
        out_rt = np.where(output.data.cpu().numpy()>0.5,1,0)

    print(' ')
    tar = tmp.view(-1).long().numpy()
    out = np.append(out_lt[0], out_rt[0])

    correct += 1 if np.sum(tar!=out)==0 else 0
    print(tar)
    print(out)

print(correct/199.0)

def assign_label(_labl):
    l1 = np.any(_labl[0:8])*1
    l2 = np.any(_labl[8:16])*1
    l3 = np.any(_labl[16:24])*1
    l4 = np.any(_labl[24:27])*1
    return np.array([l1, l2, l3, l4])