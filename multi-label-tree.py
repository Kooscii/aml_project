import numpy as np
from mynn import myDataset
from mytree import Tree

dataset = myDataset('yeast')
trnLoader = dataset.getTrnLoader(shuf=False)
tstLoader = dataset.getTstLoader(shuf=False)

ptree = Tree()

for epoch in range(20):
    print('Epoch', epoch)
    trnLoader = dataset.getTrnLoader(shuf=True)
    for idx, (data, label) in enumerate(trnLoader):
        label = np.where(label.view(-1).numpy() == 1)[0].tolist()
        # print('new example: ', label)
        ptree.learn(data, label)
        if idx % 100 == 0:
            print('%.1f' % (idx * 100 / len(trnLoader)) + '% trained')

    correct = 0
    for idx, (data, label) in enumerate(tstLoader):
        label = np.where(label.view(-1).numpy() == 1)[0].tolist()
        pred = ptree.predict(data)
        pred.sort()
        if label == pred:
            correct += 1
        # print('tst:', idx)
        # print('labl:', label)
        # print('pred:', pred)
        #
    print(correct / len(tstLoader))
pass
