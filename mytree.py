from mynn import Classifier
from queue import LifoQueue


class Node(object):
    def __init__(self):
        self.lt = None              # left subtree root
        self.rt = None              # right subtree root
        self.isleaf = True          # if this node is a leaf
        self.K = [0, 0]             # sub tree labels count
        self.L = [[], []]           # sub tree labels list
        self.classifier = None      # classifier handler

    def __call__(self, data):
        if self.isleaf:
            return [1, 0]
        else:
            return self.classifier(data)

    """ 
        This function splits the labels into left subtree labels 
        and right labels. If new labels exist, add them into node.L 
        by applying our assign-new-algorithm.
        
    """
    def split(self, label):
        # find out new labels
        newlabel = list(set(label)-set(self.L[0])-set(self.L[1]))
        if newlabel:
            self.assign_new_algorithm(newlabel)
            # if current node is a leaf, check if it grows
            if self.isleaf:
                self.grow()
        # return sub labels
        sublabel = [[], []]
        sublabel[0] = list(set(label) - set(self.L[1]))
        sublabel[1] = list(set(label) - set(sublabel[0]))
        return sublabel

    def grow(self):
        if all(self.K):
            # upgrade to branch
            self.isleaf = False
            self.classifier = Classifier()
            # grow leaves
            self.lt = Node()
            self.rt = Node()
            # TODO: too tricky here
            self.lt.K[0] = 1
            self.lt.L[0] = [self.L[0][0]]

    def train(self, data, label):
        self.classifier.train(data, label)

    # TODO
    """ 
        Our new labels assignment algorithm
        
    """
    def assign_new_algorithm(self, newlabel):
        for l in newlabel:
            if self.K[0] <= self.K[1]:
                self.L[0].append(l)
                self.K[0] += 1
            else:
                self.L[1].append(l)
                self.K[1] += 1


class Tree(object):
    def __init__(self):
        self.root = Node()

    def predict(self, data, node=None):
        # node = root if node not assigned
        node = self.root if node == None else node

        predict_q = LifoQueue()
        predict_q.put(node)
        label = []

        while not predict_q.empty():
            node = predict_q.get()

            if node.isleaf:
                label.extend(node.L[0])
            else:
                pred = node(data)
                if pred[1]:
                    predict_q.put(node.rt)
                if pred[0]:
                    predict_q.put(node.lt)

        return label

    def learn(self, data, label, node=None):
        # node = root if node not assigned
        node = self.root if not node else node

        learn_q = LifoQueue()
        learn_q.put((label, node))

        while not learn_q.empty():
            (label, node) = learn_q.get()

            # assign labels
            sublabel = node.split(label)
            # if isbranch, train the node
            if not node.isleaf:
                # print('train node(', node.L, ')')
                node.train(data, list(map(lambda x: 1 if x else 0, sublabel)))

                # if this data has labels belong to left subtree
                if sublabel[1]:
                    # left subtree learns this data
                    learn_q.put((sublabel[1], node.rt))
                # if this data has labels belong to right subtree
                if sublabel[0]:
                    # right subtree learns this data
                    learn_q.put((sublabel[0], node.lt))


