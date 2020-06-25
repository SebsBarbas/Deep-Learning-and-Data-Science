#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random

GDParams={'n_batch' : 100, 'n_epoch' : 40, 'eta' : 0.001}

np.random.seed(300)
random.seed(300)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    data_X = dict[b'data']

    test_labels = dict[b'labels']
    return data_X, test_labels

def normalize(data, mean_x, std_x):
    data_norm = data / np.max(data)
    data_norm = data_norm - mean_x
    data_norm = data_norm / std_x

    return data_norm


def oneHotEncoder(data):
    label_encoder = LabelEncoder()
    values = np.array(data)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def PlotAccuracyGradients(acc_W1, acc_W2, acc_b1, acc_b2):
    plt.plot([1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_W1), [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_W2), [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_b1), [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_b2))
    plt.legend(['W1', 'W2', 'b1', 'b2'])
    plt.xlabel('Threshold level')
    plt.ylabel('Accuracy of the gradient')
    plt.title('Accuracy in the derivative calculations')
    plt.show()

def Plot_Cost_Loss(cost, cost_val, loss, loss_val, acc, acc_val):

    fig1, axs1 = plt.subplots(1, 3)
    fig1.suptitle('Cost plot')
    axs1[0].plot(np.array(range(len(cost)))*80, cost, np.array(range(len(cost)))*80, cost_val)
    axs1[0].legend(['Cost in training set', 'Cost in validation set'])
    axs1[0].set_title('Cost values')
    axs1[0].set_xlabel('Number of steps')
    axs1[0].set_ylabel('Cost')

    axs1[1].plot(np.array(range(len(cost)))*80, loss, np.array(range(len(cost)))*80, loss_val)
    axs1[1].legend(['Loss in training set', 'Loss in validation set'])
    axs1[1].set_title('Loss values')
    axs1[1].set_xlabel('Number of steps')
    axs1[1].set_ylabel('Loss')

    axs1[2].plot(np.array(range(len(cost)))*80, acc, np.array(range(len(cost)))*80, acc_val)
    axs1[2].legend(['Accuracy in training set', 'Accuracy in validation set'])
    axs1[2].set_title('Accuracy values')
    axs1[2].set_xlabel('Number of steps')
    axs1[2].set_ylabel('Loss')
    fig1.show()

class Network:

    def __init__(self, n_inputs, nodes, outputs, eta_min = 1e-5, eta_max = 1e-1):
        mu = 0
        self.Ws = []
        self.bs = []

        W = np.random.normal(mu, 1 / np.sqrt(n_inputs), (nodes, n_inputs))
        b = np.random.normal(mu, 1 / np.sqrt(n_inputs), (nodes,1))

        self.Ws.append(W.copy())
        self.bs.append(b.copy())

        W = np.random.normal(mu, 1 / np.sqrt(nodes), (outputs, nodes))
        b = np.random.normal(mu, 1 / np.sqrt(nodes), (outputs, 1))

        self.Ws.append(W.copy())
        self.bs.append(b.copy())

        self.eta_min = eta_min
        self.eta_max = eta_max

        self.ens = []
        self.Dropout = False

    def get_weights(self):
        return self.Ws, self.bs

    def act_relu(self, input):
        indx = np.where(input < 0)
        output = input.copy()
        output[indx] = 0
        return output

    def EvaluateClassifier(self, X, W, b):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0] * X.shape[1] * X.shape[2], X.shape[3])
        out = np.matmul(W, X) + b
        soft = np.exp(out) / (sum(np.exp(out)))
        return soft

    def ComputeCost(self, X, lab, W, b, lamb):
        out_layer_1 = np.matmul(W[0], X.T) + b[0]
        act_layer_1 = self.act_relu(out_layer_1)
        soft = self.EvaluateClassifier(act_layer_1, W[1], b[1])

        return sum(sum(lab.T * (-np.log(soft)))) / X.shape[0] + lamb * (sum(sum(W[0] ** 2)) + sum(sum(W[1]**2)))

    def NumericalGradientsFast(self, X, Y, h, lamb):
        grad_W = [None] * len(self.Ws)
        grad_b = [None] * len(self.bs)
        c = self.ComputeCost(X, Y, self.Ws, self.bs, lamb)

        W_work = []
        b_work = []
        for i in range(len(self.Ws)):
            W_work.append(self.Ws[i].copy())
            b_work.append(self.bs[i].copy())

        for k in range(len(self.bs)):
            grad_b[k] = np.zeros(self.bs[k].shape).copy()
            for i in range(grad_b[k].shape[0]):
                b_work[k][i] = b_work[k][i] + h
                c2 = self.ComputeCost(X, Y, W_work, b_work, lamb)
                grad_b[k][i] = (c2 - c) / h
                b_work[k] = self.bs[k].copy()

        for k in range(len(self.Ws)):
            grad_W[k] = np.zeros(self.Ws[k].shape).copy()
            for i in range(self.Ws[k].shape[0]):
                for j in range(self.Ws[k].shape[1]):
                    W_work[k][i, j] += h
                    c2 = self.ComputeCost(X, Y, W_work, b_work, lamb)
                    grad_W[k][i, j] = (c2 - c) / h
                    W_work[k] = self.Ws[k].copy()

        return grad_W, grad_b

    def NumericalGradientsSlow(self, X, Y, h, lamb):

        grad_W = [None] * len(self.Ws)
        grad_b = [None] * len(self.bs)


        b_work = []
        W_work = []
        for i in range(len(self.Ws)):
            W_work.append(self.Ws[i].copy())
            b_work.append(self.bs[i].copy())

        for k in range(len(self.Ws)):
            grad_b[k] = np.zeros(b_work[k].shape).copy()
            for i in range(b_work[k].shape[0]):
                b_work[k][i] -= h
                c1 = self.ComputeCost(X, Y, W_work, b_work, lamb)
                b_work[k][i] += 2 * h
                c2 = self.ComputeCost(X, Y, W_work, b_work, lamb)
                grad_b[k][i] = (c2 - c1) / (2 * h)
                b_work[k] = self.bs[k].copy()

        for k in range(len(self.Ws)):
            grad_W[k] = np.zeros(self.Ws[k].shape).copy()
            for i in range(self.Ws[k].shape[0]):
                for j in range(self.Ws[k].shape[1]):
                    W_work[k][i, j] -= h
                    c1 = self.ComputeCost(X, Y, W_work, b_work, lamb)
                    W_work[k][i, j] += 2 * h
                    c2 = self.ComputeCost(X, Y, W_work, b_work, lamb)
                    grad_W[k][i, j] = (c2 - c1) / (2 * h)
                    W_work[k] = self.Ws[k].copy()

        return grad_W, grad_b

    def forward(self, data, state = None):

        self.out1 = np.matmul(self.Ws[0], data.T) + self.bs[0]
        self.act_out1 = self.act_relu(self.out1)
        test = self.act_out1.copy()
        if self.Dropout == True and state == 'Train':
            probs = np.random.uniform(size = self.act_out1.shape[1]).reshape(-1, 1)
            indx = np.where(probs < self.p[0])
            self.act_out1[:, indx[0]] = 0
        if self.Dropout == True and state == 'Test':
            self.act_out1 = self.act_out1*self.p[0]



        self.out2 = np.matmul(self.Ws[1], self.act_out1) + self.bs[1]
        self.softmx = self.EvaluateClassifier(self.act_out1, self.Ws[1], self.bs[1])

    def Accuracy(self, X, Y, state = None):
        self.forward(X, state)
        max_perc = np.argmax(self.softmx, axis = 0)
        max_lab = np.argmax(Y, axis = 1)
        error = 0
        for i in range(max_perc.shape[0]):
            if max_perc[i] != max_lab[i]:
                error = error + 1

        return 1 - error / X.shape[0]


    def UpdateLearningRate(self, t, l, ns, Ensemble = False):

        if t < (2*l + 1)*ns:
            self.GDParams['eta'] = self.eta_min + (t - 2*l*ns)*(self.eta_max - self.eta_min)/ns
        else:
            self.GDParams['eta'] = self.eta_max - (t - (2*l + 1)*ns)*(self.eta_max - self.eta_min)/ns
            if t == 2*(l + 1)*ns:
                l = l + 1
                if Ensemble and l>=9:
                    self.ens.append([self.Ws.copy(), self.bs.copy()])

        return l

    def TestUpdate(self, X, GDParams):
        self.GDParams = GDParams
        self.GDParams['n_epoch'] = 30
        ns = 500
        t = 0
        l = 0
        values = []
        for i in range(self.GDParams['n_epoch']):

            for j in range(100):
                l = self.UpdateLearningRate(t, l, ns)
                values.append(self.GDParams['eta'])
                t = t + 1

        plt.plot(np.array(range(len(values))), np.array(values))
        plt.show()



    def MiniBatchGD(self, X, Y, Xval, Yval, GDParams, lamb, n_cycles, ns, adapt_learning, Ensemble = False, Dropout = False):

        self.Dropout = Dropout

        if self.Dropout == True:
            self.p = np.array(0.7).reshape(-1,1)

        self.GDParams = GDParams
        loss = []
        loss_val = []
        cost = []
        cost_val = []
        acc = []
        acc_val = []
        self.GDParams['n_epoch'] = 2*n_cycles*ns*self.GDParams['n_batch'] // X.shape[0]
        t = 0
        l = 0
        k = 0
        cost.append(self.ComputeCost(X, Y, self.Ws, self.bs, lamb))
        cost_val.append(self.ComputeCost(Xval, Yval, self.Ws, self.bs, lamb))
        loss.append(self.ComputeCost(X, Y, self.Ws, self.bs, 0))
        loss_val.append(self.ComputeCost(Xval, Yval, self.Ws, self.bs, 0))
        acc.append(self.Accuracy(X, Y))
        acc_val.append(self.Accuracy(Xval, Yval))

        for i in range(self.GDParams['n_epoch']):

            indx = np.random.permutation(range(X.shape[0]))
            X = X[indx]
            Y = Y[indx]

            for j in range(X.shape[0] // self.GDParams['n_batch']):
                l = self.UpdateLearningRate(t, l, ns, Ensemble)
                j_start = j * self.GDParams['n_batch']
                j_end = (j + 1) * self.GDParams['n_batch']
                Xbatch = X[j_start:j_end, :]
                Ybatch = Y[j_start:j_end, :]

                self.forward(Xbatch, 'Train')
                dJw1, dJw2, dJb1, dJb2 = self.back_prop(Xbatch, Ybatch, lamb)

                self.Ws[0] = self.Ws[0] - self.GDParams['eta'] * dJw1
                self.bs[0] = self.bs[0] - self.GDParams['eta'] * dJb1
                self.Ws[1] = self.Ws[1] - self.GDParams['eta'] * dJw2
                self.bs[1] = self.bs[1] - self.GDParams['eta'] * dJb2
                t = t + 1

                if k == 80:
                    cost.append(self.ComputeCost(X, Y, self.Ws, self.bs, lamb))
                    cost_val.append(self.ComputeCost(Xval, Yval, self.Ws, self.bs, lamb))
                    loss.append(self.ComputeCost(X, Y, self.Ws, self.bs, 0))
                    loss_val.append(self.ComputeCost(Xval, Yval, self.Ws, self.bs, 0))
                    acc.append(self.Accuracy(X, Y))
                    acc_val.append(self.Accuracy(Xval, Yval))
                    k = 0
                else:
                    k = k + 1


            if adapt_learning == True:
                self.GDParams['eta'] = self.GDParams['eta'] * 0.9

        if Ensemble == True:
            self.ens.append([self.Ws.copy(), self.bs.copy()])

        return cost, cost_val, loss, loss_val, acc, acc_val

    def Accuracy_Ensemble(self, vote, Y):
        out = vote - Y.T
        error = (np.where(out == -1)[0]).shape[0] / out.shape[1]
        return 1 - error

    def Ensemble_Learning(self, Xval, Yval):
        votes = np.zeros((self.Ws[1].shape[0], Xval.shape[0]))
        ohd = np.zeros((self.Ws[1].shape[0], Xval.shape[0]))
        for ens_meth in self.ens:
            self.Ws = ens_meth[0]
            self.bs = ens_meth[1]
            self.forward(Xval)

            max = np.argmax(self.softmx, axis = 0)
            for l in range(max.shape[0]):
                votes[max[l], l] += 1
        max = np.argmax(votes, axis = 0)
        for l in range(max.shape[0]):
            ohd[max[l], l] += 1

        accuracy = self.Accuracy_Ensemble(ohd, Yval)

        return accuracy

    def Rel_Error(self, grad_num, grad_anal, eps):
        err = np.abs(grad_num - grad_anal) / np.maximum(eps*np.ones(grad_num.shape), np.abs(grad_num) + np.abs(grad_anal))
        return err

    def CompareGradients(self, X, Y, e, h, lamb):
        acc = [1e-6, 1e-7, 1e-8, 1e-9]
        per_W1 = []
        per_b1 = []
        per_W2 = []
        per_b2 = []


        self.forward(X)
        dJw1, dJw2, dJb1, dJb2 = self.back_prop(X, Y, lamb)
        dJw = [dJw1, dJw2]
        dJb = [dJb1, dJb2]

        dJw_com, dJb_com = self.NumericalGradientsSlow(X, Y, h, lamb)

        for err in acc:
            err_W1 = self.Rel_Error(dJw_com[0], dJw[0], e)
            err_b1 = self.Rel_Error(dJb_com[0], dJb[0], e)
            err_W2 = self.Rel_Error(dJw_com[1], dJw[1], e)
            err_b2 = self.Rel_Error(dJb_com[1], dJb[1], e)
            per_W1.append(np.sum(err_W1 <= err)/(self.Ws[0].shape[0]*self.Ws[0].shape[1]))
            per_b1.append(np.sum(err_b1 <= err)/(self.bs[0].shape[0]*self.bs[0].shape[1]))
            per_W2.append(np.sum(err_W2 <= err)/(self.Ws[1].shape[0]*self.Ws[1].shape[1]))
            per_b2.append(np.sum(err_b2 <= err)/(self.bs[1].shape[0]*self.bs[1].shape[1]))

        return per_W1, per_W2, per_b1, per_b2


    def back_prop(self, X, Y, lamb):

        Gbatch = -(Y.T - self.softmx)

        dLw2 = np.matmul(Gbatch, self.act_out1.T) / self.act_out1.shape[1]
        dLb2 = np.matmul(Gbatch, np.ones(self.act_out1.shape[1])) / self.act_out1.shape[1]
        dJw2 = dLw2 + 2 * lamb * self.Ws[1]
        dJb2 = dLb2.reshape(-1,1)
        Gbatch = self.Ws[1].T @ Gbatch
        index = self.act_out1 > 0
        Ind = np.zeros(self.act_out1.shape)
        Ind[index] = 1
        Gbatch = Gbatch * Ind
        dLw1 = np.matmul(Gbatch, X) / X.shape[0]
        dLb1 = np.matmul(Gbatch, np.ones(X.shape[0])) / X.shape[0]
        dJw1 = dLw1 + 2 * lamb * self.Ws[0]
        dJb1 = dLb1.reshape(-1,1)
        return dJw1, dJw2, dJb1, dJb2

    def max_learning_rate(self, X, Y, min_l, max_l, GDParams, n_epochs, lamb):

        etas = np.linspace(min_l, max_l, n_epochs*(X.shape[0] // GDParams['n_batch']))
        acc = []
        for e in range(n_epochs):

            for j in range(X.shape[0] // GDParams['n_batch']):
                j_start = j * GDParams['n_batch']
                j_end = (j + 1) * GDParams['n_batch']
                Xbatch = X[j_start:j_end, :]
                Ybatch = Y[j_start:j_end, :]

                self.forward(Xbatch)
                dJw1, dJw2, dJb1, dJb2 = self.back_prop(Xbatch, Ybatch, lamb)
                a = etas[e * (X.shape[0] // GDParams['n_batch']) + j]
                self.Ws[0] = self.Ws[0] - etas[e*(X.shape[0] // GDParams['n_batch']) + j] * dJw1
                self.bs[0] = self.bs[0] - etas[e*(X.shape[0] // GDParams['n_batch']) + j] * dJb1
                self.Ws[1] = self.Ws[1] - etas[e*(X.shape[0] // GDParams['n_batch']) + j] * dJw2
                self.bs[1] = self.bs[1] - etas[e*(X.shape[0] // GDParams['n_batch']) + j] * dJb2
                acc.append(self.Accuracy(X, Y))

        plt.plot(etas, acc)
        plt.ylabel('Accuracies')
        plt.xlabel('Learning rates')
        plt.title('Test to determinate lower and upper limits of the learning rate')
        plt.show()




def lambda_search(X, Y, Xval, Yval, GDParams, n_cycles, ns):
    lamb_min = -5
    iter = [0, 1]
    lamb_max = -1
    l = lamb_min + (lamb_max-lamb_min)*np.random.rand(20)

    lamb = 10**l
    score = np.zeros(len(lamb))
    for i in iter:
        for j in range(len(l)):
            net = Network(X.shape[1], 50, 10)
            net.MiniBatchGD(X, Y, Xval, Yval, GDParams, lamb[j], n_cycles, ns, False)
            score[j] = net.Accuracy(Xval, Yval)
        best = sorted(zip(score, lamb), reverse = True)
        best = best[:2]
        best = np.array(best)
        pos_min = np.argmin(best[:, 1])
        pos_max = np.argmax(best[:, 1])
        lamb_min = best[pos_min, 1]
        lamb_max = best[pos_max, 1]
        lamb = lamb_min + (lamb_max-lamb_min)*np.random.rand(20)
    lamb_max = best[0, 1]

    return lamb_max, best[0, 0]

def best_hyper(X, Y, Xval, Yval, GDParams):
    n_cycles = [2, 4, 6]
    batch_size = [50, 100]
    ns = [500, 800]
    b_acc = 0
    b_batch = None
    b_cyc = None
    b_lamb = 0
    for c in n_cycles:
        for b in batch_size:
            for n in ns:
                GDParams['n_batch'] = b
                best_lambd, accuracy = lambda_search(X, Y, Xval, Yval, GDParams, c, n)

                if accuracy > b_acc:
                    b_acc = accuracy
                    b_batch = b
                    b_cyc = c
                    b_lamb = best_lambd
                    b_ns = n

    return b_acc, b_batch, b_cyc, b_lamb, b_ns






if __name__ == "__main__":

    filetrain1 = "C:\cifar-10-batches-py\data_batch_1"
    filetrain2 = "C:\cifar-10-batches-py\data_batch_2"
    filetrain3 = "C:\cifar-10-batches-py\data_batch_3"
    filetrain4 = "C:\cifar-10-batches-py\data_batch_4"
    filetrain5 = "C:\cifar-10-batches-py\data_batch_5"
    filetest = "C:\\cifar-10-batches-py\\test_batch"


    train_X, train_lab = unpickle(filetrain1)

    train_X2, train_lab2 = unpickle(filetrain2)
    train_X3, train_lab3 = unpickle(filetrain3)
    train_X4, train_lab4 = unpickle(filetrain4)
    train_X5, train_lab5 = unpickle(filetrain5)
    train_X = np.concatenate([train_X, train_X2, train_X3, train_X4, train_X5], axis = 0)
    train_lab.extend(train_lab2)
    train_lab.extend(train_lab3)
    train_lab.extend(train_lab4)
    train_lab.extend(train_lab5)



    #val_X, val_lab = unpickle(filetrain2)


    val_X = train_X[train_X.shape[0]-1000:,:]
    train_X = train_X[:train_X.shape[0] - 1000, :]
    val_lab = train_lab[len(train_lab)-1000:]
    train_lab = train_lab[:len(train_lab)-1000:]



    test_X, test_lab = unpickle(filetest)

    mean_X = np.mean(train_X / np.max(train_X))
    std_X = np.std(train_X / np.max(train_X))
    normTrain = normalize(train_X, mean_X, std_X)
    normVal = normalize(val_X, mean_X, std_X)
    normTest = normalize(test_X, mean_X, std_X)

    h = 1e-5
    eps = h
    ohd = oneHotEncoder(train_lab)
    ohd_val = oneHotEncoder(val_lab)
    ns = 800
    #max_lamb, acc_val = lambda_search(normTrain, ohd, normVal, ohd_val, GDParams, 2, ns)

    #lamb = 0.00026227 #from previous experiments

    new_max_lamb = 0.01504577 #for the accurate learning rates
    #lamb = new_max_lamb
    #b_acc, b_batch, b_cyc, b_lamb, b_ns = best_hyper(normTrain, ohd, normVal, oneHotEncoder(val_lab), GDParams)
    # b_acc = 0.5292, b_batch = 100, b_cyc = 4, b_lamb = 0.00026227,  b_ns = 800
    #GDParams['n_batch'] = b_batch
    #net = Network(normTrain.shape[1], 50, np.max(val_lab) + 1)
    #net.max_learning_rate(normTrain, ohd, 0, 0.1, GDParams, 8, lamb)
    eta_min = 0.00225
    eta_max = 0.019
    net = Network(normTrain.shape[1], 50, np.max(val_lab) + 1, eta_min, eta_max)

    cost, cost_val, loss, loss_val, acc, acc_val = net.MiniBatchGD(normTrain, ohd, normVal, oneHotEncoder(val_lab), GDParams, lamb, 14, ns, False, False, False)
    #acc_ens = net.Ensemble_Learning(test_X, oneHotEncoder(test_lab))
    acc2 = net.Accuracy(normTest, oneHotEncoder(test_lab), 'Test')
    #per_W1, per_W2, per_b1, per_b2 = net.CompareGradients(testData, ohd[0:2, :], eps, h, lamb)
    Plot_Cost_Loss(cost, cost_val, loss, loss_val, acc, acc_val)
    #PlotAccuracyGradients(per_W1, per_W2, per_b1, per_b2)
