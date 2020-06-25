#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
import copy
import pickle

GDParams = {'n_batch': 100, 'n_epoch': 2000, 'eta': 0.001, 'rho': 0.9}

np.random.seed(380)
random.seed(380)


def unpickle(file):
    f = open(file, "r")
    data = []
    y = []
    for lines in f:
        fields = lines.split(" ")
        if len(fields) > 2:
            string = ''
            for i in range(len(fields) - 1):
                string = string + fields[i] + ' '

            string = string[0:len(string) - 1]
            fields[0] = string
        data.append(fields[0].lower())
        y.append(fields[-1][0:len(fields[-1]) - 1])

    for d in range(len(y)):
        if len(y[d]) == 1:
            y[d] = ord(y[d][0]) - ord('1')
        else:
            y[d] = ord(y[d][1]) - ord('1') + 10

    return data, y


def unpickle2(file):
    f = open(file, "r")
    for lines in f:
        fields = lines.split(" ")
        data = fields

    for i in range(len(data)):
        data[i] = int(data[i])

    return data


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


def oneHotEncoderNames(data, d, nl):
    onehot_encoded = []

    for dat in data:
        one_hot = np.zeros((d, nl))
        for i in range(len(dat)):
            if i < nl:
                pos = ord(dat[i]) - ord('a')
                if pos == -58:  # There is a surname d'alembert which gives this ascii number with the '''
                    pos = 27

                if pos == -65:  # This is for the ascii codes ' '
                    pos = 28
                one_hot[pos][i] = 1
            else:
                break

        onehot_encoded.append(one_hot.copy())

    return onehot_encoded


def max_length(data):
    max = 0
    for d in data:
        if len(d) > max:
            max = len(d)

    return max


def PlotAccuracyGradients(acc_F1, acc_F2, acc_W, n_layer_1, n_layer_2):#acc_F3, acc_W, n_layer_1, n_layer_2):
    plt.plot([1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_F1), [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_F2), #[1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_F3),
             [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_W))
    plt.legend(['F1', 'F2', 'W']) #'F3', 'W'])
    plt.xlabel('Threshold level')
    plt.ylabel('Accuracy of the gradient')
    plt.title(
        'Accuracy in the derivative calculations with ' + str(n_layer_1) + ' filters in the first layer and ' + str(
            n_layer_2) + ' in the second layer')
    plt.show()


def Plot_Cost_Loss(cost, cost_val, acc, acc_val):
    fig1, axs1 = plt.subplots(1, 2)
    fig1.suptitle('Cost plot')
    axs1[0].plot(np.array(range(len(cost))) * 500, cost, np.array(range(len(cost))) * 500, cost_val)
    axs1[0].legend(['Cost in training set', 'Cost in validation set'])
    axs1[0].set_title('Cost values')
    axs1[0].set_xlabel('Number of steps')
    axs1[0].set_ylabel('Cost')

    axs1[1].plot(np.array(range(len(cost))) * 500, acc, np.array(range(len(cost))) * 500, acc_val)
    axs1[1].legend(['Accuracy in training set', 'Accuracy in validation set'])
    axs1[1].set_title('Accuracy values')
    axs1[1].set_xlabel('Number of steps')
    axs1[1].set_ylabel('Accuracy')
    fig1.show()
    fig1.savefig('cost.pdf')


class ConvNetwork:

    def __init__(self, nlayers, n_filters_layer, ks, d, K, nlen, GDParams, X_input, start, bias = False, strides = None, Pads = None):

        self.n_filters_layer = n_filters_layer
        self.nlayers = nlayers
        self.ks = ks
        self.bias = bias

        if strides == None or Pads == None:
            self.mode_stride = False
        else:
            self.mode_stride = True
            self.strides = strides
            self.Pads = Pads
            self.Pads.append(0)

        self.nlen = []  # Tiene un numero mas de entradas que el resto ya que W depende de nlen

        if self.mode_stride == False:
            for i in range(nlayers + 1):
                if i == 0:
                    self.nlen.append(nlen)
                else:
                    self.nlen.append(self.nlen[i - 1] - self.ks[i - 1] + 1)

        else:
            for i in range(nlayers + 1):
                if i == 0:
                    self.nlen.append(nlen + 2*self.Pads[i])
                else:
                    self.nlen.append((self.nlen[i - 1] - self.ks[i - 1]) // self.strides[i - 1]  + 1 + 2*self.Pads[i])


        self.data_size = d
        self.classes = K
        self.GDParams = GDParams
        '''
        inps =[]
        for k in range(len(X_input)):
            inps.append(len(np.where(X_input[k] != 0)[0]))
        '''

        # After testing, it was seen that the mean length of the inputs is of 7.14 letters, using it as a constant in order to not have to repeat this process

        # We initialize with He's algorithm
        self.sig = []
        self.F = []
        self.mu = 0

        for k in range(nlayers):
            if k == 0:
                self.sig.append(2 / 7.1405)
                self.F.append(np.random.normal(self.mu, np.sqrt(self.sig[k]), (d, ks[k], n_filters_layer[k])))

            else:
                self.sig.append(2 / (self.nlen[k] * self.n_filters_layer[k - 1]))
                self.F.append(
                    np.random.normal(self.mu, np.sqrt(self.sig[k]), (n_filters_layer[k - 1], ks[k], n_filters_layer[k])))

        self.sig.append(2 / (self.nlen[-1] * self.n_filters_layer[-1]))
        self.W = np.random.normal(self.mu, np.sqrt(self.sig[-1]), (K, n_filters_layer[-1] * self.nlen[-1]))

        if self.bias == True:
            self.bFs = []
            self.bW = np.random.normal(self.mu, np.sqrt(self.sig[-1]), (K, 1))
            for k in range(nlayers):
                if self.mode_stride == True:
                    self.bFs.append(np.random.normal(self.mu, np.sqrt(self.sig[k]), ((self.nlen[k + 1] - 2*self.Pads[k + 1])*self.n_filters_layer[k], 1)))
                else:
                    self.bFs.append(np.random.normal(self.mu, np.sqrt(self.sig[k]), ((self.nlen[k + 1]) * self.n_filters_layer[k], 1)))



        self.MXinputs_glob = []

        self.start = start

        self.xin_global = X_input[0].reshape((-1, 1), order = 'F')

        if self.mode_stride == True and self.Pads[0] != 0:
            for i in range(len(X_input)):
                X_input[i] = np.concatenate([np.zeros((self.data_size, self.Pads[0])), X_input[i]], axis = 1)
                X_input[i] = np.concatenate([ X_input[i], np.zeros((self.data_size, self.Pads[0]))], axis = 1)


        for i in range(len(X_input) - 1):
            self.xin_global = np.concatenate([self.xin_global, X_input[i + 1].reshape((-1, 1), order = 'F')], axis = 1)

        if self.mode_stride == False:

            for k in range(len(X_input)):
                #self.MXinputs_glob.append(self.MXMatrix(X_input[k], nlen, ks[0], n_filters_layer[0]).copy())
                self.MXinputs_glob.append((self.MXMatrix_Efficient(X_input[k], d, ks[0])).copy())

        else:
            for k in range(len(X_input)):
                self.MXinputs_glob.append(self.MXMatrix_Efficient(X_input[k], d, ks[0], self.strides[0], self.Pads[1], self.n_filters_layer[0]).copy())


        self.MXinputs = self.MXinputs_glob

    def new_GDParams(self, GDParams):
        self.GDParams = GDParams

    def update_structure(self, new_layers, new_ks, new_layers_filter, X_input):

        nlen = self.nlen[0]
        self.n_filters_layer = new_layers_filter
        self.nlayers = new_layers
        self.ks = new_ks
        self.nlen = []  # Tiene un numero mas de entradas que el resto ya que W depende de nlen
        for i in range(new_layers + 1):
            if i == 0:
                self.nlen.append(nlen)
            else:
                self.nlen.append(self.nlen[i - 1] - self.ks[i - 1] + 1)

        sig = []
        self.F = []
        mu = 0

        for k in range(self.nlayers):
            if k == 0:
                sig.append(2 / 7.1405)
                self.F.append(np.random.normal(mu, np.sqrt(sig[k]), (self.data_size, self.ks[k], self.n_filters_layer[k])))

            else:
                sig.append(2 / (self.nlen[k] * self.n_filters_layer[k - 1]))
                self.F.append(
                    np.random.normal(mu, np.sqrt(sig[k]), (self.n_filters_layer[k - 1], self.ks[k], self.n_filters_layer[k])))

        sig.append(2 / (self.nlen[-1] * self.n_filters_layer[-1]))
        self.W = np.random.normal(mu, np.sqrt(sig[-1]), (self.classes, self.n_filters_layer[-1] * self.nlen[-1]))

        if self.bias == True:
            self.bFs = []
            self.bW = np.random.normal(mu, np.sqrt(sig[-1]), (self.classes, 1))
            for k in range(self.nlayers):
                self.bFs.append(np.random.normal(mu, np.sqrt(sig[k]), (self.nlen[k + 1]*self.n_filters_layer[k], 1)))

        self.MXinputs_glob = []

        for k in range(len(X_input)):
            #self.MXinputs_glob.append(self.MXMatrix(X_input[k], nlen, ks[0], n_filters_layer[0]).copy())
            self.MXinputs_glob.append(self.MXMatrix_Efficient(X_input[k], self.data_size, self.ks[0]).copy())

        self.MXinputs = self.MXinputs_glob


    def MFMatrix(self, nlen, X, state):

        if state == 'Accuracy':
            self.xin = self.xin_global
        else:
            if isinstance(X, list):
                self.xin = X[0].reshape((-1, 1), order = 'F')
                for i in range(len(X) - 1):
                    self.xin = np.concatenate([self.xin, X[i + 1].reshape((-1, 1), order = 'F')], axis = 1)
            else:
                self.xin = X.reshape((-1, 1), order = 'F')

        self.VFlStored = []

        for k in range(self.nlayers):
            self.VFl = self.F[k].reshape((1, -1), order = 'F')
            self.VFl = self.VFl.reshape((self.n_filters_layer[k], -1))
            self.VFlStored.append(self.VFl)

        # Constructing the first matrix  for filtering

        self.MFMatrixStored = []
        if self.mode_stride == False:
            for l in range(self.nlayers):
                self.MFMatrixStored.append(np.zeros(((self.nlen[l] - self.F[l].shape[1] + 1) * self.n_filters_layer[l], self.nlen[l] * self.F[l].shape[0])))
                for i in range(self.nlen[l] - self.F[l].shape[1] + 1):
                    self.MFMatrixStored[l][i * self.n_filters_layer[l]: i * self.n_filters_layer[l] + self.n_filters_layer[l] , i * self.F[l].shape[0] : i * self.F[l].shape[0] + self.F[l].shape[0]*self.F[l].shape[1]] = self.VFlStored[l]
        else:
            for l in range(self.nlayers):
                self.MFMatrixStored.append(np.zeros((((self.nlen[l] - self.F[l].shape[1]) // self.strides[l] + 1) * self.n_filters_layer[l], self.nlen[l] * self.F[l].shape[0])))
                for i in range((self.nlen[l] - self.F[l].shape[1]) // self.strides[l] + 1):
                    self.MFMatrixStored[l][
                    i * self.n_filters_layer[l]: i * self.n_filters_layer[l] + self.n_filters_layer[l],
                    i * self.F[l].shape[0]: i * self.F[l].shape[0] + self.F[l].shape[0] * self.F[l].shape[1]] = \
                    self.VFlStored[l]


    def MXMatrix_Efficient(self, x_input, d, k, s = 1, P = 0, filters = 1):
        nlenaid = x_input.shape[1]
        if self.mode_stride == False:
            self.MXef = np.zeros((nlenaid - k + 1, k*d))
            for i in range(nlenaid - k + 1):
                self.MXef[i] = x_input[:, i: i + k].reshape((1, -1), order = 'F')
        else:
            self.MXef = np.zeros(((nlenaid - k)//s + 1 + 2*P, k*d))
            for i in range((nlenaid - k)//s + 1 + 2*P):
                self.MXef[i] = x_input[:, i : i + k].reshape((1, -1), order = 'F')

        return self.MXef

    def MXMatrix(self, x_input, d, k, nf):

        self.MX1 = np.array([])
        for i in range(d - k + 1):
            if self.MX1.shape[0] == 0:
                test = x_input[:, i:i + k]
                self.MX1 = np.kron(np.identity(nf), x_input[:, i:i + k].reshape((1, -1), order = 'F'))
            else:
                test = x_input[:, i:i + k]
                aid = np.kron(np.identity(nf), x_input[:, i:i + k].reshape((1, -1), order = 'F'))
                self.MX1 = np.concatenate([self.MX1, aid], axis = 0)

        return self.MX1

    def act_relu(self, input):
        indx = np.where(input < 0)
        output = input.copy()
        output[indx] = 0
        return output

    def EvaluateClassifier(self, X, W, b = 0):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0] * X.shape[1] * X.shape[2], X.shape[3])
        out = np.matmul(W, X) + b
        soft = np.exp(out) / (sum(np.exp(out)))
        return soft

    def ComputeCost(self, X, lab, state = None):
        self.forward(X, state)
        return sum(sum(lab.T * (-np.log(self.softmx)))) / self.xin.shape[1]

    def forward(self, X, state = None):

        self.MFMatrix(self.nlen[0], X, state)

        self.conv = []
        self.RelU_conv = []

        if self.bias == False:
            for k in range(self.nlayers):
                if k == 0:
                    self.conv.append(self.MFMatrixStored[k] @ self.xin)

                    if self.mode_stride == True and self.Pads[k + 1] != 0:
                        aid = []
                        for t in range(self.conv[k].shape[1]):
                            aid2 = self.conv[k][:, t].reshape((-1, self.n_filters_layer[k]), order = 'F')
                            aid2 = np.concatenate([np.zeros((self.Pads[k + 1], aid2.shape[1])), aid2], axis = 0)
                            aid2 = np.concatenate([aid2, np.zeros((self.Pads[k + 1], aid2.shape[1]))], axis = 0)
                            if len(aid) == 0:
                                aid = aid2.reshape((-1, 1), order = 'F')
                            else:
                                aid = np.concatenate([aid, aid2.reshape((-1, 1), order = 'F')], axis = 1)

                        self.conv[k] = aid

                else:
                    self.conv.append(self.MFMatrixStored[k] @ self.RelU_conv[k - 1])

                    if self.mode_stride == True and self.Pads[k + 1] != 0:
                        aid = []
                        for t in range(self.conv[k].shape[1]):
                            aid2 = self.conv[k][:, t].reshape((-1, self.n_filters_layer[k]), order = 'F')
                            aid2 = np.concatenate([np.zeros((self.Pads[k + 1], aid2.shape[1])), aid2], axis = 0)
                            aid2 = np.concatenate([aid2, np.zeros((self.Pads[k + 1], aid2.shape[1]))], axis = 0)
                            if len(aid) == 0:
                                aid = aid2.reshape((-1, 1), order = 'F')
                            else:
                                aid = np.concatenate([aid, aid2.reshape((-1, 1), order = 'F')], axis = 1)

                        self.conv[k] = aid

                self.RelU_conv.append(self.act_relu(self.conv[k]))

            self.softmx = self.EvaluateClassifier(self.RelU_conv[-1], self.W)

        else:
            for k in range(self.nlayers):
                if k == 0:
                    self.conv.append(self.MFMatrixStored[k] @ self.xin)
                    self.conv[k] = self.conv[k] + self.bFs[k]

                    if self.mode_stride == True and self.Pads[k + 1] != 0:
                        aid = []
                        for t in range(self.conv[k].shape[1]):
                            aid2 = self.conv[k][:, t].reshape((-1, self.n_filters_layer[k]), order = 'F')
                            aid2 = np.concatenate([np.zeros((self.Pads[k + 1], aid2.shape[1])), aid2], axis = 0)
                            aid2 = np.concatenate([aid2, np.zeros((self.Pads[k + 1], aid2.shape[1]))], axis = 0)
                            if len(aid) == 0:
                                aid = aid2.reshape((-1, 1), order = 'F')
                            else:
                                aid = np.concatenate([aid, aid2.reshape((-1, 1), order = 'F')], axis = 1)

                        self.conv[k] = aid

                else:
                        self.conv.append(self.MFMatrixStored[k] @ self.RelU_conv[k - 1])
                        self.conv[k] = self.conv[k] + self.bFs[k]

                        if self.mode_stride == True and self.Pads[k + 1] != 0:
                            aid = []
                            for t in range(self.conv[k].shape[1]):
                                aid2 = self.conv[k][:, t].reshape((-1, self.n_filters_layer[k]), order = 'F')
                                aid2 = np.concatenate([np.zeros((self.Pads[k + 1],aid2.shape[1])), aid2], axis = 0)
                                aid2 = np.concatenate([aid2, np.zeros((self.Pads[k + 1], aid2.shape[1]))], axis = 0)
                                if len(aid) == 0:
                                    aid = aid2.reshape((-1, 1), order = 'F')
                                else:
                                    aid = np.concatenate([aid, aid2.reshape((-1, 1), order = 'F')], axis = 1)

                            self.conv[k] = aid

                self.RelU_conv.append(self.act_relu(self.conv[k]))

            self.softmx = self.EvaluateClassifier(self.RelU_conv[-1], self.W, self.bW)

    def Accuracy(self, X, Y, state = None):
        self.forward(X, state)
        max_perc = np.argmax(self.softmx, axis = 0)
        max_lab = np.argmax(Y, axis = 1)
        error = 0
        for i in range(max_perc.shape[0]):
            if max_perc[i] != max_lab[i]:
                error = error + 1

        return 1 - error / len(X)

    def Confussion_Matrix(self, Y):
        self.Confussion = np.zeros((self.classes, self.classes))
        max_perc = np.argmax(self.softmx, axis = 0)
        max_lab = np.argmax(Y, axis = 1)
        for i in range(max_perc.shape[0]):
            self.Confussion[max_lab[i], max_perc[i]] += 1
        max = np.amax(self.Confussion)
        self.Confussion = self.Confussion / max



    def training(self, X, Y, Xval, Yval):

        self.Momentum = [0 for i in range(self.nlayers + 1)]

        if self.bias == True:
            self.bFMomentum = [0 for i in range(self.nlayers)]
            self.bWMomentum = 0

        accuracy_train = [self.Accuracy(X, Y, 'Accuracy')]
        accuracy_val = [self.Accuracy(Xval, Yval)]
        costs_train = [self.ComputeCost(X, Y, 'Accuracy')]
        costs_val = [self.ComputeCost(Xval, Yval)]

        print("\n First set of accuracies done \n")


        t = 0
        f = 0

        for i in range(self.GDParams['n_epoch']):


            indx = []
            for k in range(len(self.start)-1):
                diff = self.start[k + 1] - self.start[k]
                aid = np.random.permutation(diff)
                aid = aid + self.start[k]
                indx.extend(aid[:50])
            
                

            X_epoch = [X[v] for v in indx]
            Y_epoch = Y[indx]



            '''
            indx = [i for i in range(len(X))]
            X_epoch = X
            Y_epoch = Y
            
            X_epoch = X
            Y_epoch = Y
            '''

            self.MXinputs = [self.MXinputs_glob[l] for l in indx]
            #self.MXinputs = self.MXinputs_glob

            for j in range(len(X_epoch) // self.GDParams['n_batch']):
                j_start = j * self.GDParams['n_batch']
                j_end = (j + 1) * self.GDParams['n_batch']
                Xbatch = X_epoch[j_start:j_end]
                Ybatch = Y_epoch[j_start:j_end]

                self.forward(Xbatch)
                self.back_prop(Ybatch, j_start)

                for m in range(len(self.Momentum)):
                    self.Momentum[m] = self.Momentum[m] * self.GDParams['rho'] + self.deriv[m] * self.GDParams['eta']

                for n in range(self.nlayers):
                    aid = self.Momentum[n].reshape((1, -1))
                    self.F[n] = self.F[n] - aid.reshape((self.F[n].shape), order = 'F')

                self.W = self.W - self.Momentum[-1]

                if self.bias == True:

                    for m in range(self.nlayers):
                        self.bFMomentum[m] = self.bFMomentum[m] * self.GDParams['rho'] + self.GradbF[m] * self.GDParams['eta']
                        self.bFs[m] = self.bFs[m] - self.bFMomentum[m]

                    self.bWMomentum = self.bWMomentum * self.GDParams['rho'] + self.GradbW * self.GDParams['eta']
                    self.bW = self.bW - self.bWMomentum


                t = t + 1


                if t % 500 == 0:
                    accuracy_train.append(self.Accuracy(X, Y, 'Accuracy'))
                    costs_train.append(self.ComputeCost(X, Y, 'Accuracy'))
                    accuracy_val.append(self.Accuracy(Xval, Yval))
                    costs_val.append(self.ComputeCost(Xval, Yval))
                    print("\n Accuracy test done n: ", f)
                    f = f + 1
                    t = 0

        accuracy_train.append(self.Accuracy(X, Y, 'Accuracy'))
        costs_train.append(self.ComputeCost(X, Y, 'Accuracy'))
        accuracy_val.append(self.Accuracy(Xval, Yval))
        costs_val.append(self.ComputeCost(Xval, Yval))
        self.Confussion_Matrix(Yval)


        return accuracy_train, accuracy_val, costs_train, costs_val, self.Confussion

    def back_prop(self, Y, alfa = 0):

        Gbatch = -(Y.T - self.softmx)

        if self.bias == False:

            self.dJw = (Gbatch @ self.RelU_conv[-1].T) / self.xin.shape[1]
            self.dLFstore = []
            self.dLF = []

            for k in range(self.nlayers):

                if k == 0:
                    Gbatch = self.W.T @ Gbatch
                else:
                    Gbatch = self.MFMatrixStored[-k].T @ Gbatch

                Gbatch = Gbatch * (self.RelU_conv[-1 - k] > 0)

                for i in range(Gbatch.shape[1]):
                    gj = Gbatch[:, i].reshape(-1, 1)

                    if k != self.nlayers - 1:
                        gj = gj.reshape(-1, self.F[-1 - k].shape[2],)

                        if self.mode_stride == False:
                            MxLayer = self.MXMatrix_Efficient( self.RelU_conv[-2 - k][:, i].reshape((self.n_filters_layer[-2 - k], -1), order = 'F'),
                                self.n_filters_layer[-k - 2], self.F[-1 - k].shape[1])
                        else:
                            MxLayer = self.MXMatrix_Efficient(
                                self.RelU_conv[-2 - k][:, i].reshape((self.n_filters_layer[-2 - k], -1), order = 'F'),
                                self.n_filters_layer[-k - 2], self.F[-1 - k].shape[1], self.strides[-2 - k])


                    else:
                        MxLayer = self.MXinputs[alfa + i]
                        gj = gj.reshape(-1, self.F[-1 - k].shape[2],)



                    if len(self.dLF) == 0:
                        self.dLF = (MxLayer.T @ gj).reshape((1, -1), order = 'F')

                    else:
                        self.dLF = self.dLF + (MxLayer.T @ gj).reshape((1, -1), order = 'F')

                self.dLF = self.dLF / Gbatch.shape[1]
                self.dLFstore.insert(0, self.dLF)
                self.dLF = []

            self.deriv = self.dLFstore
            self.deriv.append(self.dJw)

        else:

            self.GradbF = []

            self.GradbW = Gbatch @ np.ones((Gbatch.shape[1], 1)) / Gbatch.shape[1]
            self.dJw = (Gbatch @ self.RelU_conv[-1].T) / self.xin.shape[1]
            self.dLFstore = []
            self.dLF = []

            for k in range(self.nlayers):

                if k == 0:
                    Gbatch = self.W.T @ Gbatch
                else:
                    Gbatch = self.MFMatrixStored[-k].T @ Gbatch

                Gbatch = Gbatch * (self.RelU_conv[-1 - k] > 0)


                if self.mode_stride == False or self.Pads[-k - 1] == 0:
                    self.GradbF.insert(0, Gbatch@ np.ones((Gbatch.shape[1], 1)) / Gbatch.shape[1] )

                else:
                    Gaid = []

                    for t in range(Gbatch.shape[1]):
                        temp = self.n_filters_layer[-k - 1]
                        aid2 = Gbatch[:, t].reshape((-1, self.n_filters_layer[-k - 1]), order = 'F')
                        aid2 = aid2[self.Pads[-k - 1] : -self.Pads[-k - 1]]
                        if len(Gaid) == 0:
                            Gaid = aid2.reshape((-1, 1), order = 'F')
                        else:
                            Gaid = np.concatenate([Gaid, aid2.reshape((-1, 1), order = 'F')], axis = 1)

                    self.GradbF.insert(0, Gaid @ np.ones((Gbatch.shape[1], 1)) / Gbatch.shape[1])

                for i in range(Gbatch.shape[1]):
                    gj = Gbatch[:, i].reshape(-1, 1)

                    if k != self.nlayers - 1:
                        gj = gj.reshape(-1, self.F[-1 - k].shape[2], )

                        if self.mode_stride == False:
                            MxLayer = self.MXMatrix_Efficient(
                                self.RelU_conv[-2 - k][:, i].reshape((self.n_filters_layer[-2 - k], -1), order = 'F'),
                                self.n_filters_layer[-k - 2], self.F[-1 - k].shape[1])
                        else:
                            MxLayer = self.MXMatrix_Efficient(
                                self.RelU_conv[-2 - k][:, i].reshape((self.n_filters_layer[-2 - k], -1), order = 'F'),
                                self.n_filters_layer[-k - 2], self.F[-1 - k].shape[1], self.strides[-1 - k])

                    else:
                        MxLayer = self.MXinputs[alfa + i]
                        gj = gj.reshape(-1, self.F[-1 - k].shape[2], )

                    if len(self.dLF) == 0:
                        self.dLF = (MxLayer.T @ gj).reshape((1, -1), order = 'F')

                    else:
                        self.dLF = self.dLF + (MxLayer.T @ gj).reshape((1, -1), order = 'F')

                self.dLF = self.dLF / Gbatch.shape[1]
                self.dLFstore.insert(0, self.dLF)
                self.dLF = []

            self.deriv = self.dLFstore
            self.deriv.append(self.dJw)

    def Rel_Error(self, grad_num, grad_anal, eps):
        err = np.abs(grad_num - grad_anal) / np.maximum(eps * np.ones(grad_num.shape),
                                                        np.abs(grad_num) + np.abs(grad_anal))
        return err

    def CompareGradients(self, X, Y, e, h):
        acc = [1e-6, 1e-7, 1e-8, 1e-9]
        per_F1 = []
        per_F2 = []
        per_F3 = []
        per_W = []

        self.forward(X)
        self.back_prop(Y)
        dLF_anal = []
        for i in range(self.nlayers):
            dLF_anal.append(self.dLFstore[i].copy())
        dJw_anal = self.dJw.copy()

        GradF, GradW = self.NumericalGradient(X, Y, h)

        for err in acc:
            err_F1 = self.Rel_Error(dLF_anal[0], GradF[0].reshape((dLF_anal[0].shape), order = 'F'), e)
            err_F2 = self.Rel_Error(dLF_anal[1], GradF[1].reshape((dLF_anal[1].shape), order = 'F'), e)
            #err_F3 = self.Rel_Error(dLF_anal[2], GradF[2].reshape((dLF_anal[2].shape), order = 'F'), e)
            err_W = self.Rel_Error(dJw_anal, GradW, e)
            per_F1.append(np.sum(err_F1 <= err) / (self.F[0].shape[0] * self.F[0].shape[1] * self.F[0].shape[2]))
            per_F2.append(np.sum(err_F2 <= err) / (self.F[1].shape[0] * self.F[1].shape[1] * self.F[1].shape[2]))
            #per_F3.append(np.sum(err_F3 <= err) / (self.F[2].shape[0] * self.F[2].shape[1] * self.F[2].shape[2]))
            per_W.append(np.sum(err_W <= err) / (self.W.shape[0] * self.W.shape[1]))

        return per_F1, per_F2, per_W #per_F3, per_W

    def NumericalGradient(self, X_inputs, Ys, h):

        Fs = [self.F[k].copy() for k in range(len(self.F))]
        GradF = [np.zeros(self.F[k].shape) for k in range(len(self.F))]

        for k in range(len(Fs)):
            for i in range(Fs[k].shape[0]):
                for j in range(Fs[k].shape[1]):
                    for l in range(Fs[k].shape[2]):
                        F_try = Fs[k].copy()
                        F_try[i, j, l] = F_try[i, j, l] - h

                        self.F[k] = F_try

                        l1 = self.ComputeCost(X_inputs, Ys)
                        F_try[i, j, l] = F_try[i, j, l] + 2 * h

                        self.F[k] = F_try

                        l2 = self.ComputeCost(X_inputs, Ys)

                        self.F[k] = Fs[k].copy()

                        GradF[k][i, j, l] = (l2 - l1) / (2 * h)

        # compute the gradient for the fully connected layer
        W_try = self.W.copy()
        GradW = np.zeros(W_try.shape)
        for i in range(W_try.shape[0]):
            for j in range(W_try.shape[1]):
                W_try1 = W_try.copy()
                W_try1[i, j] = W_try[i, j] - h
                self.W = W_try1

                l1 = self.ComputeCost(X_inputs, Ys)

                self.W[i, j] = self.W[i, j] + 2 * h

                l2 = self.ComputeCost(X_inputs, Ys)

                GradW[i, j] = (l2 - l1) / (2 * h)

                self.W = W_try.copy()

        return GradF, GradW


def best_hyper(X, Y, Xval, Yval, network):
    n_layers = [3, 4, 5]
    ks = [4]
    nlayers_up = [20]
    b_acc = 0
    b_nlayer = 0
    b_k = 0
    b_nfilter = 0
    flag = 0

    for n in n_layers:
        for k in ks:
            for c in nlayers_up:
                print('\n Now testing: ')
                print(n, k, c)
                k_to_send = [k for i in range(n)]
                nlayers_to_send = [c for i in range(n)]
                if flag != 0:
                    network.update_structure(n, k_to_send, nlayers_to_send, X)
                else:
                    flag = 1
                acc_t, acc_val_t, cost_train_t, cost_val_t, matrix_t = network.training(X, Y, Xval, Yval)
                n_acc = acc_val_t[-1]

                if n_acc > b_acc:
                    b_acc = n_acc
                    b_nlayer = n
                    b_k = k
                    b_nfilter = c
                    b_net = copy.deepcopy(network)
                    to_draw = [acc_t.copy(), acc_val_t.copy(), cost_train_t.copy(), cost_val_t.copy()]
                    b_mat = matrix_t.copy()


    print('The best network has ' + str(b_nlayer) + ' layers, with ' + str(b_nfilter) + ' filters per layer, a depth of ' + str(b_k) + 'and we achieve an accuracy of ' + str(b_acc))
    return b_acc, b_nlayer, b_k, b_nfilter, b_net, b_mat, to_draw


def class_lengths(Y):
    start = [0]
    prev = 0
    curr = 0

    for i in range(len(Y)):
        curr = Y[i]
        if curr != prev:
            prev = curr
            start.append(i)
    start.append(i)
    return start


if __name__ == "__main__":
    FileTrain1 = "C:\\Users\\sebas\\Documents\\Universidad\\Máster\\KTH\\Deep Learning in Data Science\\Assignment 3\\ascii_names.txt"
    Val_Inds = "C:\\Users\\sebas\\Documents\\Universidad\\Máster\\KTH\\Deep Learning in Data Science\\Assignment 3\\Validation_Inds.txt"
    Cat_lab = "C:\\Users\\sebas\\Documents\\Universidad\\Máster\\KTH\\Deep Learning in Data Science\\Assignment 3\\category_labels.txt"
    '''
    test = [i for i in range(32)]
    ntest = np.array(test).reshape((4, 4, 2), order = 'F')
    test2 = ntest.reshape((1, -1), order ='F')
    test3 = test2.reshape((2, -1))
    f1 = test3.reshape((1, -1))
    f2 = f1.reshape(ntest.shape, order = 'F')
    '''

    all_categories = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French',
                      'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean',
                      'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']

    train_X, train_lab = unpickle(FileTrain1)
    indx = unpickle2(Val_Inds)
    n_len = max_length(train_X)
    ohd_names = oneHotEncoderNames(train_X, 29, n_len)
    Y_ohd = oneHotEncoder(train_lab)

    val_name = []
    Y_val = []

    for i in indx:
        val_name.append(ohd_names[i])
        Y_val.append(Y_ohd[i, :].reshape(1, -1))

    for i in range(len(indx)):
        ohd_names.pop(indx[i] - i)
        train_lab.pop(indx[i] - i)

    Y_ohd = np.delete(Y_ohd, indx, axis = 0)

    Y_val = np.array(Y_val).reshape(-1, 18)


    # n_len = 5
    h = 1e-5
    eps = h

    print(h, eps)

    test = ['barbas', 'repetto', 'cappa', 'le bardou', 'al janabbi']
    test_lab = [16, 9, 9, 5, 0]

    ohd_test = oneHotEncoderNames(test, 29, n_len)
    Y_test = np.zeros((len(test), 18))
    for i in range(len(test_lab)):
        Y_test[i, test_lab[i]] = 1


    start = class_lengths(train_lab)

    #GDParams['n_epoch'] = 120
    # (self, n_filters_layer_1, k1, n_filters_layer_2, k2, d, K, nlen, GDParams)
    # def __init__(self, nlayers, n_filters_layer, ks, d, K, nlen, GDParams, X_input, start):
    #net = ConvNetwork(4, [20, 20, 20, 20], [4, 4, 4, 4], 29, 18, n_len, GDParams, ohd_names, start, True)
    #net = ConvNetwork(3, [5, 4, 3], [5, 3, 3], 29, 18, n_len, GDParams, ohd_names[:2], start)
    # net = ConvNetwork(2, 2, 2, 3, 4, 4, 4, GDParams)
    # X = np.arange(16).reshape(4, -1)
    # t1 = net.MFMatrix(n_len, ohd_names[0])
    # t2 = net.MXMatrix(ohd_names[0], n_len, 2, 2)
    # test = t1 - t2
    # = ConvNetwork(2, [20, 20], [10, 3], 29, 18, n_len, GDParams, ohd_names, start, False)
    net = ConvNetwork(3, [7, 7, 7], [5, 2, 2], 29, 18, n_len, GDParams, ohd_names, start, False)
    print("\n About to train \n")

    '''
    b_acc, b_nlayer, b_k, b_nfilter, b_net, b_mat, to_draw = best_hyper(ohd_names, Y_ohd, val_name, Y_val, net)

    acc = to_draw[0]
    acc_val = to_draw[1]
    cost_train = to_draw[2]
    cost_val = to_draw[3]

    print('\n Confussion matrix \n')
    print(b_mat)
    '''


    #net.forward(ohd_names[:2])
    #net.back_prop(Y_ohd[:2])








    acc, acc_val, cost_train, cost_val, matrix = net.training(ohd_names, Y_ohd, val_name, Y_val)

    #acc, acc_val, cost_train, cost_val, matrix = net.training(ohd_names, Y_ohd, val_name, Y_val)

    with open('trained__basic.pkl', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
    net.forward(ohd_test)
    acc_test = net.Accuracy(ohd_test, Y_test)
    print("\n Probability vector: \n")
    print(net.softmx)
    net.Confussion_Matrix(Y_test)
    test_mtx = net.Confussion

    Plot_Cost_Loss(cost_train, cost_val, acc, acc_val)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation = 90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

    print("Acabo todo")
    a = input()


    print("\n General matrix \n")
    print(matrix)
    print("\n Test matrix \n")
    print(test_mtx)
    print("\n Accuracy in the test sample: ", acc_test)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    cax2 = ax2.matshow(test_mtx)
    fig2.colorbar(cax2)

    # Set up axes
    ax2.set_xticklabels([''] + all_categories, rotation = 90)
    ax2.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()



    #per_F1, per_F2, per_F3, per_W = net.CompareGradients(ohd_names[:2], Y_ohd[:2], eps, h)
    #PlotAccuracyGradients(per_F1, per_F2, per_F3, per_W, net.n_filters_layer[0], net.n_filters_layer[1])

    '''
    with open('trained_net.pkl', 'rb') as input:
        net2 = pickle.load(input)
    '''
    print("Acabo todo")
    a = input()