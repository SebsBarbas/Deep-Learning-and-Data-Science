#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
import copy
import pickle
import json

GDParams={'n_batch' : 100, 'n_epoch' : 12, 'eta' : 0.01, 'eps' : 1e-8}

np.random.seed(400)
random.seed(400)

def unpickle(file):

    char_2_int = {}
    int_2_char = {}
    with open(file) as f:
        content = f.read()
    all_chars = list(content)
    unique_chars = np.unique(all_chars)
    for indx in range(len(unique_chars)):
        char_2_int[unique_chars[indx]] = indx
        int_2_char[str(indx)] = unique_chars[indx]

    return unique_chars, char_2_int, int_2_char, all_chars

def read_tweets(file):
    f = open(file)
    data = json.load(f)
    temp = []
    for i in data:
        if i['is_retweet'] == False:
            temp.append(i['text'])

    return temp

def maps(tweets):
    temp = []
    for i in tweets:
        temp.extend(list(i))

    char_2_int = {}
    int_2_char = {}
    unique_chars = np.unique(temp)
    for indx in range(len(unique_chars)):
        char_2_int[unique_chars[indx]] = indx
        int_2_char[str(indx)] = unique_chars[indx]

    return char_2_int, int_2_char, unique_chars


def PlotSmooth(smooth):
    plt.plot(np.arange(len(smooth)), smooth)
    plt.xlabel('Iteration')
    plt.ylabel('Smooth Loss')
    plt.title('Smooth Loss over iterations')
    plt.show()

def PlotAccuracyGradients(acc_U, acc_W, acc_V, acc_b, acc_c):
    plt.plot([1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_U), [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_W),\
             [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_V), [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_b), [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_c))
    plt.legend(['h0', 'U', 'W', 'V', 'b', 'c'])
    plt.xlabel('Threshold level')
    plt.ylabel('Accuracy of the gradient')
    plt.title('Accuracy in the derivative calculations')
    plt.show()

class RNN:

    def __init__(self, K, seq_length, GDParams, dict_c_2_i, dict_i_2_c, m):

        self.hidden_state = m
        self.GDParams = GDParams
        self.seq_length = seq_length
        self.K = K
        self.map_char_2_int = dict_c_2_i
        self.map_int_2_char = dict_i_2_c

        mu = 0
        self.h0 = np.random.normal(mu, 0.01, (self.hidden_state, 1))
        self.W = np.random.normal(mu, 0.01, (self.hidden_state, self.hidden_state))
        self.U = np.random.normal(mu, 0.01, (self.hidden_state, self.K))
        self.b = np.random.normal(mu, 0.01, (self.hidden_state, 1))
        self.V = np.random.normal(mu, 0.01, (self.K, self.hidden_state))
        self.c = np.random.normal(mu, 0.01, (self.K, 1))

        self.Grad_W = []
        self.Grad_U = []
        self.Grad_b = []
        self.Grad_V = []
        self.Grad_c = []

        self.ht = []
        self.all_soft = []

        self.lab = []
        self.xin = []
        self.written_text = []

    def ohd_letters(self, character):
        ohd = np.zeros((self.K, 1))
        indx = self.map_char_2_int[character]
        ohd[indx] = 1
        return ohd

    def Compute_X_and_Y(self, X_input):

        self.xin = []
        self.lab = []

        for i in range(self.seq_length - 1):
            if i == 0:
                self.lab = self.ohd_letters(X_input[i + 1])
            else:
                self.lab = np.concatenate([self.lab, self.ohd_letters(X_input[i + 1])], axis = 1)

        self.xin = self.ohd_letters(X_input[0])
        self.xin = np.concatenate([self.xin, self.lab[:, :-1]], axis = 1)

    def probability_letter(self):
        cumulative = np.cumsum(self.all_soft[:, -1].reshape(1, -1), dtype = float)
        random_prob = random.uniform(0, 1)
        test = np.where(cumulative > random_prob)
        indx = np.where(cumulative > random_prob)[0][0]
        return self.map_int_2_char[str(indx)]



    def forward(self, X_input, mode = 'train'):
        if mode == 'train':
            self.Compute_X_and_Y(X_input)
        else:
            self.xin = self.ohd_letters(X_input)
        self.ht = []

        for i in range(self.seq_length - 1):

            if i == 0:
                self.a = self.W @ self.h0 + self.U @ self.xin[:, i].reshape(-1, 1) + self.b
                self.ht = np.tanh(self.a[:, i].reshape(-1, 1))
            else:
                self.a = np.concatenate([self.a, self.W @ self.ht[:, i - 1].reshape(-1, 1) + self.U @ self.xin[:, i].reshape(-1, 1) + self.b], axis = 1)
                self.ht = np.concatenate([self.ht, np.tanh(self.a[:, -1].reshape(-1, 1))], axis = 1)

            if i == 0:
                self.all_soft = self.EvaluateClassifier(self.ht[:, -1].reshape(-1, 1), self.V, self.c)
            else:
                self.all_soft = np.concatenate([self.all_soft, self.EvaluateClassifier(self.ht[:, -1].reshape(-1, 1), self.V, self.c)], axis = 1)

            if mode == 'test':
                if i == 0:
                    self.to_synth = [X_input]
                    self.to_synth.append(self.probability_letter())
                    self.h0 = self.ht[:, -1].reshape(-1, 1)
                    self.xin = np.concatenate([self.xin, self.ohd_letters(self.to_synth[-1])], axis = 1)
                else:
                    self.to_synth.append(self.probability_letter())
                    self.h0 = self.ht[:, -1].reshape(-1, 1)
                    self.xin = np.concatenate([self.xin, self.ohd_letters(self.to_synth[-1])], axis = 1)


    def EvaluateClassifier(self, X, V, c = 0):
        out = np.matmul(V, X) + c
        soft = np.exp(out) / (sum(np.exp(out)))
        return soft

    def ComputeCost(self):
        #test = self.lab.T @ self.all_soft
        #test2 = self.lab * self.all_soft
        return sum(sum(self.lab * (-np.log(self.all_soft))))

    def ComputeLoss(self, X):
        self.forward(X)
        return self.ComputeCost()

    def back_prop(self):
        Gbatch = -(self.lab - self.all_soft)
        self.Grad_c = np.sum(Gbatch, axis = 1).reshape(-1, 1)
        self.Grad_V = Gbatch @ self.ht.T

        dLh = np.zeros([self.h0.shape[0], self.lab.shape[1]])
        dLa = np.zeros([self.h0.shape[0], self.lab.shape[1]])

        t = np.diag(1 - np.tanh(self.a[:, -1])**2)
        dLh[:, -1] = Gbatch[:, -1].reshape(1, -1) @ self.V
        dLa[:, -1] = dLh[:, -1].reshape(1, -1) @ np.diag(1 - np.tanh(self.a[:, -1])**2)
        for i in range(dLh.shape[1] - 1):
            t = dLa[:, dLh.shape[1] -1 -i]
            t2 = Gbatch[:, dLh.shape[1] - 2 - i]
            dLh[:, dLh.shape[1] -2 -i] = Gbatch[:, dLh.shape[1] -2 -i].reshape(1, -1) @ self.V + dLa[:, dLh.shape[1] -1 -i].reshape(1, -1) @ self.W
            dLa[:, dLh.shape[1] -2 -i] = dLh[:, dLh.shape[1] -2 -i].reshape(1, -1) @ np.diag(1 - np.tanh(self.a[:, dLh.shape[1] -2 -i])**2)

        stack_h = np.concatenate([self.h0, self.ht[:, :dLa.shape[1] - 1]], axis = 1)
        self.Grad_W = dLa @ stack_h.T
        self.Grad_U = dLa @ self.xin.T
        self.Grad_b = np.sum(dLa, axis = 1).reshape(-1, 1)

        #Clipping process
        indx = np.where(self.Grad_W < -5)
        self.Grad_W[indx] = -5
        indx = np.where(self.Grad_W > 5)
        self.Grad_W[indx] = 5

        indx = np.where(self.Grad_U < -5)
        self.Grad_U[indx] = -5
        indx = np.where(self.Grad_U > 5)
        self.Grad_U[indx] = 5

        indx = np.where(self.Grad_b < -5)
        self.Grad_b[indx] = -5
        indx = np.where(self.Grad_b > 5)
        self.Grad_b[indx] = 5

        indx = np.where(self.Grad_c < -5)
        self.Grad_c[indx] = -5
        indx = np.where(self.Grad_c > 5)
        self.Grad_c[indx] = 5

        indx = np.where(self.Grad_V < -5)
        self.Grad_V[indx] = -5
        indx = np.where(self.Grad_V > 5)
        self.Grad_V[indx] = 5


        return self.Grad_W, self.Grad_U, self.Grad_b, self.Grad_c, self.Grad_V

    def Rel_Error(self, grad_num, grad_anal, eps):
        err = np.abs(grad_num - grad_anal) / np.maximum(eps * np.ones(grad_num.shape),
                                                        np.abs(grad_num) + np.abs(grad_anal))
        return err

    def Compare_Gradients(self, X, h, e):
        accuracies = [1e-6, 1e-7, 1e-8, 1e-9]
        per_W = []
        per_U = []
        per_V = []
        per_b = []
        per_c = []

        self.forward(X)
        dLw_anal, dLu_anal, dLb_anal, dLc_anal, dLv_anal = self.back_prop()
        dLw_num, dLu_num, dLb_num, dLc_num, dLv_num = self.ComputeGradNum(X, h)

        for err in accuracies:
            err_W = self.Rel_Error(dLw_anal, dLw_num, e)
            err_U = self.Rel_Error(dLu_anal, dLu_num, e)
            err_V = self.Rel_Error(dLv_anal, dLv_num, e)
            err_b = self.Rel_Error(dLb_anal, dLb_num, e)
            err_c = self.Rel_Error(dLc_anal, dLc_num, e)
            per_U.append(np.sum(err_U <= err) / (self.U.shape[0] * self.U.shape[1]))
            per_W.append(np.sum(err_W <= err) / (self.W.shape[0] * self.W.shape[1]))
            per_V.append(np.sum(err_V <= err) / (self.V.shape[0] * self.V.shape[1]))
            per_b.append(np.sum(err_b <= err) / (self.b.shape[0] * self.b.shape[1]))
            per_c.append(np.sum(err_c <= err) / (self.c.shape[0] * self.c.shape[1]))


        return per_U, per_W, per_V, per_b, per_c

    def synthesize(self, length):
        self.h0 = np.zeros(self.h0.shape)
        for i in range(length // self.seq_length):
            if i == 0:
                indx = random.randint(0, self.K - 1)
                self.forward(self.map_int_2_char[str(indx)], mode = 'test')
            else:
                self.forward(self.written_text[-1], mode = 'test')

            self.written_text.extend(self.to_synth)



    def training(self, tweets):

        epochs = 7
        self.moments =  [np.zeros(self.U.shape), np.zeros(self.W.shape),\
                         np.zeros(self.V.shape), np.zeros(self.b.shape),\
                         np.zeros(self.c.shape)]
        self.smooth_loss = 0
        t = 0
        count = 0
        for e in range(epochs):
            print("epoch ", e)
            i = 0
            self.h0 = np.zeros(self.h0.shape)
            for tweet in tweets:
                i = 0
                self.h0 = np.zeros(self.h0.shape)
                self.seq_length = 20
                while i < len(tweet):
                    if i + self.seq_length > len(tweet) - 1:
                        self.seq_length = len(tweet) - i
                    else:
                        if len(tweet) - i - self.seq_length == 1:
                            self.seq_length = self.seq_length + 1
                        else:
                            self.seq_length = 20

                    self.forward(tweet[i : i + self.seq_length])
                    self.back_prop()
                    self.h0 = self.ht[:, -1].reshape(-1, 1)

                    #Calculate the AdaGrad
                    self.moments[0] = self.moments[0] + self.Grad_U**2
                    self.moments[1] = self.moments[1] + self.Grad_W**2
                    self.moments[2] = self.moments[2] + self.Grad_V**2
                    self.moments[3] = self.moments[3] + self.Grad_b**2
                    self.moments[4] = self.moments[4] + self.Grad_c**2

                    self.U = self.U - self.GDParams['eta'] * self.Grad_U / (np.sqrt(self.moments[0] + self.GDParams['eps']))
                    self.W = self.W - self.GDParams['eta'] * self.Grad_W / (np.sqrt(self.moments[1] + self.GDParams['eps']))
                    self.V = self.V - self.GDParams['eta'] * self.Grad_V / (np.sqrt(self.moments[2] + self.GDParams['eps']))
                    self.b = self.b - self.GDParams['eta'] * self.Grad_b / (np.sqrt(self.moments[3] + self.GDParams['eps']))
                    self.c = self.c - self.GDParams['eta'] * self.Grad_c / (np.sqrt(self.moments[4] + self.GDParams['eps']))

                    if t == 0:
                        val = self.ComputeCost()
                        if val > 100:
                            val = 100
                        self.smooth_loss = [val]

                    else:
                        self.smooth_loss.append(0.999 * self.smooth_loss[-1] + 0.001 * self.ComputeCost())
                    i = i + self.seq_length
                    t = t + 1
                    if t % 1000 == 0:
                        self.seq_length = 20
                        self.synthesize(140)
                        print("\n Iter: ", t)
                        print('\n Smooth loss: ', self.smooth_loss[-1])
                        self.written_text = "".join(self.written_text)
                        ending = self.written_text.find('·')
                        if ending >= 0:
                         print("".join(self.written_text[:ending]))
                        else:
                            print("".join(self.written_text))

                        self.written_text = []
                        self.to_synth = []
                        self.h0 = np.zeros(self.h0.shape)

        return self.smooth_loss






    def ComputeGradNum(self, X, h):

        gradU = np.zeros(self.U.shape)
        gradV = np.zeros(self.V.shape)
        gradW = np.zeros(self.W.shape)
        gradc = np.zeros(self.c.shape)
        gradb = np.zeros(self.b.shape)

        U_try = self.U.copy()
        #ComputeLoss(self, Y, W, h0, U, V, c, b)
        for i in range(self.U.shape[0]):
            for j in range(self.U.shape[1]):
                U_try = self.U.copy()
                self.U[i][j] = U_try[i][j] - h
                l1 = self.ComputeLoss(X)
                self.U[i][j] = U_try[i][j] + h
                l2 = self.ComputeLoss(X)
                self.U = U_try
                gradU[i][j] = (l2 - l1) / (2 * h)

        W_try = self.W.copy()
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                W_try = self.W.copy()
                self.W[i][j] = W_try[i][j] - h
                l1 = self.ComputeLoss(X)
                self.W[i][j] = W_try[i][j] + h
                l2 = self.ComputeLoss(X)
                self.W = W_try
                gradW[i][j] = (l2 - l1) / (2 * h)

        b_try = self.b.copy()
        for i in range(self.b.shape[0]):
            for j in range(self.b.shape[1]):
                b_try = self.b.copy()
                self.b[i][j] = b_try[i][j] - h
                l1 = self.ComputeLoss(X)
                self.b[i][j] = b_try[i][j] + h
                l2 = self.ComputeLoss(X)
                self.b = b_try
                gradb[i][j] = (l2 - l1) / (2 * h)

        c_try = self.c.copy()
        for i in range(self.c.shape[0]):
            for j in range(self.c.shape[1]):
                c_try = self.c.copy()
                self.c[i][j] = c_try[i][j] - h
                l1 = self.ComputeLoss(X)
                self.c[i][j] = c_try[i][j] + h
                l2 = self.ComputeLoss(X)
                self.c = c_try
                gradc[i][j] = (l2 - l1) / (2 * h)


        V_try = self.V.copy()
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                V_try = self.V.copy()
                self.V[i][j] = V_try[i][j] - h
                l1 = self.ComputeLoss(X)
                self.V[i][j] = V_try[i][j] + h
                l2 = self.ComputeLoss(X)
                self.V = V_try
                gradV[i][j] = (l2 - l1) / (2 * h)

        return gradW, gradU, gradb, gradc, gradV

def best_net(text, u_chars, GDParams, m_c_2_i, m_i_2_c):

    length_of_seq = [25, 40]
    hidden_nodes = [50, 200, 500]
    eta = [0.01, 0.1]
    b_smooth = 0
    b_len = 0
    b_nod = 0
    b_eta = 0
    for leng in length_of_seq:
        for nod in hidden_nodes:
            for e in eta:
                GDParams['eta'] = e
                net = RNN(len(u_chars), leng, GDParams, m_c_2_i, m_i_2_c, nod)
                smooth = net.training(text)
                av_smooth = sum(smooth[-10:]) / len(smooth[-10:])
                av_smooth = av_smooth / leng
                if av_smooth > b_smooth:
                    b_len = leng
                    b_nod = nod
                    b_eta = e

    return b_len, b_nod, b_eta



if __name__ == "__main__":
    file_2017 = ".\condensed_2017.json\condensed_2017.json"
    file_2018 = ".\condensed_2018.json\condensed_2018.json"
    files = [file_2017, file_2018]
    train_tweets = []
    for i in files:
        tweets = read_tweets(i)
        train_tweets.extend(tweets)

    map_char_2_int, map_int_2_char, unique = maps(train_tweets)
    j = np.where(unique == '·')
    for t in range(len(train_tweets)):
        train_tweets[t] = train_tweets[t] + '·'

    map_char_2_int['·'] = len(map_char_2_int)
    map_int_2_char[str(len(map_int_2_char))] = '·'

    #unique_chars, map_char_2_int, map_int_2_chars, text = unpickle(textfile)
    h = 1e-5
    e = h
    leng = 20
    #text = 'Clara no quiere ser-·+hijueputa maricaaaaaa loco poco'

    tweet = train_tweets[2472]
    i = 0
    while i < len(tweet):
        if i + leng > len(tweet) - 1:
            leng = len(tweet) - i
        else:
            if len(tweet) - i - leng == 1:
                leng = leng + 1
            else:
                leng = 20

        print(tweet[i : i + leng])
        i = i + leng






    #text = ['h', 'e', 'l', 'l', 'o', ' ', 'S', 'e', 'b']
    #length = 9
    #i = 0
    #while i <= len(text) - length:
    #    print(text[i : i + length])
    #    i = i + length

    #alf = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #al_mean = sum(alf[-2:]) / len(alf[-2:])

    #b_len, b_nod, b_eta = best_net(text, unique_chars, GDParams, map_char_2_int, map_int_2_chars)
    #print("Best text reading: " + str(b_len) + " best number of hidden nodes: " + str(b_nod) + " best eta: " + str(b_eta))

    #self, K, seq_length, GDParams, dict_c_2_i, dict_i_2_c, m
    net = RNN(len(unique) + 1, 20, GDParams, map_char_2_int, map_int_2_char, 300)
    #net.forward(['h', 'e', 'l', 'l', 'o'])
    #net.back_prop()
    smooth_loss = net.training(train_tweets)
    #net.synthesize(1000)
    print("".join(net.written_text))
    with open('trained__best_net_Trump.pkl', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
    PlotSmooth(smooth_loss)
    #per_U, per_W, per_V, per_b, per_c = net.Compare_Gradients(['h', 'e', 'l', 'l', 'o'], h, e)
    #PlotAccuracyGradients(per_U, per_W, per_V, per_b, per_c)

    print("Acabo todo")
    a = input()
    a = 1