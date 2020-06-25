#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

GDParams={'n_batch' : 20, 'n_epoch' : 70, 'eta' : 0.1}

np.random.seed(300)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def oneHotEncoder(data):
    label_encoder = LabelEncoder()
    values = np.array(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def EvaluateClassifier(X, W, b):
    if len(X.shape)>2:
        X=X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3])
    out = np.matmul(W,X.T)+b
    soft = np.exp(out)/(sum(np.exp(out)))
    return soft


def ComputeCost(X, lab, W, b, lamb, type):
    if type=='cross-entropy':
        return sum(sum(lab.T*(-np.log(EvaluateClassifier(X, W, b)))))/X.shape[0]+lamb*(sum(sum(W**2)))
    if type=='svm':
        val = np.matmul(W, X.T) + b.reshape((-1, 1))
        val_lab = np.sum(val * lab.T, axis=0)
        costs = (val - val_lab + 1)
        indx = costs > 0
        costs = costs * (indx - lab.T)
        cost = np.sum(costs)/X.shape[0] + lamb * np.sum(W ** 2)
        return cost



def ComputeAccuracy(X, y, W, b):
    soft = EvaluateClassifier(X,W,b)
    max_perc = np.argmax(soft,axis = 0)
    max_lab = np.argmax(y, axis = 1)
    error=0
    for i in range(max_perc.shape[0]):
        if max_perc[i] != max_lab[i]:
            error = error+1

    return 1-error/X.shape[0]

def ComputeL(W,b,X,Y):
    Xcalc=X.T
    Ycalc=Y.T
    val = np.dot(W, Xcalc) + b.reshape((-1, 1))
    val_lab = np.sum(val * Ycalc, axis=0)
    cost = (val - val_lab + 1)
    indx = cost > 0
    deriv = indx - Ycalc
    sum_deriv = np.sum(deriv, axis=0)
    g = np.where(Ycalc != 1, deriv, -sum_deriv)

    return g




def ComputeGradients(X, Y, P, W, lamb, type):
    if type=='cross-entropy':
        Gbatch=-(Y.T-P)

        dLw=np.matmul(Gbatch,X)/X.shape[0]
        dLb=np.matmul(Gbatch,np.ones(X.shape[0]))/X.shape[0]
        dJw=dLw+2*lamb*W
        dJb=dLb
        return dJb.reshape(-1,1), dJw

    if type=='svm':
        deriv=ComputeL(W,b,X,Y)
        dJW=np.matmul(deriv,X)/X.shape[0]+2*lamb*W
        dJb=(np.sum(deriv, axis=1))/X.shape[0]
        return dJW, dJb.reshape((W.shape[0],1))



def ComputeGradsNum(X,Y,W,b,lam,h):
    no = W.shape[0]
    d = X.shape[1]
    grad_W = np.zeros(W.shape)
    grad_b = np.zeros(no,1)

    c = ComputeCost(X,Y,W,b,lam)

    for i in range(len(b)):
        b_try = b
        b_try[i] = b_try[i] + h
        c2 = ComputeCost(X,Y,W,b_try,lam)
        grad_b[i]=(c2-c)/h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try=W
            W_try[i,j]=W_try[i,j]+h
            c2 = ComputeCost(X,Y,W_try,b,lam)
            grad_W[i,j] = (c2-c)/h

    return grad_b, grad_W

def MiniBatchGD(X,Y, Xval, Yval,GDParams,W,b,lamb,adapt_learning):
    loss=[]
    loss_val = []

    for i in range(GDParams['n_epoch']):

        indx = np.random.permutation(range(X.shape[0]))
        X = X[indx]
        Y = Y[indx]

        for j in range(X.shape[0] // GDParams['n_batch']):
            j_start = j * GDParams['n_batch']
            j_end = (j + 1) * GDParams['n_batch']
            Xbatch = X[j_start:j_end,:]
            Ybatch = Y[j_start:j_end,:]
            P = EvaluateClassifier(Xbatch, W, b)
            grad_b, grad_W = ComputeGradients(Xbatch, Ybatch, P, W, lamb,'cross-entropy')
            #grad_W, grad_b = ComputeGradients(Xbatch, Ybatch, P, W, lamb, 'svm')
            W = W-GDParams['eta']*grad_W
            b = b-GDParams['eta']*grad_b


        loss.append(ComputeCost(X, Y, W, b, lamb, 'cross-entropy'))
        loss_val.append(ComputeCost(Xval, Yval, W, b, lamb, 'cross-entropy'))
        #loss.append(ComputeCost(X,Y,W,b,lamb,'svm'))
        #loss_val.append(ComputeCost(Xval,Yval,W,b,lamb,'svm'))

        if adapt_learning==True:
            GDParams['eta']=GDParams['eta']*0.9
    GDParams['eta']=0.1
    plt.plot(np.array(range(GDParams['n_epoch'])),np.array(loss),np.array(range(GDParams['n_epoch'])),np.array(loss_val))
    plt.title('SVM with lambda= ' + str(lamb) + ' n_epochs=' + str(GDParams['n_epoch']) + ' n_batch= '+ str(GDParams['n_batch']) + ' learning rate =' + str(GDParams['eta']))
    plt.xlabel('Number of epochs')
    plt.ylabel('Cost')
    plt.show()
    return W, b, loss_val


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    c = ComputeCost(X, Y, W, b, lamda);

    for i in range(len(b)):
        print(i)
        b_try = b
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h

    for i in range(W.shape[0]):
        print(i)
        for j in range(W.shape[1]):
            W_try = W
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2 - c) / h

    return [grad_W, grad_b]


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = b
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = b
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W
            W_try[i, j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = W
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return [grad_W, grad_b]



def Accuracy_Ensemble(vote,Y):
    out=vote-Y.T
    error=(np.where( out == -1 )[0]).shape[0]/out.shape[1]
    return 1-error

def Ensemble_Method(X,Y,Xval,Yval,GDParams,lamb,n_class, mu, sigma):
    W = np.random.normal(mu, sigma, (Y.shape[1], X.shape[1]))
    b = np.random.normal(mu, sigma, (Y.shape[1], 1))
    Ws=[]
    bs=[]
    Ws.append(W)
    bs.append(b)
    GDEnsemble=GDParams.copy()
    GDEnsemble['n_epoch']=1
    for i in range(n_class-1):
        W = np.random.normal(mu, sigma, (Y.shape[1], X.shape[1]))
        b = np.random.normal(mu, sigma, (Y.shape[1], 1))
        Ws.append(W)
        bs.append(b)
    acc=[]
    for i in range(GDParams['n_epoch']):
        votes = np.zeros((W.shape[0], X.shape[0]))
        ohd = np.zeros((W.shape[0], X.shape[0]))
        for j in range(n_class):
            GDEnsemble['eta']=0.001
            indx = np.random.permutation(range(X.shape[0]))
            X = X[indx]
            Y = Y[indx]
            W, b = MiniBatchGD(X[:int(0.15*X.shape[0])],Y[:int(0.15*Y.shape[0])],Xval,Yval,GDEnsemble,Ws[j],bs[j],lamb,adapt_learning=True)
            Ws[j] = W
            bs[j] = b
            #Evaluate classifier
            P=EvaluateClassifier(X,W,b)
            max=np.argmax(P, axis=0)
            for l in range(max.shape[0]):
                votes[max[l],l] += 1
        max=np.argmax(votes,axis=0)

        for l in range(max.shape[0]):
                ohd[max[l],l] += 1
        acc.append(Accuracy_Ensemble(ohd,Y))


    return Ws, bs, acc

def Maximum_Votes(n_class, Ws, bs, Xtest):
    votes = np.zeros((Ws[0].shape[0], Xtest.shape[0]))
    for i in range(n_class):
        P = EvaluateClassifier(normTest, Ws[i], bs[i])
        max = np.argmax(P, axis = 0)
        for l in range(max.shape[0]):
            votes[max[l], l] += 1
    max = np.argmax(votes, axis = 0)

    ohd_ens = np.zeros((W.shape[0], normTest.shape[0]))

    for l in range(max.shape[0]):
        ohd_ens[max[l], l] += 1
    return ohd_ens


def Graph_Search(X, Y, Xval, Yval, GDParams, Xavier, Xtest, Ytest):
    if Xavier == True:
        sigma = 1/np.sqrt(X.shape[0])
    else:
        sigma = 0.01
    W = np.random.normal(mu, sigma, (Y.shape[1], X.shape[1]))
    b = np.random.normal(mu, sigma, (Y.shape[1], 1))
    learning = [0.1, 0.01, 0.001]
    shrinkage = [0.01, 0.1, 0.5]
    batch = [20, 50, 100]
    n_epochs=[70]
    best_W=W.copy()
    best_b=b.copy()
    best_val_costs=[]
    b_acc = 0

    for learn in learning:
        for shr in shrinkage:
            for ba in batch:
                for ep in n_epochs:
                    GDParams['eta']=learn
                    GDParams['n_batch']=ba
                    GDParams['n_epoch']=ep
                    W_opt, b_opt, cost_val = MiniBatchGD(X, Y, Xval, Yval, GDParams, W, b, shr, True)
                    accuracy = ComputeAccuracy(Xtest, Ytest, W_opt, b_opt)
                    if accuracy > b_acc:
                        best_W = W_opt.copy()
                        best_b = b_opt.copy()
                        best_val_costs = cost_val.copy()
                        b_acc = accuracy
                        b_learn = learn
                        b_shrink = shr
                        b_batch = ba
                        b_epoch = ep

    plt.plot(np.array(range(GDParams['n_epoch'])),np.array(best_val_costs))
    plt.title('SVM with lambda= ' + str(b_shrink) + ' n_epochs=' + str(b_epoch) + ' n_batch= '+ str(b_batch) + ' learning rate =' + str(b_learn))
    plt.xlabel('Number of epochs')
    plt.ylabel('Cost')
    plt.show()

    return best_W, best_b, best_val_costs, b_acc, b_learn, b_shrink, b_batch, b_epoch


filetrain = "C:\cifar-10-batches-py\data_batch_1"
filetest = "C:\\cifar-10-batches-py\\test_batch"
fileval = "C:\cifar-10-batches-py\data_batch_2"
datapic = unpickle(filetrain)
valpic = unpickle(fileval)
testpic = unpickle(filetest)

#Validation dataset
val_X = valpic[b'data']
val_Xv = val_X.reshape(len(val_X), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
val_labels = valpic[b'labels']

#Test dataset
test_X = testpic[b'data']
test_Xv = test_X.reshape(len(test_X), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
test_labels = testpic[b'labels']

#Train dataset
train_X = datapic[b'data']
train_labels = datapic[b'labels']

L = train_X.reshape(len(train_X), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
labels = np.array(train_labels)
normX = train_X/np.max(train_X)
#fig, ax = plt.subplots(5,5,figsize=(3,3))
#for h in range(5):
    #for p in range(5):
        #i = np.random.choice(range(len(L)))
        #ax[h][p].set_axis_off()
        #ax[h][p].imshow(L[i])
#fig.show()

ohd = oneHotEncoder(labels)
mean_X = np.mean(normX, axis=0)
std_X = np.std(normX,axis=0)

#Normalize train set
Xtrain = normX-mean_X
#Xtrain = Xtrain[:1000]/std_X
Xtrain = Xtrain/std_X

#Normalize validation
#normVal = val_X[:1000]/np.max(val_X)
normVal = val_X/np.max(val_X)
normVal = normVal-mean_X
normVal = normVal/std_X


#Normalize test
#normTest = test_X[:1000]/np.max(test_X)
normTest = test_X/np.max(test_X)
normTest = normTest-mean_X
normTest = normTest/std_X


#Initialize weights
mu = 0
#sigma = 0.01 # mean and standard deviation
sigma = 1/np.sqrt(Xtrain.shape[0]) # with Xavier initialization
W = np.random.normal(mu, sigma, (ohd.shape[1],train_X.shape[1]))
b = np.random.normal(mu, sigma, (ohd.shape[1],1))
W2=W.copy()
b2=b.copy()
#n son samples y d=32*32*3

lamb=0.1


ohd_val=oneHotEncoder(train_labels)

W, b, _ = MiniBatchGD(Xtrain, ohd, normVal, oneHotEncoder(train_labels),  GDParams, W, b, lamb, adapt_learning = False)

test_acc = ComputeAccuracy(normTest, oneHotEncoder(test_labels), W, b)
#train_acc = ComputeAccuracy(Xtrain, ohd, W, b)


#n_class=10
#Ws, bs, acc=Ensemble_Method(Xtrain,ohd,normVal, ohd_val, GDParams, lamb, n_class, mu, sigma)

#ohd_test=oneHotEncoder(test_labels)

#ohd_ens = Maximum_Votes(n_class, Ws, bs, normTest)
#W, b, val_costs, acc, learn_op, shrink_op, batch_op, ep_op  = Graph_Search(Xtrain, ohd, normVal, ohd_val, GDParams, True, normTest, oneHotEncoder(test_labels))

#acc_test=Accuracy_Ensemble(ohd_ens,oneHotEncoder(test_labels))
#plt.plot(range(GDParams['n_epoch']),acc)
#plt.show()



#Now we will draw the weight matrixes
fig2, axs = plt.subplots(1, W.shape[0])
for i in range(W.shape[0]):
    d = W[i].reshape(32, 32, 3)
    axs[i].set_axis_off()
    axs[i].imshow(d*255)
plt.show()
print("Aqui")
var=input("prompt")






