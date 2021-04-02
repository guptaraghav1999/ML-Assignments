import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd
import cvxopt
import cvxopt.solvers

data=pd.read_csv('2017EE10544.csv', header=None)
data=np.array(data)
n=3000
m=25
x1=data[0:n,0:25]
t1=data[0:n,25]

class1=0
class2=1
x=np.reshape([],(0,m))
t=[]
for i in range(n):
    if t1[i]==class1:
        x=np.concatenate((x,x1[i:i+1]),axis=0)
        t.append(1)
    if t1[i]==class2:
        x=np.concatenate((x,x1[i:i+1]),axis=0)
        t.append(-1)
        
n=len(t)

def kernelL(x1,x2):
    return np.dot(x1,x2)

def kernelR(x1,x2,gm):
    return np.exp((-linalg.norm(x1-x2)**2)*(gm))

def accuracy(y_pred,t_test):
    a=0
    k=len(t_test)
    for i in range(k):
        if y_pred[i]==t_test[i]:
            a=a+1
    return a/k


def cvxR(xTrain,tTrain,C,gm):
    n=len(xTrain)
    K=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
    #         K[i,j]=kernelL(x[i],x[j])
            K[i,j]=kernelR(xTrain[i],xTrain[j],gm)
        
    P=cvxopt.matrix(np.outer(tTrain,tTrain)*K) #K is kernel matrix
    q=cvxopt.matrix(np.ones(n)*(-1))
    A=cvxopt.matrix(tTrain,(1,n),tc='d')
    b=cvxopt.matrix(0, tc='d')
    G=cvxopt.matrix(np.vstack((np.diag(np.ones(n)*(-1)),np.identity(n))))
    h=cvxopt.matrix(np.hstack((np.zeros(n),np.ones(n)*C)))   
    solution=cvxopt.solvers.qp(P,q,G,h,A,b)
    a=np.ravel(solution['x']) #Lagrange Multipliers
    sv=a > 1e-5 #Support Vectors
    sv_a=a[sv]
    sv_x=xTrain[sv]
    tTrain=np.array(tTrain,dtype=int)
    sv_t=tTrain[sv]
    sv_index=np.arange(len(a))[sv]
    # print(len(sv_index))
    # print(sv_index)
    plt.scatter(range(len(sv_index)),sv_index)
    intercept=0
    for i in range(len(sv_a)):
        intercept+=sv_t[i]
        temp=0
        for j in range(len(sv_a)):
            temp+=sv_t[j]*sv_a[j]*K[sv_index[i],sv_index[j]]
        intercept-=temp 
    intercept/=len(sv_a)
    # print(intercept)
    y_pred=np.zeros(len(xTrain))
    for i in range(len(xTrain)):
        s=0
        for j in range(len(sv_a)):
            s+=sv_a[j]*sv_t[j]*K[i,sv_index[j]]
        y_pred[i]=s
    y_pred+=intercept
    y_pred=np.sign(y_pred) #Prediction on given data set
    return accuracy(y_pred,tTrain)



def cvxL(xTrain,tTrain,C):
    n=len(xTrain)
    K=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j]=kernelL(xTrain[i],xTrain[j])
#             K[i,j]=kernelR(xTrain[i],xTrain[j],gm)
        
    P=cvxopt.matrix(np.outer(tTrain,tTrain)*K) #K is kernel matrix
    q=cvxopt.matrix(np.ones(n)*(-1))
    A=cvxopt.matrix(tTrain,(1,n),tc='d')
    b=cvxopt.matrix(0, tc='d')
    G=cvxopt.matrix(np.vstack((np.diag(np.ones(n)*(-1)),np.identity(n))))
    h=cvxopt.matrix(np.hstack((np.zeros(n),np.ones(n)*C)))   
    solution=cvxopt.solvers.qp(P,q,G,h,A,b)
    a=np.ravel(solution['x']) #Lagrange Multipliers
    sv=a > 1e-5 #Support Vectors
    sv_a=a[sv]
    sv_x=xTrain[sv]
    tTrain=np.array(tTrain,dtype=int)
    sv_t=tTrain[sv]
    sv_index=np.arange(len(a))[sv]
    # print(len(sv_index))
    # print(sv_index)
    intercept=0
    for i in range(len(sv_a)):
        intercept+=sv_t[i]
        temp=0
        for j in range(len(sv_a)):
            temp+=sv_t[j]*sv_a[j]*K[sv_index[i],sv_index[j]]
        intercept-=temp 
    intercept/=len(sv_a)
    # print(intercept)
    y_pred=np.zeros(len(xTrain))
    for i in range(len(xTrain)):
        s=0
        for j in range(len(sv_a)):
            s+=sv_a[j]*sv_t[j]*K[i,sv_index[j]]
        y_pred[i]=s
    y_pred+=intercept
    y_pred=np.sign(y_pred) #Prediction on given data set
    return accuracy(y_pred,tTrain)

# cvxL(x,t,0.1)


def crossScoreR(folds,c,gm):
    b=n//folds
    ac=0
    y=[]
    y1=[]
    f=0
    for i in range(folds):
        temp=np.array(x[f:f+b,0:m])
        temp1=np.array(t[f:f+b])
        f=f+b
        y.append(temp)
        y1.append(temp1)
    
    for i in range(folds):
        x_test=y[i]
        t_test=y1[i]
        x_train=np.reshape([],(0,m))
        t_train=[]
        
        for j in range(folds):
            if i!=j:
                x_train=np.concatenate((x_train,y[j]),axis=0)
                t_train=np.concatenate((t_train,y1[j]),axis=0)
             

        ac+=cvxR(x_train,t_train,c,gm)
    
    return ac/folds

def crossScoreL(folds,c):
    b=n//folds
    ac=0
    y=[]
    y1=[]
    f=0
    for i in range(folds):
        temp=np.array(x[f:f+b,0:m])
        temp1=np.array(t[f:f+b])
        f=f+b
        y.append(temp)
        y1.append(temp1)
    
    for i in range(folds):
        x_test=y[i]
        t_test=y1[i]
        x_train=np.reshape([],(0,m))
        t_train=[]
        
        for j in range(folds):
            if i!=j:
                x_train=np.concatenate((x_train,y[j]),axis=0)
                t_train=np.concatenate((t_train,y1[j]),axis=0)
             

        ac+=cvxL(x_train,t_train,c)
    
    return ac/folds

# crossScoreL(4,0.1)