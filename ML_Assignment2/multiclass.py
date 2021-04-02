from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('2017EE10544.csv',header=None)
data=np.array(data)
n=3000
m=10
x=data[0:n,0:m]
t=data[0:n,25]



def accuracy(y_pred,t_test):
    a=0
    k=len(t_test)
    for i in range(k):
        if y_pred[i]==t_test[i]:
            a=a+1
    return a/k

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
                
        clf = svm.SVC(kernel='linear', C = c)
        clf.fit(x_train,t_train)
        y_pred=clf.predict(x_test)
        ac+=accuracy(t_test,y_pred)
    
    return ac/folds

def crossScoreP(folds,c,d):
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
                
        clf = svm.SVC(kernel='poly', C = c,degree=d)
        clf.fit(x_train,t_train)
        y_pred=clf.predict(x_test)
        ac+=accuracy(t_test,y_pred)
    
    return ac/folds

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
             
        clf = svm.SVC(kernel='rbf', C = c,gamma=gm)
        clf.fit(x_train,t_train)
        y_pred=clf.predict(x_test)
        ac+=accuracy(t_test,y_pred)
    
    return ac/folds

def crossScoreS(folds,c,gm):
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
             
        clf = svm.SVC(kernel='sigmoid', C = c,gamma=gm)
        clf.fit(x_train,t_train)
        y_pred=clf.predict(x_test)
        ac+=accuracy(t_test,y_pred)
    
    return ac/folds

def scoreL(c):
    clf=svm.SVC(kernel='linear',C=c)
    clf.fit(x,t)
    y_pred=clf.predict(x)
    return accuracy(t,y_pred)

def scoreP(c,d):
    clf=svm.SVC(kernel='poly',degree=d,C=c)
    clf.fit(x,t)
    y_pred=clf.predict(x)
    return accuracy(t,y_pred)

def scoreR(c,gm):
    clf=svm.SVC(kernel='rbf',C=c,gamma=gm)
    clf.fit(x,t)
    y_pred=clf.predict(x)
    return accuracy(t,y_pred)

def scoreS(c,gm):
    clf=svm.SVC(kernel='sigmoid',C=c,gamma=gm)
    clf.fit(x,t)
    y_pred=clf.predict(x)
    return accuracy(t,y_pred)

def tuneL(folds):
    arr=np.linspace(0.1,0.5,20)
    c=arr[0]
    y=[]
    score=crossScoreL(4,c)
    for i in arr:
        temp=crossScoreL(4,i)
        y.append(temp)
        if temp>score:
            score=temp
            c=i
    plt.plot(arr,y,'ro')
    plt.xlabel('C')
    plt.ylabel('Cross-Validation Accuracy')
    return (c,score)


def tuneP1(folds):
    arr1=np.linspace(0.1,1.5,20)
    arr2=range(1,10)
    c=0.1
    degree=2
    y=[]
    score=crossScoreP(4,c,degree)
    for i in arr1:
#         for j in arr2:
        temp=crossScoreP(4,i,degree)
        y.append(temp)
        if temp>score:
            score=temp
            c=i
#             degree=j
    plt.plot(arr1,y,'ro')
    plt.xlabel('C')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Degree=5')
    return (c,degree,score)

def tuneRc(folds,gm):
    arrc=np.linspace(0.1,1.5,20)
    arrg=range(1,10)
    c=0.1
    y=[]
    score=crossScoreR(4,c,gm)
    for i in arrc:
        temp=crossScoreR(4,i,gm)
        y.append(temp)
        if temp>score:
            score=temp
            c=i
    plt.plot(arrc,y,'ro')
    plt.xlabel('C')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Gamma=0.1')
    return (c,gm,score)

def tuneRg(folds,c):
    arrg=np.linspace(1,5,10)
    gm=1
    y=[]
    score=crossScoreR(4,c,gm)
    for i in arrg:
        temp=crossScoreR(4,c,i)
        y.append(temp)
        if temp>score:
            score=temp
            gm=i
    plt.plot(arrg,y,'ro')
    plt.xlabel('Gamma')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('C=1')
    return (c,gm,score)

def plotR(arrx,arry):
    lenx=len(arrx)
    leny=len(arry)
    Z=np.zeros((leny,lenx))
    for i in range(leny):
        for j in range(lenx):
            Z[i,j]=crossScoreR(4,arrx[j],arry[i])
    X,Y=np.meshgrid(arrx,arry)
#     print(Z)
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.xlabel('C')
    plt.ylabel('Gamma')
    plt.colorbar()              

def plotP(arrx,arry):
    lenx=len(arrx)
    leny=len(arry)
    Z=np.zeros((leny,lenx))
    for i in range(leny):
        for j in range(lenx):
            Z[i,j]=crossScoreP(4,arrx[j],arry[i])
    X,Y=np.meshgrid(arrx,arry)
#     print(Z)
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.xlabel('C')
    plt.ylabel('Degree')
    plt.colorbar()     

def plotS(arrx,arry):
    lenx=len(arrx)
    leny=len(arry)
    Z=np.zeros((leny,lenx))
    for i in range(leny):
        for j in range(lenx):
            Z[i,j]=crossScoreS(4,arrx[j],arry[i])
    X,Y=np.meshgrid(arrx,arry)
#     print(Z)
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.xlabel('C')
    plt.ylabel('Gamma')
    plt.colorbar()           

def tuneR(folds):
    arrc=np.linspace(0.1,3,30)
    arrg=np.linspace(0.01,0.2,20)
    c=arrc[0]
    gm=arrg[0]
    y=[]
    score=crossScoreR(4,c,gm)
    for i in arrc:
        for j in arrg:
            temp=crossScoreR(4,i,j)
            if temp>score:
                score=temp
                c=i
                gm=j
    plotR(arrc,arrg)
    return (c,gm,score)



def tuneP(folds):
    arrc=np.linspace(0.1,1.5,20)
    arrg=range(1,8)
    c=arrc[0]
    degree=arrg[0]
    y=[]
    score=crossScoreP(4,c,degree)
    for i in arrc:
        for j in arrg:
            temp=crossScoreP(4,i,j)
            if temp>score:
                score=temp
                c=i
                degree=j
    plotP(arrc,arrg)
    return (c,degree,score)

