from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('train_set.csv',header=None)
data=np.array(data)
n=10000
m=25
x=data[0:n,0:m]
# temp=np.max(x,axis=0)-np.min(x,axis=0)
# x/=temp
temp=[]
for j in range(25):
    k=max(x[:,j:j+1])-min(x[:,j:j+1])
    x[:,j:j+1]/=k
    temp.append(k)
# print(x)
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
        temp=np.array(x[f:f+b,0:25])
        temp1=np.array(t[f:f+b])
        f=f+b
        y.append(temp)
        y1.append(temp1)
    
    for i in range(folds):
        x_test=y[i]
        t_test=y1[i]
        x_train=np.reshape([],(0,25))
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

def scoreL(c):
    clf=svm.SVC(kernel='linear',C=c)
    clf.fit(x,t)
    y_pred=clf.predict(x)
    return accuracy(t,y_pred)

def crossScoreP(folds,c,d):
    b=n//folds
    ac=0
    y=[]
    y1=[]
    f=0
    for i in range(folds):
        temp=np.array(x[f:f+b,0:25])
        temp1=np.array(t[f:f+b])
        f=f+b
        y.append(temp)
        y1.append(temp1)
    
    for i in range(folds):
        x_test=y[i]
        t_test=y1[i]
        x_train=np.reshape([],(0,25))
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

def scoreP(c,d):
    clf=svm.SVC(kernel='poly',degree=d,C=c)
    clf.fit(x,t)
    y_pred=clf.predict(x)
    return accuracy(t,y_pred)

def crossScoreR(folds,c,gm):
    b=n//folds
    ac=0
    y=[]
    y1=[]
    f=0
    for i in range(folds):
        temp=np.array(x[f:f+b,0:25])
        temp1=np.array(t[f:f+b])
        f=f+b
        y.append(temp)
        y1.append(temp1)
    
    for i in range(folds):
        x_test=y[i]
        t_test=y1[i]
        x_train=np.reshape([],(0,25))
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

def scoreR(c,gm):
    clf=svm.SVC(kernel='rbf',C=c,gamma=gm)
    clf.fit(x,t)
    y_pred=clf.predict(x)
    return accuracy(t,y_pred)

def tuneL(folds):
    arr=np.linspace(0.1,1.5,20)
    c=0.1
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
    plt.show()
    return (c,score)

def tuneP(folds):
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
    plt.show()
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
    plt.show()
    return (c,gm,score)


def tuneRg(folds,c):
    arrg=np.linspace(0.1,1,50)
    print(arrg)
    gm=0.1
    y=[]
    score=crossScoreR(4,c,gm)
    for i in arrg:
        temp=crossScoreR(4,c,i)
        y.append(temp)
        if temp>score:
            score=temp
            gm=i
    plt.plot(arrg,y,'ro')
    print(y)
    plt.xlabel('Gamma')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('C=1')
    plt.show()
    return (c,gm,score)


def tuneR(folds):
    arrc=np.linspace(1,5,10)
    arrg=np.linspace(1,5,10)
    c=1
    gm=1
    y=[]
    score=crossScoreR(4,c,gm)
    for i in arrc:
        for j in arrg:
            temp=crossScoreR(4,i,j)
            if temp>score:
                score=temp
                c=i
                gm=j
#     plt.plot(arr1,y,'ro')
#     plt.xlabel('C')
#     plt.ylabel('Cross-Validation Accuracy')
#     plt.title('Degree=5')
        # plt.show()
    return (c,gm,score)

# tuneR(4)

# crossScoreR(4,2.33,3.66)

data=pd.read_csv('test_set.csv',header=None)
data=np.array(data)
x_test=data[0:2000,0:25]
for j in range(25):
    x_test[:,j:j+1]/=temp[j]
c=5.6
gm=4.8
clf = svm.SVC(kernel='rbf', C = c,gamma=gm)
clf.fit(x,t)
y_pred=clf.predict(x_test)
print(x_test)


# y_pred=np.array(y_pred,dtype=int)
# z=range(2000)
# dict = {'id' : z, 'class' : y_pred}
# df = pd.DataFrame(dict)
# df.to_csv('pred.csv', index=False)

