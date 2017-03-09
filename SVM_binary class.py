#Import Necessary Packages
import math
import pandas as pd
import numpy as np
import sklearn
import scipy as sp
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import time

#Specify Data Set
Data_Set='Advertisement'

#Import CSV Data from Source (Can be Hyperlink or Saved File)
df1=pd.io.parsers.read_csv('C:/Users/BoSun/Documents/'+str(Data_Set)+'_Data_Scaled1.csv',header=None)
df2=pd.io.parsers.read_csv('C:/Users/BoSun/Documents/'+str(Data_Set)+'_Data_Scaled2.csv',header=None)
df3=pd.io.parsers.read_csv('C:/Users/BoSun/Documents/'+str(Data_Set)+'_Data_Scaled3.csv',header=None)
Params={i:[] for i in range(1,4)}
DF={1:df1,2:df3,3:df3}
Features={i:[] for i in range(1,4)}
Classifier={i:[] for i in range(1,4)}
for i in range(1,4):
    #Remove Classification Column and Convert to Desired Classification Numbers
    c=len(DF[i].columns)
    c1=c-1
    C=pd.DataFrame()
    C['Class']=DF[i].iloc[:,c1]
    Classifier[i]=C.values[:,0]
    m=len(Classifier[i])
    df=DF[i].drop(DF[i].columns[c1],axis=1)
    l=len(df.columns)
    
    #Create Features Array
    Features[i]=df.values[:,0:l]
    
    fld=5
    state=12
    kf=KFold(m,n_folds=fld,shuffle=True,random_state=state)
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=kf)
    grid.fit(Features[i], Classifier[i])
    
    print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
    Params[i]=grid.best_params_
print Params
SVMA=0
SVMC=Params[1]['C']
SVMG=Params[1]['gamma']
start=time.time()
kf=KFold(m,n_folds=fld,shuffle=True,random_state=state)
for train_index, test_index in kf:
    Clf_Train=[]
    Clf_Test=[]
    m_train=len(train_index)
    m_test=len(test_index)
    Ft_Train=np.zeros((m_train,l))
    Ft_Test=np.zeros((m_test,l))
    for i in range(0,len(train_index)):
        Ft_Train[i,:]=Features[1][train_index[i]]
        Clf_Train.append(Classifier[1][train_index[i]])
    for i in range(0,len(test_index)):
        Ft_Test[i,:]=Features[1][test_index[i]]
        Clf_Test.append(Classifier[1][test_index[i]])
    clf=SVC(C=SVMC,gamma=SVMG)
    clf.fit(Ft_Train,Clf_Train)
    SVMA=SVMA+clf.score(Ft_Test,Clf_Test)
SVMA=SVMA/fld
SVM_Time=time.time()-start
COP=SVMA/SVM_Time
print("SVM Accuracy for "+str(fld)+"-fold cross-validation is "+str(SVMA))
print("SVM Computation Time is "+str(SVM_Time)+" with Coefficient of Performance "+str(COP))
