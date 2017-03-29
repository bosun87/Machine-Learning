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
import time

#Define Values of Structural Constants

fld=5 #Number of Folds for k-Fold Cross-Validation
state=12 #Seed Number for random generation of k-Fold Labels
Seed=1 #Replica Generation Seed
np.random.seed(Seed) #Activate Seed

#Define the functional forms of the kernels which will passed onto the data and used in classification
def Gauss_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Gaussian=math.exp(-(Vector.dot(Vector))/(2*sigma**2))
    return Gaussian
def Sine_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Sine=math.sin(k*(np.linalg.norm(Vector)))
    return Sine
def Hyper_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Hyper=math.tanh(np.linalg.norm(Vector))
    return Hyper
def Sinc_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Sinc=np.sinc(kappa*(np.linalg.norm(Vector)))
    return Sinc
def Exp_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Exp=math.exp(-np.linalg.norm(Vector)/xi)
    return Exp
def Erfc_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Erfc=math.erfc((1/chi)*np.linalg.norm(Vector))
    return Erfc
def Gamma_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Gamma=sp.special.gamma(g*np.linalg.norm(Vector))
    return Gamma
def Airy_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Airy, aip, bi, bip=sp.special.airy(a*np.linalg.norm(Vector))
    return Airy
def Fermi_Kernel(Vector1, Anchor_Vector):
    Vector=Vector1-Anchor_Vector
    Fermi=1/(math.exp(q*np.linalg.norm(Vector))+1)
    return Fermi

#Provide User Specified Functional Caller
Fn=1 #Enter Number Corresponding to Function to Be Employed
Fun=""
Which_Fn=None
if Fn==1:
    Which_Fn=Gauss_Kernel
    Fun="Gaussian"
elif Fn==2:
    Which_Fn=Sine_Kernel
    Fun="Sine"
elif Fn==3:
    Which_Fn=Hyper_Kernel
    Fun="Hyperbolic"
elif Fn==4:
    Which_Fn=Sinc_Kernel
    Fun="Sinc"
elif Fn==5:
    Which_Fn=Exp_Kernel
    Fun="Exponential"
elif Fn==6:
    Which_Fn=Erfc_Kernel
    Fun="Erfc"
elif Fn==7:
    Which_Fn=Gamma_Kernel
    Fun="Gamma"
elif Fn==8:
    Which_Fn=Airy_Kernel
    Fun="Airy"
elif Fn==9:
    Which_Fn=Fermi_Kernel
    Fun="Fermi"
    
#Specify Data Set
Data_Set='LVST'

#Import CSV Data from Source (Can be Hyperlink or Saved File)
df1=pd.io.parsers.read_csv('C:/Users/BoSun/Documents/'+str(Data_Set)+'_Data_Scaled1.csv',header=None)
df2=pd.io.parsers.read_csv('C:/Users/BoSun/Documents/'+str(Data_Set)+'_Data_Scaled2.csv',header=None)
df3=pd.io.parsers.read_csv('C:/Users/BoSun/Documents/'+str(Data_Set)+'_Data_Scaled3.csv',header=None)

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

#Define Values of All Constants
sigma=math.sqrt(len(Features[1][0])) #Gaussian Kernel Parameter
k=1 #Sine Kernel
kappa=1/sigma #Sinc Kernel
xi=sigma #Exp Kernel
chi=sigma #Erfc Kernel
g=1/sigma #Gamma Kernel
a=1/sigma #Airy Kernel
q=1/sigma #Fermi Kernel

#Initialize Values of Scoring Functions
FVMA=0
Overlap=0
Overlap2=0

#Specify Number of Replicas and Number of Anchor Points
Num_Replicas=15 #Number of Replicas
v=39 #Number of Anchor Points

#Generate Replica Vectors according to continuous random interval 
Num_Features=len(Features[1][0])
Replicas={i:[] for i in range(0,Num_Replicas)}
Replicas2={i:[] for i in range(0,Num_Replicas)}
for i in range(0,Num_Replicas):
    Replicas[i]=np.random.normal(loc=0,scale=2,size=(v,Num_Features))
for i in range(0,Num_Replicas):
    Replicas2[i]=np.random.random((v,Num_Features))   
            
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
   
    CoefficientsG={i:[] for i in range(0,Num_Replicas)}
    CoefficientsE={i:[] for i in range(0,Num_Replicas)}
    CoefficientsER={i:[] for i in range(0,Num_Replicas)}
    CoefficientsA={i:[] for i in range(0,Num_Replicas)}
    CoefficientsF={i:[] for i in range(0,Num_Replicas)}
    G={i:[] for i in range(0,Num_Replicas)}
    E={i:[] for i in range(0,Num_Replicas)}
    ER={i:[] for i in range(0,Num_Replicas)}
    A={i:[] for i in range(0,Num_Replicas)}
    F={i:[] for i in range(0,Num_Replicas)}
    for i in range(0,Num_Replicas):
        G[i]=np.zeros((m_train,v))
        E[i]=np.zeros((m_train,v))
        ER[i]=np.zeros((m_train,v))
        A[i]=np.zeros((m_train,v))
        F[i]=np.zeros((m_train,v))
    for k in range(0,Num_Replicas):
        for i in range(0,m_train):
            for j in range(0,v):
                G[k][i][j]=Gauss_Kernel(Ft_Train[i],Replicas[k][j])
                E[k][i][j]=Exp_Kernel(Ft_Train[i],Replicas[k][j])
                ER[k][i][j]=Erfc_Kernel(Ft_Train[i],Replicas[k][j])
                A[k][i][j]=Airy_Kernel(Ft_Train[i],Replicas[k][j])
                F[k][i][j]=Fermi_Kernel(Ft_Train[i],Replicas[k][j])
    for i in range(0,Num_Replicas):
        CoefficientsG[i]=np.dot(np.linalg.pinv(G[i]),Clf_Train)
        CoefficientsE[i]=np.dot(np.linalg.pinv(E[i]),Clf_Train)
        CoefficientsER[i]=np.dot(np.linalg.pinv(ER[i]),Clf_Train)
        CoefficientsA[i]=np.dot(np.linalg.pinv(A[i]),Clf_Train)
        CoefficientsF[i]=np.dot(np.linalg.pinv(F[i]),Clf_Train)

    MG={i:[] for i in range(0,Num_Replicas)}
    ME={i:[] for i in range(0,Num_Replicas)}
    MER={i:[] for i in range(0,Num_Replicas)}
    MA={i:[] for i in range(0,Num_Replicas)}
    MF={i:[] for i in range(0,Num_Replicas)}
    for i in range(0,Num_Replicas):
        MG[i]=np.zeros((m_test,v))
        ME[i]=np.zeros((m_test,v))
        MER[i]=np.zeros((m_test,v))
        MA[i]=np.zeros((m_test,v))
        MF[i]=np.zeros((m_test,v))
    for k in range(0,Num_Replicas):
        for i in range(0,m_test):
            for j in range(0,v):
                MG[k][i][j]=Gauss_Kernel(Ft_Test[i],Replicas[k][j])
                ME[k][i][j]=Exp_Kernel(Ft_Test[i],Replicas[k][j])
                MER[k][i][j]=Erfc_Kernel(Ft_Test[i],Replicas[k][j])
                MA[k][i][j]=Airy_Kernel(Ft_Test[i],Replicas[k][j])
                MF[k][i][j]=Fermi_Kernel(Ft_Test[i],Replicas[k][j])
    Predicted_ValuesG={i:[] for i in range(0,Num_Replicas)}
    Predicted_ValuesE={i:[] for i in range(0,Num_Replicas)}
    Predicted_ValuesER={i:[] for i in range(0,Num_Replicas)}
    Predicted_ValuesA={i:[] for i in range(0,Num_Replicas)}
    Predicted_ValuesF={i:[] for i in range(0,Num_Replicas)}
    for i in range(0,Num_Replicas):
        Predicted_ValuesG[i]=np.dot(MG[i],CoefficientsG[i])
        Predicted_ValuesE[i]=np.dot(ME[i],CoefficientsE[i])
        Predicted_ValuesER[i]=np.dot(MER[i],CoefficientsER[i])
        Predicted_ValuesA[i]=np.dot(MA[i],CoefficientsA[i])
        Predicted_ValuesF[i]=np.dot(MF[i],CoefficientsF[i])
        for j in range(0,m_test):
            Predicted_ValuesG[i][j]=cmp(Predicted_ValuesG[i][j],0)
            Predicted_ValuesE[i][j]=cmp(Predicted_ValuesE[i][j],0)
            Predicted_ValuesER[i][j]=cmp(Predicted_ValuesER[i][j],0)
            Predicted_ValuesA[i][j]=cmp(Predicted_ValuesA[i][j],0)
            Predicted_ValuesF[i][j]=cmp(Predicted_ValuesF[i][j],0)
    Voting=[0 for i in range(0,m_test)]
    for i in range(0,m_test):
        for j in range(0,Num_Replicas):
            Voting[i]=Voting[i]+Predicted_ValuesG[j][i]+Predicted_ValuesE[j][i]+Predicted_ValuesER[j][i]+Predicted_ValuesA[j][i]+Predicted_ValuesF[j][i]
        Voting[i]=Voting[i]/(5*Num_Replicas)
        Voting[i]=cmp(Voting[i],0)
    Accuracy2=0
    for i in range(0,m_test):
        if float(Voting[i])==float(Clf_Test[i]):
            Accuracy2=Accuracy2+1
        elif float(Voting[i])!=float(Clf_Test[i]):
            Accuracy2=Accuracy2+0
    Accuracy2=float(Accuracy2)/m_test
    FVMA=FVMA+Accuracy2
FVMA=FVMA/fld
FVM_Time=time.time()-start
COP=FVMA/FVM_Time
print("FVMA Accuracy After "+str(fld)+"-fold cross validation with "+str(v)+" Anchor Points and "+str(Num_Replicas)+" Replicas is "+str(FVMA))
print("FVMA Computation Time is "+str(FVM_Time)+" with Coefficient of Performace "+str(COP))