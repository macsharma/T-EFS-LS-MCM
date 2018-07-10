#plot decision boundary
import numpy as np
import pandas as pd
from time import time
#from sklearn.model_selection import StratifiedKFold
import os
#from sklearn.cluster import KMeans
#from sklearn import datasets
#import matplotlib.pyplot as plt


#from scipy.stats import mode
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
#from sklearn.utils import resample
#from sklearn.neighbors import NearestNeighbors
#from numpy.matlib import repmat
#from sklearn.covariance import OAS,LedoitWolf
#%%

hpc=True
print (os.getcwd())
if(hpc==False):
    path2 = 'D:\\Mega\\phd\\classification_datasets\\label_partition\\label_noise\\temp_enc\\LSMCM\\'
    path1= path2 + "input_noise_experiments\\noise_0\\LSMCM_L1_RBF_FULL_compress"
    datapath = 'D:\\Hubic\\deep learning datasets\\dissimilarity datasets\\openML_datasets\\gaussian_noise'
else:
    path2 = "/home/ee/phd/eez142368/classification_datasets/label_partition/label_noise/temp_enc/LSMCM/"
    path1= path2 + "input_noise_experiments/noise_0/LSMCM_L1_RBF_FULL_compress"
    datapath = '/scratch/ee/phd/eez142368/openML_datasets/gaussian_noise'
os.chdir(path1)
print (os.getcwd())
#import the classifier
from train.LSMCM_C_CLR_compress import LSMCM_classifier
#%%
def standardize(xTrain):
    me=np.mean(xTrain,axis=0)
    std_dev=np.std(xTrain,axis=0)
    #remove columns with zero std
    idx=(std_dev!=0.0)
    print(idx.shape)
    xTrain[:,idx]=(xTrain[:,idx]-me[idx])/std_dev[idx]
    return xTrain,me,std_dev
#%%

#1,3,10,11,12
#2,4,5,6,7,8,9,13
noise = 0
for dataset_name in [1,3,10,11,12]:
    # datasets are named xTrain_(dataset_num)_(noise) in my case, use your own file here
    #load the dataset, in the form samples X features
    xTrain = np.loadtxt(datapath+'/xTrain_%d_%d.csv'%(dataset_name,noise),delimiter=',')
    xTest = np.loadtxt(datapath+'/xTest_%d_%d.csv'%(dataset_name,0),delimiter=',')
    xVal = np.loadtxt(datapath+'/xVal_%d_%d.csv'%(dataset_name,0),delimiter=',')
    yTrain = np.array(np.loadtxt(datapath+'/yTrain_%d_%d.csv'%(dataset_name,0),delimiter=','),dtype=np.int32)
    yTest = np.array(np.loadtxt(datapath+'/yTest_%d_%d.csv'%(dataset_name,0),delimiter=','),dtype=np.int32)
    yVal = np.array(np.loadtxt(datapath+'/yVal_%d_%d.csv'%(dataset_name,0),delimiter=','),dtype=np.int32)
    #perform zero mean unit variance transform
    if(dataset_name in [20,30]):
        if(dataset_name ==12):
            xTrain[:,0:10],me,std=standardize(xTrain[:,0:10])
            idx = (std!=0.0)
            xTest[:,0:10][:,idx] = (xTest[:,0:10][:,idx]-me[idx])/std[idx]
            xVal[:,0:10][:,idx] = (xVal[:,0:10][:,idx]-me[idx])/std[idx]
        if(dataset_name ==10):
            xTrain = xTrain/255
            xTest = xTest/255
            xVal = xVal/255            
    else:
        xTrain,me,std=standardize(xTrain)
        idx = (std!=0.0)
        xTest[:,idx] = (xTest[:,idx]-me[idx])/std[idx]
        xVal[:,idx] = (xVal[:,idx]-me[idx])/std[idx]

    # parameter 'xyz' indices are denoted by 'xyz_idx' and it saves the indices, instead of strings in 'xyz' to be saved in a pandas dataframe
    # when running the functions the parameter 'xyz = 'abc' eg. kernel_type = 'rbf' can be passed as is 
    # the parameter indices eg. 'xyz_idx' : kernel_type_idx is not required unless you wish to save the results in a numpy array as I have
#    Ca = [0,1e-05,1e-03,1e-02,1e-01,1] #hyperparameter 1 #loss function parameter
    Ca = [0]
    Cb = [1e-04,1e-03,1e-02,1e-01] #hyperparameter 2 #when using L1 or L2 or ISTA penalty

    Cc = [0,1e-04,1e-03,1e-02,1e-01,1,10] #hyperparameter 2 #when using elastic net penalty (this parameter should be between 0 and 1)
    Cc = [0] #hyperparameter 2 #when using elastic net penalty (this parameter should be between 0 and 1)


    kernel_type1 = {0:'linear', 1:'rbf', 2:'sin', 3:'tanh', 4:'TL1', 5:'linear_primal', 6:'rff_primal', 7:'nystrom_primal'} #type of kernel to use
    kernel_type = 'rbf'
    kernel_type_idx = 1
    gamma1 = [1e-04,1e-03,1e-02,1e-01,1,10,100,1000,10000] #hyperparameter3 (kernel parameter for non-linear classification or regression)
#    gamma1 = [1] #hyperparameter3 (kernel parameter for non-linear classification or regression)

    batch_sz = int(np.min([128,xTrain.shape[0]/2]))
    iterMax1 = 1000
    iterMax2 = 10
    eta = 1e-02
    tol = 1e-04
    update_type1 =  {0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}#{0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}
    update_type ='adam'
    update_type_idx = 6
    reg_type1 = {0:'l1', 1:'l2', 2:'en', 4:'ISTA', 5:'M'} #{0:'l1', 1:'l2', 2:'en', 4:ISTA, 5:'M'}#ISTA: iterative soft thresholding (proximal gradient) #type of regularization
    reg_type = 'l1'
    reg_type_idx = 0
    class_weighting1 = {0:'average', 1:'balanced'}#{0:'average', 1:'balanced'} #whether to use class weighting for imbalanced learning
    class_weighting = 'average'
    class_weighting_idx = 0
    upsample1a =  {0:False, 1:True} #{0:'False', 1:'True'} #whether to perform upsampling of the data
    upsample1  = False
    upsample1_idx = 0
    PV_scheme1 = {0:'kmeans', 1:'renyi'}  #{0:'kmeans', 1:'renyi'} #prototype vector selection scheme
    PV_scheme = 'renyi'
    PV_scheme_idx = 1
    n_components = int(np.sqrt(xTrain.shape[0])) #number of prototype vectors
    do_pca_in_selection1 = {0:False,1:True}   #whether to perform PCA while selecting the prototype vectors
    do_pca_in_selection = False 
    do_pca_in_selection_idx = 0
    levels1 = [3]  #not to be used here
    result=np.zeros((1,25))
    #tune only two hyperparameters C2 and gamma1, keep others at default
    for C1 in Ca: #Not required with L1 regularization, leave it at default
        for C2 in Cb:
            for C3 in Cc: #Not required unless using elastic net or ISTA or Margin (M)
                for gamma in gamma1:
                    for levels in levels1: # not required

                        t0=time()
                        #train classifier
                        mcm = LSMCM_classifier(C1 = C1, C2 = C2, C3 = C3, gamma = gamma, kernel_type = kernel_type, batch_sz = batch_sz, iterMax1 = iterMax1, iterMax2 =iterMax2, eta = eta, 
                                  tol = tol, update_type = update_type, reg_type = reg_type, class_weighting = class_weighting,PV_scheme = PV_scheme, 
                                  n_components = n_components, do_pca_in_selection = do_pca_in_selection,do_upsample = upsample1 )
                        W, f,iters,fvals,subset,me,std = mcm.fit(xTrain,yTrain)
                        t1=time()
                        time_elapsed=t1-t0
                        #predict on train and test set
                        train_pred,train_scores = mcm.predict(xTrain, xTrain, W,  me, std, subset)
                        val_pred, val_scores = mcm.predict(xVal, xTrain, W,  me, std, subset)
                        #calculate accuracy and F1 scores
                        train_acc = mcm.accuracy_classifier(yTrain,train_pred)
                        val_acc = mcm.accuracy_classifier(yVal,val_pred)
                        train_f1= f1_score(yTrain, train_pred, average='weighted') 
                        val_f1 =f1_score(yVal, val_pred, average='weighted') 
                        print ('C1=%0.3f, C2=%0.3f -> train acc= %0.2f, val acc=%0.2f'%(C1,C2,train_acc,val_acc))
    #                                                                                        print('batch_sz=%d'%(batch_sz))
                        #save the results
                        non_zero_weights=0
                        total_weights = 0
                        if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):
                            W2=np.zeros(W.shape)
                            W2[W!=0.0]=1
                            W2 =np.sum(W2,axis=1)
                            non_zero_weights+=np.sum(W2 != 0)                            
                            total_weights += np.sum(W!=0)
                        else:
                            non_zero_weights+=np.sum(W!=0)
                            total_weights = non_zero_weights

                        result=np.append(result,np.array([[train_acc, val_acc, time_elapsed, non_zero_weights, C1, C2, C3, kernel_type_idx, gamma,
                                                            batch_sz,iterMax1, eta, tol, update_type_idx, reg_type_idx, class_weighting_idx, 
                                                            upsample1_idx ,train_f1, val_f1, PV_scheme_idx, n_components, do_pca_in_selection_idx,
                                                            iterMax2,total_weights,levels]]),axis=0)

    
    result=result[1:,]
    #print result
    result_pd=pd.DataFrame(result,index=range(0,result.shape[0]),columns=['0_train_acc', '1_val_acc', '2_time_elapsed', '3_non_zero_weights', '4_C1', '5_C2'
                                                                           , '6_C3', '7_kernel_type_idx', '8_gamma', '9_batch_sz','10_iterMax1', '11_eta', '12_tol', 
                                                                           '13_update_type_idx', '14_reg_type_idx', '15_class_weighting_idx', 
                                                                '16_upsample1_idx' ,'17_train_f1', '18_val_f1', '19_PV_scheme_idx', '20_n_components', '21_do_pca_in_selection_idx',
                                                                '22_iterMax2','23_total_weights','24_levels'])
    
    result_pd.to_csv(path1+"/results/Train_results_%d"%(dataset_name)+".csv")
    #finding the best hyperparameter settings
    max_acc=np.max(result[:,18])
#    max_acc_idx=result[:,18]==max_acc
    max_acc_idx=(result[:,18]<=max_acc)*(result[:,18]>=max_acc-0.005)
    min_sv=np.min(result[max_acc_idx,23])
    min_sv_idx=result[:,23]==min_sv
    best_val_idx=max_acc_idx*min_sv_idx
    best_val_acc=np.where(best_val_idx==True)[0][0]
    
    C1=result[best_val_acc,4]
    C2=result[best_val_acc,5]
    C3=result[best_val_acc,6]

    kernel_type_idx = int(result[best_val_acc,7]) #{0:'linear', 1:'rbf', 2:'sin', 3:'tanh', 4:'TL1'}
    kernel_type = kernel_type1[kernel_type_idx]
    gamma = result[best_val_acc,8] #hyperparameter3 (kernel parameter for non-linear classification or regression)
    batch_sz= int(result[best_val_acc,9]) #batch_size
    iterMax1 = int(result[best_val_acc,10]) #max number of iterations for inner SGD loop
    eta = result[best_val_acc,11] #initial learning rate
    tol = result[best_val_acc,12]#tolerance to cut off SGD
    update_type_idx = int(result[best_val_acc,13])#{0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}
    update_type = update_type1[update_type_idx]
    reg_type_idx = int(result[best_val_acc,14]) #{0:'l1', 1:'l2', 2:'en', 4:il2, 5:'ISTA'}#ISTA: iterative soft thresholding (proximal gradient)
    reg_type = reg_type1[reg_type_idx]    
    class_weighting_idx = int(result[best_val_acc,15]) #{0:'average', 1:'balanced'}
    class_weighting = class_weighting1[class_weighting_idx]
    upsample1_idx  = int(result[best_val_acc,16]) #{0:'False', 1:'True'}
    upsample1 = upsample1a[upsample1_idx]
    PV_scheme_idx  = int(result[best_val_acc,19]) #{0:'kmeans', 1:'renyi'}
    PV_scheme = PV_scheme1[PV_scheme_idx]
    n_components = int(result[best_val_acc,20]) 
    do_pca_in_selection_idx  = int(result[best_val_acc,21]) #{0:'False', 1:'True'}
    do_pca_in_selection = do_pca_in_selection1[do_pca_in_selection_idx]
    iterMax2 = int(result[best_val_acc,22]) #max number of iterations for outer SGD loop
    levels = int(result[best_val_acc,24])
    #%%
    #testing the dataset
    print('Training and testing')    
    result1=np.zeros((1,25))
    t0=time()
    mcm =  LSMCM_classifier(C1 = C1, C2 = C2, C3 = C3, gamma = gamma, kernel_type = kernel_type, batch_sz = batch_sz, iterMax1 = iterMax1, iterMax2 =iterMax2, eta = eta, 
                          tol = tol, update_type = update_type, reg_type = reg_type, class_weighting = class_weighting,PV_scheme = PV_scheme, 
                          n_components = n_components, do_pca_in_selection = do_pca_in_selection,do_upsample = upsample1 )
    W, f,iters,fvals,subset,me,std = mcm.fit(xTrain,yTrain)
    t1=time()
    time_elapsed=t1-t0
    #predict on train and test set
    train_pred, train_scores = mcm.predict(xTrain, xTrain, W,  me, std, subset)
    test_pred, scores_pred = mcm.predict(xTest, xTrain, W,  me, std, subset)

    train_acc=mcm.accuracy_classifier(yTrain,train_pred)
    test_acc=mcm.accuracy_classifier(yTest,test_pred)
    train_f1= f1_score(yTrain, train_pred, average='weighted') 
    test_f1 =f1_score(yTest, test_pred, average='weighted') 
    
    print ('C1=%0.3f, C2=%0.3f -> train acc= %0.2f, test acc=%0.2f'%(C1,C2,train_acc,test_acc))
    #save the result
    non_zero_weights=0
    total_weights = 0
    if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):
        W2=np.zeros(W.shape)
        W2[W!=0.0]=1
        W2 =np.sum(W2,axis=1)
        non_zero_weights+=np.sum(W2 != 0)                            
        total_weights += np.sum(W!=0)
    else:
        non_zero_weights+=np.sum(W!=0)
        total_weights = non_zero_weights
    
    result1=np.append(result1,np.array([[train_acc, test_acc, time_elapsed, non_zero_weights, C1, C2, C3, kernel_type_idx, gamma,
                                         batch_sz,iterMax1, eta, tol, update_type_idx, reg_type_idx, class_weighting_idx, 
                                         upsample1_idx ,train_f1, test_f1, PV_scheme_idx, n_components, do_pca_in_selection_idx,
                                         iterMax2,total_weights,levels]]),axis=0)
    
    
    result1=result1[1:,]
    #print result
    result_pd1=pd.DataFrame(result1,index=range(0,result1.shape[0]),columns=['0_train_acc', '1_val_acc', '2_time_elapsed', '3_non_zero_weights', '4_C1', '5_C2'
                                                                           , '6_C3', '7_kernel_type_idx', '8_gamma', '9_batch_sz','10_iterMax1', '11_eta', '12_tol', 
                                                                           '13_update_type_idx', '14_reg_type_idx', '15_class_weighting_idx', 
                                                                '16_upsample1_idx' ,'17_train_f1', '18_val_f1', '19_PV_scheme_idx', '20_n_components', '21_do_pca_in_selection_idx',
                                                                '22_iterMax2','23_total_weights','24_levels'])
    result_pd1.to_csv(path1+"/results/Test_results_%d"%(dataset_name)+".csv")
