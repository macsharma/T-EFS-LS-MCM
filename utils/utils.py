# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:19:50 2018

@author: mayank
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import linear_kernel,rbf_kernel,manhattan_distances,polynomial_kernel,sigmoid_kernel,cosine_similarity,laplacian_kernel,paired_euclidean_distances,pairwise_distances
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.utils import resample
from numpy.matlib import repmat
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from numpy.linalg import eigh
from sklearn.preprocessing import OneHotEncoder
from sparse import COO
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse import issparse
from scipy.sparse import hstack
#%%

class utils:
#    def __init__(self):
#        return None
    
    def add_bias(self,xTrain):
        """
        Adds bias to the data
        Parameters:
        -----------
        xTrain: 2D numpy ndarray/csr_matrix of shape (n_samples, n_features)
        
        Returns:
        --------
        xTrain: 2D numpy ndarray/csr_matrix of shape (n_samples, n_features + 1)
        """
        N = xTrain.shape[0]
        if(xTrain.size!=0):
            if(issparse(xTrain)==True):
                xTrain  = csr_matrix(hstack([xTrain,np.ones((N,1))]))
            else:
                xTrain=np.hstack((xTrain,np.ones((N,1))))
        return xTrain
    
    def logsig(self,x):
        return 1 / (1 + np.exp(-x))
    
    def saturate_fcn1(self,x,a = 2):
        y = np.zeros(x.shape) 
        idx1 = (x <= a)*(x >=-a)
        idx2 = x > a
        idx3 = x < -a
        y[idx1] = x[idx1]/(2*a) + 1.0/2.0
        y[idx2] = 1
        y[idx3] = 0
        return y
    
    def standardize(self,xTrain,centering):
        """
        Transform the data so that each column has zero mean and unit standard deviation
        
        Parameters:
        -----------
        xTrain: 2D numpy ndarray of shape (n_samples, n_features)
        centering: bool,
            whether to perform standardization, 
        if False, it returns me =  np.zeros((xTrain.shape[1],))
        and  std_dev = np.ones((xTrain.shape[1],))
        
        Returns:
        --------
        xTrain: 2D numpy ndarray of shape (n_samples, n_features)
        me:  mean of the columns
        std_dev: standard deviation of the columns
        """
        if(centering == True):
            me=np.mean(xTrain,axis=0)
            std_dev=np.std(xTrain,axis=0)
        else:
            me = np.zeros((xTrain.shape[1],))
            std_dev = np.ones((xTrain.shape[1],))
        #remove columns with zero std
        idx=(std_dev!=0.0)
#        print(idx.shape)
        xTrain[:,idx]=(xTrain[:,idx]-me[idx])/std_dev[idx]
        return xTrain,me,std_dev
    
    def divide_into_batches_stratified(self,yTrain,batch_sz):
        """
        Divides the data into batches such that each batch contains similar proportion of labels in it
        Parameters:
        ----------
        yTrain: np.ndarray labels for the datset of shape (n_samples, )
        
        Returns:
        --------
        idx_batches: list
            index of yTrain in each batch
        sample_weights: np.ndarray of size (n_samples,)
            weights for each sample in batch = 1/#class_j
        num_batches: int
            number of batches formed
        """
        #data should be of the form samples X features
        N=yTrain.shape[0]    
        num_batches=int(np.ceil(N/batch_sz))
        sample_weights=list()
        numClasses=np.unique(yTrain).size
        idx_batches=list()
    
        skf=StratifiedKFold(n_splits=num_batches, random_state=1, shuffle=True)
        j=0
        for train_index, test_index in skf.split(np.zeros(N), yTrain):
            idx_batches.append(test_index)
            class_weights=np.zeros((numClasses,))
            sample_weights1=np.zeros((test_index.shape[0],))
            temp=yTrain[test_index,]
            for i in range(numClasses):
                idx1=(temp==i)
                class_weights[i]=1.0/(np.sum(idx1)+1e-09)#/idx.shape[0]
                sample_weights1[idx1]=class_weights[i]            
            sample_weights.append(sample_weights1)

            j+=1
        return idx_batches,sample_weights,num_batches
    
    def margin_kernel(self, X1, kernel_type = 'linear', gamma =1.0):
        """
        Forms the kernel matrix using the samples X1
        Parameters:
        ----------
        X1: np.ndarray
            data (n_samples,n_features) to form a kernel of shape (n_samples,n_samples)
        kernel_type : str
            type of kernel to be used
        gamma: float
            kernel parameter
        Returns:
        -------
        X: np.ndarray
            the kernel of shape (n_samples,n_samples)
        """
        
        if(kernel_type == 'linear'):
            X = linear_kernel(X1,X1)
        elif(kernel_type == 'rbf'):
            X = rbf_kernel(X1,X1,gamma) 
        elif(kernel_type == 'tanh'):
            X = sigmoid_kernel(X1,X1,-gamma) 
        elif(kernel_type == 'sin'):
#            X = np.sin(gamma*manhattan_distances(X1,X1))
            X = np.sin(gamma*pairwise_distances(X1,X1)**2)
        elif(kernel_type =='TL1'):                
            X = np.maximum(0,gamma - manhattan_distances(X1,X1)) 
        else:
            print('no kernel_type, returning None')
            return None
        return X
    
    
    def kernel_transform(self, X1, X2 = None, kernel_type = 'linear_primal', n_components = 100, gamma = 1.0):
        """
        Forms the kernel matrix using the samples X1
        Parameters:
        ----------
        X1: np.ndarray
            data (n_samples1,n_features) to form a kernel of shape (n_samples1,n_samples1)
        X2: np.ndarray
            data (n_samples2,n_features) to form a kernel of shape (n_samples1,n_samples2)
        kernel_type : str
            type of kernel to be used
        gamma: float
            kernel parameter
        Returns:
        -------
        X: np.ndarray
            the kernel of shape (n_samples,n_samples)
        """
        if(kernel_type == 'linear'):
            X = linear_kernel(X1,X2)
        elif(kernel_type == 'rbf'):  
            X = rbf_kernel(X1,X2,gamma) 
        elif(kernel_type == 'tanh'):
            X = sigmoid_kernel(X1,X2,-gamma) 
        elif(kernel_type == 'sin'):
#            X = np.sin(gamma*manhattan_distances(X1,X2))
            X = np.sin(gamma*pairwise_distances(X1,X2)**2)
        elif(kernel_type =='TL1'):                
            X = np.maximum(0,gamma - manhattan_distances(X1,X2)) 
        elif(kernel_type == 'rff_primal'):
            rbf_feature = RBFSampler(gamma=gamma, random_state=1, n_components = n_components)
            X = rbf_feature.fit_transform(X1)
        elif(kernel_type == 'nystrom_primal'):
            #cannot have n_components more than n_samples1
            if(n_components > X1.shape[0]):
                raise ValueError('n_samples should be greater than n_components')
            rbf_feature = Nystroem(gamma=gamma, random_state=1, n_components = n_components)
            X = rbf_feature.fit_transform(X1)
        elif(kernel_type == 'linear_primal'):                
            X = X1
        else:
            print('No kernel_type passed: using linear primal solver')
            X = X1
        return X
    
    def generate_samples(self,X_orig,old_imbalance_ratio,new_imbalance_ratio):
        """
        Generates samples based on new imbalance ratio, such that new imbalanced ratio is achieved
        Parameters:
        ----------
        X_orig: np.array (n_samples , n_features)
            data matrix 
        old_imbalance_ratio: float
            old imbalance ratio in the samples
        new_imbalance_ratio: float
            new imbalance ratio in the samples
        
        Returns:
        -------
        X_orig: np.array (n_samples , n_features)
            data matrix 
        X1: 2D np.array
            newly generated samples of shape (int((new_imbalance_ratio/old_imbalance_ratio)*n_samples - n_samples), n_features )
        """
    
        N=X_orig.shape[0]
        M=X_orig.shape[1]
        neighbors_thresh=10
        if (new_imbalance_ratio < old_imbalance_ratio):
            raise ValueError('new ratio should be greater than old ratio')
        new_samples=int((new_imbalance_ratio/old_imbalance_ratio)*N - N)       
        #each point must generate these many samples 
        new_samples_per_point_orig=new_imbalance_ratio/old_imbalance_ratio - 1
        new_samples_per_point=int(new_imbalance_ratio/old_imbalance_ratio - 1)
        #check if the number of samples each point has to generate is > 1
        X1=np.zeros((0,M))   
            
        if(new_samples_per_point_orig>0 and new_samples_per_point_orig<=1):
            idx_samples=resample(np.arange(0,N), n_samples=int(N*new_samples_per_point_orig), random_state=1,replace=False)
            X=X_orig[idx_samples,]
            new_samples_per_point=1
            N=X.shape[0]
        else:
            X=X_orig
            
        if(N==1):
            X1=repmat(X,new_samples,1)            
        elif(N>1):        
            if(N<=neighbors_thresh):
                n_neighbors=int(N/2)
            else:
                n_neighbors=neighbors_thresh
                        
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)                
            for i in range(N):
                #for each point find its n_neighbors nearest neighbors
                inds=nbrs.kneighbors(X[i,:].reshape(1,-1), n_neighbors, return_distance=False)
                temp_data=X[inds[0],:]    
                std=np.std(temp_data,axis=0)
                me=np.mean(temp_data,axis=0)
                np.random.seed(i)                
                x_temp=me + std*np.random.randn(new_samples_per_point,M)  
                X1=np.append(X1,x_temp,axis=0)
            
        return X_orig, X1      
    
    def upsample(self,X,Y,new_imbalance_ratio): 
        """
        Upsamples the data based on label array, for classification only
        Parameters:
        ----------
        X: np.array (n_samples, n_features)
            2D data matrix
        Y: np.array (n_samples, )
            label array, takes values between [0, numClasses-1]
        new_imbalance_ratio: float
            new imbalance ratio in the data, takes values between [0.5,1]
            
        Returns:
        -------
        X3: np.array (n_samples1, n_features)
            new balanced 2D data matrix
        Y3: np.array (n_samples1, )
            new balanced label array
        """
        #xTrain: samples X features
        #yTrain : samples,
        #for classification only
        numClasses=np.unique(Y).size
        class_samples=np.zeros((numClasses,))
        X3=np.zeros((0,X.shape[1]))
        Y3=np.zeros((0,)) 
            
        #first find the samples per class per class
        for i in range(numClasses):
            idx1=(Y==i)
            class_samples[i]=np.sum(idx1)
            
        max_samples=np.max(class_samples)
    #    new_imbalance_ratio=0.5  
#        if(upsample_type==1):
        old_imbalance_ratio_thresh=0.5
#        else:
#            old_imbalance_ratio_thresh=1
            
        for i in range(numClasses):
            idx1=(Y==i)
            old_imbalance_ratio=class_samples[i]/max_samples
            X1=X[idx1,:]
            Y1=Y[idx1,]              
    
            if(idx1.size==1):
                X1=np.reshape(X1,(1,X.shape[1]))
                
            if(old_imbalance_ratio<=old_imbalance_ratio_thresh and class_samples[i]!=0):               
                X1,X2=self.generate_samples(X1,old_imbalance_ratio,new_imbalance_ratio)
                new_samples=X2.shape[0]
                Y2=np.ones((new_samples,))
                Y2=Y2*Y1[0,]
                    
                #append original and generated samples
                X3=np.append(X3,X1,axis=0)
                X3=np.append(X3,X2,axis=0)
                
                Y3=np.append(Y3,Y1,axis=0)
                Y3=np.append(Y3,Y2,axis=0)            
            else:
                #append original samples only
                X3=np.append(X3,X1,axis=0)
                Y3=np.append(Y3,Y1,axis=0)
                
        Y3=np.array(Y3,dtype=np.int32)  
        return X3,Y3
    
    def kmeans_select(self,X,represent_points,do_pca=False):
        """
        Takes in data and number of prototype vectors and returns the indices of the prototype vectors.
        The prototype vectors are selected based on the farthest distance from the kmeans centers
        Parameters
        ----------
        X: np.ndarray
            shape = n_samples, n_features
        represent_points: int
            number of prototype vectors to return
        do_pca: boolean
            whether to perform incremental pca for dimensionality reduction before selecting prototype vectors
            
        Returns
        -------
        sv: list
            list of the prototype vector indices from the data array given by X
        """
#        do_pca = self.do_pca_in_selection
        N = X.shape[0]
        if(do_pca == True):
            if(X.shape[1]>50):
                n_components = 50
                ipca = IncrementalPCA(n_components=n_components, batch_size=np.min([128,X.shape[0]]))
                X = ipca.fit_transform(X)
    
        kmeans = MiniBatchKMeans(n_clusters=represent_points, batch_size=np.min([128,X.shape[0]]),random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        sv= []
        unique_labels = np.unique(labels).size 
        all_ind = np.arange(N)
        for j in range(unique_labels):
            X1 = X[labels == j,:]
            all_ind_temp = all_ind[labels==j]
            tempK = pairwise_distances(X1,np.reshape(centers[j,:],(1,X1.shape[1])))**2
            inds = np.argmax(tempK,axis=0)
            sv.append(all_ind_temp[inds[0]])
    
        return sv
    def renyi_select(self,X,represent_points,do_pca=False):
        """
        Takes in data and number of prototype vectors and returns the indices of the prototype vectors.
        The prototype vectors are selected based on maximization of quadratic renyi entropy, which can be 
        written in terms of log sum exp which is a tightly bounded by max operator. Now for rbf kernel,
        the max_{ij}(-\|x_i-x_j\|^2) is equivalent to min_{ij}(\|x_i-x_j\|^2).
        Parameters
        ----------
        X: np.ndarray
            shape = n_samples, n_features
        represent_points: int
            number of prototype vectors to return
        do_pca: boolean
            whether to perform incremental pca for dimensionality reduction before selecting prototype vectors
            
        Returns
        -------
        sv: list
            list of the prototype vector indices from the data array given by X
        """
#        do_pca = self.do_pca_in_selection
        N= X.shape[0]    
        capacity=represent_points
        selectionset=set([])
        set_full=set(list(range(N)))
        np.random.seed(1)
        if(len(selectionset)==0):
            selectionset = np.random.permutation(N)
            sv = list(selectionset)[0:capacity]        
        else:
            extrainputs = represent_points - len(selectionset)
            leftindices =list(set_full.difference(selectionset))
            info = np.random.permutation(len(leftindices))
            info = info[1:extrainputs]
            sv = selectionset.append(leftindices[info])
    
        if(do_pca == True):
            if(X.shape[1]>50): #takes more time
                n_components = 50
                ipca = IncrementalPCA(n_components=n_components, batch_size=np.min([128,X.shape[0]]))
                X = ipca.fit_transform(X)
            
        svX = X[sv,:]
        
        min_info = np.zeros((capacity,2))

        KsV = pairwise_distances(svX,svX)**2 #this is fast
        
        KsV[KsV==0] = np.inf
        min_info[:,1] = np.min(KsV,axis=1)
        min_info[:,0] = np.arange(capacity)
        minimum = np.min(min_info[:,1])
        counter = 0
        
        for i in range(N):
        #    find for which data the value is minimum
            replace = np.argmin(min_info[:,1])
            ids = int(min_info[min_info[:,0]==replace,0])
            #Subtract from totalcrit once for row 
            tempminimum = minimum - min_info[ids,1] 
            #Try to evaluate kernel function 
            
            tempsvX = np.zeros(svX.shape)
            tempsvX[:] = svX[:]
            inputX = X[i,:]
            tempsvX[replace,:] = inputX 
            tempK = pairwise_distances(tempsvX,np.reshape(inputX,(1,X.shape[1])))**2 #this is fast
            tempK[tempK==0] = np.inf
            distance_eval = np.min(tempK)
            tempminimum = tempminimum + distance_eval 
            if (minimum < tempminimum):
                minimum = tempminimum
                min_info[ids,1] = distance_eval
                svX[:] = tempsvX[:]
                sv[ids] = i
                counter +=1
        return sv
    
    def subset_selection(self,X,Y, n_components , PV_scheme , problem_type,do_pca=False):
        """
        Takes in data matrix and label matrix and generates the subset (list) of shape n_components based on the problem type
        (classification or regression), prototype vector (PV) selection scheme
        
        Parameters:
        ----------
        X: np.array (n_samples, n_features)
            data matrix
        Y: np.array (n_samples)
            label matrix (continuous or discrete)
        PV_scheme: str
            prototype vector selection scheme  ('renyi' or 'kmeans')
        problem_type: str
            type of the problem ('classification' or 'regression')
        Returns:
        --------
        subset: list
            the index of the prototype vectors selected
        """
        
        N = X.shape[0]        
        if(problem_type == 'regression'):   
            if(PV_scheme == 'renyi'):
                subset = self.renyi_select(X,n_components,do_pca)
            elif(PV_scheme == 'kmeans'):
                subset = self.kmeans_select(X,n_components,do_pca)
            else:
                raise ValueError('Select PV_scheme between renyi and kmeans')
        else:
            numClasses = np.unique(Y).size
            all_samples = np.arange(N)
            subset=[]
            subset_per_class = np.zeros((numClasses,))
            class_dist = np.zeros((numClasses,))
            for i in range(numClasses):
                class_dist[i] = np.sum(Y == i)
                subset_per_class[i] = int(np.ceil((class_dist[i]/N)*n_components))                
            for i in range(numClasses):
                xTrain = X[Y == i,]
                samples_in_class = all_samples[Y == i]
                if(PV_scheme == 'renyi'):
                    subset1 = self.renyi_select(xTrain,int(subset_per_class[i]),do_pca)
                elif(PV_scheme == 'kmeans'):
                    subset1 = self.kmeans_select(xTrain,int(subset_per_class[i]),do_pca)
                else:
                    raise ValueError('Select PV_scheme between renyi and kmeans')

                temp=list(samples_in_class[subset1])
                subset.extend(temp)
                
        return subset
    
    def matrix_decomposition(self, X):
        """
        Finds the matrices consisting of positive and negative parts of kernel matrix X
        Parameters:
        ----------
        X: n_samples X n_samples

        Returns:
        --------
        K_plus: kernel corresponding to +ve part
        K_minus: kernel corresponding to -ve part            
        """
        [D,U]=eigh(X)
        U_plus = U[:,D>0.0]
        U_minus = U[:,D<=0.0]
        D_plus = np.diag(D[D>0.0])
        D_minus = np.diag(D[D<=0.0])
        K_plus = np.dot(np.dot(U_plus,D_plus),U_plus.T)
        K_minus = -np.dot(np.dot(U_minus,D_minus),U_minus.T)
        return K_plus, K_minus
    
    def zero_one_normalization(self,x):
        """
        perform 0-1 normalization on the data x
        Parameters:
        ----------
        x: 2d np.array (n_samples,n_features)
            data matrix
        Returns:
        --------
        x: 2d np.array (n_samples,n_features)
            normalized data matrix
        """
        return (x-np.min(x,axis = 0))/(np.max(x,axis = 0)-np.min(x,axis = 0))

    def quantize(self,x,levels):
        """
        perform the quantization of a matrix based on the number of levels given.
        Data should be zero-one normalized before quantization can be applied
        Parameters:
        ----------
        x: np.ndarray 
            data matrix
        level: int
            number of levels in quantization
        Returns:
        -------
        q: np.ndarray
            quantized data matrix
        """
        if(np.sum(x<0) > 0):
            raise ValueError('data is not zero-one normalized')
        q = np.zeros(x.shape,dtype = np.int8)
        for i in range(1,levels):
            q[x > 1.0*i/levels] +=1
        return q
    
    def dequantize(self,x,levels):
        """
        perform the dequantization of a matrix based on the number of levels given.
        data should be quantized with levels = levels
        Parameters:
        ----------
        x: np.ndarray 
            quantized data matrix 
        level: int
            number of levels in quantization
        Returns:
        -------
        q: np.ndarray
            dequantized data matrix
        """
        if(levels ==1):
            raise ValueError('levels should be greater than 1!')
        x = x/(levels-1)
        return x
        
    def labels2onehot(self,labels):
        """
        performs one hot encoding on labels in range (0,levels)
        Parameters:
        -----------
        labels: np.array (n_samples,)
            labels for each sample
        Returns:
        -------
        onehotvec: CSR matrix (n_samples,levels)
            one hot vector of labels
        """
        levels = np.unique(labels).size
        N = labels.shape[0]
        labels = labels.reshape(N,-1)
        enc = OneHotEncoder(n_values= levels)
        onehotvec = enc.fit_transform(labels)
        return onehotvec
    
    def onehot(self,arr,levels,issparse=False):
        """
        performs one hot encoding of the quantized 2D data matrix into levels specified by user
        Parameters:
        ----------
        arr: np.array (n_samples,n_features)
            data matrix
        levels: int
            number of one hot levels
        issparse: bool
            whether to output a sparse COO matrix, requires 'sparse' package
        Returns:
        -------
        arr: np.ndarray or sparse.COO matrix
            one hot encoded matrix
        """
        N,M = arr.shape
        arr = arr.reshape(N,-1)
        enc = OneHotEncoder(n_values=levels,sparse=False,dtype = np.int8)
        arr = enc.fit_transform(arr)
        arr = arr.reshape(N,M,levels)
        if(issparse ==True):
            arr = COO.from_numpy(arr)
        return arr
   
    
    def onehot_minibatch(self,arr,levels):
        """
        performs one hot encoding of the quantized 2D data matrix into levels specified by user
        Parameters:
        ----------
        arr: np.array (n_samples,n_features)
            data matrix
        levels: int
            number of one hot levels
        Returns:
        -------
        arr: csr matrix
            one hot encoded matrix
        """
        N,M = arr.shape
        arr = arr.reshape(N,-1)
        enc = OneHotEncoder(n_values=levels,sparse=False,dtype = np.int8)
        arr2 = lil_matrix((N,M*levels),dtype = np.int8)
        batch_sz = np.min([10000,N])
        num_batches=int(np.ceil(N/batch_sz))
        for j in range(num_batches):
            if(j==num_batches-1):
                remainder= N-batch_sz*(num_batches-1)
                test_idx=np.array(range(0,remainder),dtype = np.int32)+ j*batch_sz
            else:        
                test_idx=np.array(range(0,batch_sz),dtype = np.int32)+ j*batch_sz
            arr1 = enc.fit_transform(arr[test_idx,])
            arr2[test_idx,:] = arr1
        arr2 = csr_matrix(arr2)
        return arr2

    def tempcode_minibatch(self,arr,levels):
        """
        performs thermometer encoding of the one hot encoded 2D data matrix into levels specified by user
        Parameters:
        ----------
        arr: np.array (n_samples,n_features)
            data matrix
        levels: int
            number of thermometer encoding levels
            levels should be equal to the levels of arr
        Returns:
        -------
        arr: csr matrix
            one hot encoded matrix
        """
        N, M1 =arr.shape
        tempcode1 = lil_matrix((N,M1),dtype = np.int8)
        batch_sz = np.min([10000,N])
        num_batches=int(np.ceil(N/batch_sz))    
        for j in range(num_batches):
            if(j==num_batches-1):
                remainder= N-batch_sz*(num_batches-1)
                test_idx=np.array(range(0,remainder),dtype = np.int32)+ j*batch_sz
            else:        
                test_idx=np.array(range(0,batch_sz),dtype = np.int32)+ j*batch_sz
            arr1 = arr[test_idx,:].toarray()
            N1 = arr1.shape[0]
            arr1 = np.reshape(arr1,(N1,int(M1/levels),levels))
            tempcode = np.zeros(arr1.shape,dtype =np.int8)
            for i in range(levels-1):
                tempcode[:,:,i+1] = np.sum(arr1[:,:,:i+1],axis=2)
            idx_zero = tempcode ==0
            idx_one = tempcode ==1
            tempcode[idx_zero ] =1
            tempcode[idx_one] = 0
            tempcode = np.reshape(tempcode,(N1,M1))
            tempcode1[test_idx,] = tempcode
        tempcode1 = csr_matrix(tempcode1)
        return tempcode1
    
    def tempcode_ICLR2018(self,arr,levels,issparse=False):
        """
        performs thermometer encoding of the one hot encoded 3D data matrix into levels specified by user, as in ICLR 2018 paper
        Parameters:
        ----------
        arr: np.array (n_samples,n_features)
            data matrix
        levels: int
            number of thermometer encoding levels
            levels should be equal to the levels of arr
        issparse: bool
            whether to output a sparse COO matrix, requires 'sparse' package
        Returns:
        -------
        arr: np.ndarray or sparse.COO matrix
            one hot encoded matrix
        """
        if(levels != arr.shape[2]):
            raise ValueError('Levels specified by the user does not match the one hot encoded input')
        tempcode = np.zeros(arr.shape,dtype = np.int8)
        for i in range(levels):
            tempcode[:,:,i] = np.sum(arr[:,:,:i+1],axis=2)
        if(issparse ==True):
            tempcode = COO.from_numpy(tempcode)
        return tempcode 
    
    def select_(self, xTest, xTrain, kernel_type, subset, idx_features, idx_samples):
        """
        selects samples and features based on indices of the data
        
        Parameters:
        ----------
        xTest: np.array (n_samples,n_features)
            test data
        xTrain: np.array (n_samples,n_features)
            train data
        kernel_type: 'str'    
            type of kernel: linear,rbf,sin,tanh,TL1,linear_primal,rff_primal,nystrom_primal
        subset: list
            subset of n_features    
        idx_features: np.array
            array of indices of features
        idx_samples: np.array
            array of indices of samples
        
        Returns:
        -------
        X1: np.array
            subset of xTest
        X2: np.array
            subset of xTrain (None if kernel type is not linear,rbf,sin,tanh,TL1)
        """
        non_linear_kernels = ['linear','rbf','sin','tanh','TL1']
        if(kernel_type in non_linear_kernels):
            if(len(subset) == 0):
                raise ValueError('Subset cannot be of zero length!')
            X2 = xTrain[idx_samples,:]
            X2 = X2[:,idx_features] 
            X2 = X2[subset,]
            X1 = xTest[:,idx_features]
        else:
            X1 = xTest[:,idx_features]
            X2 = None
        return X1, X2
    
    def normalize_(self,xTrain, me, std):
        """
        normalizes the data to have mean = me and standard_deviation = std
        Parameters:
        -----------
        xTrain: 2D np.array  (n_samples,n_features)
            data matrix
        me: np.array
            mean of samples (n_features,)
        std: np.array
            standard deviations of samples (n_features,)
            
        Returns:
        -------    
        xTrain: 2D np.array  (n_samples,n_features)
            normalized data matrix
        """
        idx = (std!=0.0)
        xTrain[:,idx] = (xTrain[:,idx]-me[idx])/std[idx]
        return xTrain
    
