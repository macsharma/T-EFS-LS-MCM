# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:36:57 2018
Quantized LSMCM
@author: mayank
"""

from utils.utils import utils
import numpy as np
from scipy.sparse import issparse
from utils.CyclicLR import CyclicLR
class LSMCM_Q_classifier:
    def __init__(self,C1 = 0.0, C2 = 1e-03, C3 = 1e-03, gamma =1e-02, kernel_type = 'linear_primal', batch_sz =128, iterMax1 =1000, iterMax2 =1, eta = 0.01, 
             tol = 1e-04, update_type = 'adam', reg_type = 'l1', class_weighting = 'balanced',PV_scheme = 'renyi', 
             n_components = 100, do_pca_in_selection = False, do_upsample = False,levels = 10, compress_type = 'saturate'):
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.gamma = gamma
        self.kernel_type = kernel_type
        self.batch_sz = batch_sz
        self.iterMax1 = iterMax1
        self.iterMax2 = iterMax2
        self.eta = eta
        self.tol = tol
        self.update_type = update_type
        self.reg_type = reg_type
        self.class_weighting =  class_weighting
        self.PV_scheme = PV_scheme
        self.n_components = n_components
        self.do_pca_in_selection = do_pca_in_selection
        self.do_upsample = False
        self.levels = levels
        self.compress_type = compress_type
        
    def inner_opt(self, X, Y, data1):
        """
        It performs optimization using the MCM classifier
        Parameters:
        X: np.array (n_samples,n_features)
            data matrix
        Y: np.array (n_samples,)
            label matrix
        data1: np.array (n_samples1, n_features)
            data for ['linear','rbf','sin','tanh','TL1'] forming kernels
        Returns:
        --------
        W : np.array
            Weights learnt
        f: np.array
            best value of f            
        iters: int
            total number of iterations run
        fvals: np.float
            function values for each itertion
        """
        gamma = self.gamma
        kernel_type = self.kernel_type
        iterMax2 = self.iterMax2
        iterMax1 = self.iterMax1
        tol = self.tol
        util = utils()
        non_linear_kernels = ['linear','rbf','sin','tanh','TL1']
        #if data1 = None implies there is no kernel computation, i.e., there is only primal solvers applicable
        if(data1 is not None):
            if(self.reg_type == 'M'):                
                K = util.margin_kernel(X1 = data1, kernel_type = kernel_type, gamma = gamma)
                
                if(kernel_type in non_linear_kernels):
                    K_plus, K_minus = util.matrix_decomposition(K)
                    W_prev,f,iters,fvals = self.train(X, Y, K_plus = K_plus, K_minus = None, W = None) 
                    if(kernel_type == 'linear' or kernel_type == 'rbf'):
                        #for mercer kernels no need to train for outer loop
                        print('Returning for mercer kernels')
                        return W_prev,f,iters,fvals
                    else:
                        print('Solving for non - mercer kernels')
                        #for non mercer kernels, train for outer loop with initial point as W_prev
                        W_best = np.zeros(W_prev.shape)
                        W_best[:] = W_prev[:]
                        f_best = np.inf
                        iter_best = 0
                        fvals = np.zeros((iterMax1+1,))
                        iters = 0
                        fvals[iters] = f
                        rel_error = 1.0
                        print('iters =%d, f_outer = %0.9f'%(iters,f))
                        train_acc_best = 0.0
                        while(iters < iterMax2 and rel_error > tol):
                            iters = iters + 1 
                            W,f,iters1,fvals1 = self.train(X, Y, K_plus = K_plus, K_minus = K_minus, W = W_prev) 
                            rel_error = np.abs(np.linalg.norm(W - W_prev,'fro'))/(np.linalg.norm(W_prev,'fro') + 1e-08)
                            W_prev[:] = W[:]
                            X1=util.add_bias(X)
                            scores=X1.dot(W)
                            pred = np.argmax(scores,axis=1)
                            train_acc = np.sum(pred == Y)*100/Y.shape[0]
                            if(train_acc>=train_acc_best):
                                train_acc_best=train_acc-0.5
                                W_best[:]=W[:]  
                                f_best = f
                                iter_best = iters
                            else:
                                break
                        fvals[iters] = -1
                        return W_best,f_best,iter_best,fvals
                else:
                    #if kernel_type is wrong
                    raise ValueError('Please choose a kernel_type from linear, rbf, sin, tanh or TL1 for reg_type = M to work')
            else:
                W,f,iters,fvals = self.train(X, Y, K_plus = None, K_minus = None, W = None)            
        else:
            #i.e., data1 is None -> we are using primal solvers with either l1, l2, ISTA or elastic net penalty
            if(self.reg_type == 'M'): 
                raise ValueError('Please choose a kernel_type from linear, rbf, sin, tanh or TL1 for reg_type = M to work')
            else:
                W,f,iters,fvals = self.train(X,Y, K_plus = None, K_minus = None, W = None)
                return W,f,iters,fvals                   
        return W,f,iters,fvals    
    
    def fit(self,xTrain,yTrain):
        """
        fits a classifier to the data
        Parameters:
        -----------
        xTrain: np.array (2D array) (n_samples,n_features)
            data matrix
        yTrain: np.array (n_samples,)
            label array, labels are in range [0,numClasses-1]
        Returns:
        -------
        W: np.array (n_features, numClasses)
            weight matrix
        me: np.array (n_features,) 
            mean of train features
        std: np.array (n_features,)
            standard deviation of train features
        subset: list
            list of selected subset for kernel_type = linear, sin, tanh, TL1, rbf and empty list otherwise
        """
        util = utils()
        gamma = self.gamma
        kernel_type = self.kernel_type
        n_components = self.n_components
        PV_scheme = self.PV_scheme
        do_pca = self.do_pca_in_selection
        levels = self.levels
        compress_type = self.compress_type
        
        non_linear_kernels = ['linear','rbf','sin','tanh','TL1']
        if(kernel_type in non_linear_kernels):
            subset = util.subset_selection(xTrain,yTrain, n_components, PV_scheme, 'classification',do_pca)
            data1 = xTrain[subset,]
        else:
            subset = []
            data1 = None
        if(compress_type == 'zero_one'):
            xTrain = util.zero_one_normalization(xTrain) 
        elif(compress_type == 'sigmoid'):
            xTrain = util.logsig(xTrain)
        elif(compress_type == 'saturate'):
            xTrain = util.saturate_fcn1(xTrain)
        else:
            raise ValueError('wrong compress_type selected!')
            
        xTrain = util.quantize(xTrain,levels)
        xTrain = util.dequantize(xTrain,levels)
#        xTrain = util.onehot_minibatch(xTrain,levels)
#        xTrain = util.tempcode_minibatch(xTrain,levels)
        if(data1 is not None):
            if(compress_type == 'zero_one'):
                data1 = util.zero_one_normalization(data1)
            elif(compress_type == 'sigmoid'):
                data1 = util.logsig(data1)
            elif(compress_type == 'saturate'):
                data1 = util.saturate_fcn1(data1)
            else:
                raise ValueError('wrong compress_type selected!')

            data1 = util.quantize(data1,levels)
            data1 = util.dequantize(data1,levels)
#            data1 = util.onehot_minibatch(data1,levels)
#            data1 = util.tempcode_minibatch(data1,levels)
            
        xTrain1 = util.kernel_transform(X1 = xTrain, X2 = data1, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
        
        #standardize the dataset
        if(kernel_type != 'linear_primal'):
            centering = True
        else:
            centering = False
        xTrain1, me, std  = util.standardize(xTrain1,centering = centering)
        
        W,f,iters,fvals = self.inner_opt(xTrain1, yTrain, data1)
        return W,f,iters,fvals,subset,me,std
        
    def train(self, xTrain, yTrain, K_plus = None, K_minus = None, W = None):
        """
        Training procedure for MCM classifier
        Parameters:
        -----------
        xTrain: np.array (n_samples,n_features)
            data matrix
        yTrain: np.array (n_samples,)
            label matrix
        K_plus: np.array (n_samples1,n_samples2)
            kernel matrix for positive definite part of matrix
        K_minus: np.array (n_samples1,n_samples2)
            kernel matrix for negative definite part of matrix
        W: np.array (n_features, n_classes)
            This is passed only if we are doing multiple outer loop iterations
        Returns:
        -------
        W_best : np.array (n_features, n_classes)
            Best weights learnt
        f_best: np.array
            best value of f            
        iters_best: int
            total number of iterations run
        fvals: np.float
            function values for each itertion
        """
        #min D(E|w|_1 + (1-E)*0.5*|W|_2^2) + C*\sum_i\sum_(j)|f_j(i)| + \sum_i\sum_(j_\neq y_i)max(0,(1-f_y_i(i) + f_j(i)))
        #setting C = 0 gives us SVM
        # or when using margin term i.e., reg_type = 'M'
        #min D(E|w|_1) + (E)*0.5*\sum_j=1 to numClasses (w_j^T(K+ - K-)w_j) + C*\sum_i\sum_(j)|f_j(i)| + \sum_i\sum_(j_\neq y_i)max(0,(1-f_y_i(i) + f_j(i)))
        #setting C = 0 gives us SVM with margin term
        util = utils()
        if(self.do_upsample==True):            
            xTrain,yTrain=util.upsample(xTrain,yTrain,new_imbalance_ratio=0.5,upsample_type=1)
            
        xTrain=util.add_bias(xTrain)
        
        M=xTrain.shape[1]
        N=xTrain.shape[0]
        numClasses=np.unique(yTrain).size
        verbose = False

        C = self.C1 #for loss function of MCM
        D = self.C2 #for L1 or L2 penalty
        E = self.C3 #for elastic net penalty or margin term
            
        iterMax1 = self.iterMax1
        eta_zero = self.eta
        class_weighting = self.class_weighting
        reg_type = self.reg_type
        update_type = self.update_type
        tol = self.tol
        np.random.seed(1)
        
        if(W is None):
            W=0.001*np.random.randn(M,numClasses)
            W=W/np.max(np.abs(W))
        else:
            W_orig = np.zeros(W.shape)
            W_orig[:] = W[:]
        
        class_weights=np.zeros((numClasses,))
        sample_weights=np.zeros((N,))
        #divide the data into K clusters
    
        for i in range(numClasses):
            idx=(yTrain==i)           
            class_weights[i]=1.0/np.sum(idx)
            sample_weights[idx]=class_weights[i]
                        
        G_clip_threshold = 10000
        W_clip_threshold = 50000
        eta=eta_zero
                       
        scores = xTrain.dot(W) #samples X numClasses
        N = scores.shape[0]
        correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
        mat = (scores.transpose()-correct_scores.transpose()).transpose() 
        mat = mat+1.0
        mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
        
        scores1  = np.zeros(scores.shape)
        scores1[:] = scores[:]
        scores1[range(N),np.array(yTrain,dtype='int32')] = -np.inf
        max_scores = np.max(scores1,axis =1)
        mat1 = 1 - correct_scores + max_scores

        f=0.0
        if(reg_type=='l2'):
            f += D*0.5*np.sum(W**2) 
        if(reg_type=='l1'):
            f += D*np.sum(np.abs(W))
        if(reg_type=='en'):
            f += D*0.5*(1-E)*np.sum(W**2)  +  D*E*np.sum(np.abs(W))
            
            
        if(class_weighting=='average'):
            f1 = C*0.5*np.sum(scores**2) + 0.5*np.sum((mat1)**2)
            f += (1.0/N)*f1 
        else:
            f1 = C*0.5*np.sum((scores**2)*sample_weights[:,None]) + 0.5*np.sum((mat1**2)*sample_weights[:,None])
            f+= (1.0/numClasses)*f1
        
        if(K_minus is not None):
            temp_mat = np.dot(K_minus,W_orig[0:(M-1),])        
        
        for i in range(numClasses):
            #add the term (E/2*numclasses)*lambda^T*K_plus*lambda for margin
            if(K_plus is not None):
                w = W[0:(M-1),i]
                f2 = np.dot(np.dot(K_plus,w),w)
                f+= ((0.5*E)/(numClasses))*f2  
             #the second term in the objective function
            if(K_minus is not None):
                f3 = np.dot(temp_mat[:,i],w)
                f+= -((0.5*E)/(numClasses))*f3
        
        
        iter1=0
        print('iter1=%d, f=%0.3f'%(iter1,f))
                
        f_best=f
        fvals=np.zeros((iterMax1+1,))
        fvals[iter1]=f_best
        W_best=np.zeros(W.shape)
        iter_best=iter1
        f_prev=f_best
        rel_error=1.0
#        f_prev_10iter=f
        
        if(reg_type=='l1' or reg_type =='en' or reg_type == 'M'):
            # from paper: Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
            if(update_type == 'adam' or update_type == 'adagrad' or update_type == 'rmsprop'):
                u = np.zeros(W.shape)
            else:
                u = 0.0
            q=np.zeros(W.shape)
            z=np.zeros(W.shape)
            all_zeros=np.zeros(W.shape)
        
        eta1=eta_zero 
        v=np.zeros(W.shape)
        v_prev=np.zeros(W.shape)    
        vt=np.zeros(W.shape)
        m=np.zeros(W.shape)
        vt=np.zeros(W.shape)
        
        cache=np.zeros(W.shape)
        eps=1e-08
        decay_rate=0.99
        mu1=0.9
        mu=mu1
        beta1 = 0.9
        beta2 = 0.999  
        iter_eval=10 #evaluate after every 10 iterations
        batch_sz = self.batch_sz
        idx_batches, sample_weights_batch, num_batches = util.divide_into_batches_stratified(yTrain,batch_sz)
#        lr_schedule = 'constant', 'decrease', 'triangular', 'triangular2' ,'exp_range'
        lr_schedule = 'triangular2'
        iter2 = 0
        cyclic_lr_schedule = ['triangular', 'triangular2' ,'exp_range']
        if(lr_schedule in cyclic_lr_schedule):
            base_lr = eta
            max_lr = 1.5*base_lr
            step_size=num_batches
            mode=lr_schedule
            gamma=1. 
            scale_fn=None
            scale_mode='cycle'
            clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode=mode,
                             gamma=gamma, scale_fn=scale_fn, scale_mode=scale_mode)
        
        while(iter1<iterMax1 and rel_error>tol):
            iter1=iter1+1            
            for batch_num in range(0,num_batches):
                iter2 += 1
                if(lr_schedule in cyclic_lr_schedule):
                    eta1 = clr.clr(iter2)
                    eta = clr.clr(iter2)
                elif(lr_schedule == 'constant'):
                    eta = eta
                    eta1 = eta1
                elif(lr_schedule == 'decrease'):
                    eta = eta_zero/np.power((iter1),1)
                    eta1 = eta_zero/np.power((iter1),1)
    #                batch_size=batch_sizes[j]
                test_idx=idx_batches[batch_num]
                data=xTrain[test_idx,]
                labels=yTrain[test_idx,] 
                N=labels.shape[0]
                scores=data.dot(W)
                correct_scores=scores[range(N),np.array(labels,dtype='int32')]#label_batches[j] for this line should be in the range [0,numClasses-1]
                mat=(scores.transpose()-correct_scores.transpose()).transpose() 
                mat=mat+1.0
                mat[range(N),np.array(labels,dtype='int32')]=0.0                
                
                scores1  = np.zeros(scores.shape)
                scores1[:] = scores[:]
                scores1[range(N),np.array(labels,dtype='int32')] = -np.inf
                max_scores = np.max(scores1,axis =1)
                max_scores_idx = np.argmax(scores1, axis = 1)
                mat1 = 1 - correct_scores + max_scores                
                
                dscores1 = np.zeros(mat.shape)
                dscores1[range(N),np.array(max_scores_idx,dtype='int32')] = mat1
                row_sum = np.sum(dscores1,axis=1)
                dscores1[range(N),np.array(labels,dtype='int32')] = -row_sum
                
                if(C !=0.0):
                    dscores2 = np.zeros(scores.shape)
                    dscores2[:] = scores[:]
                else:
                    dscores2 = np.zeros(scores.shape)
                    
                dscores1 = 2*dscores1
                dscores2 = 2*dscores2
                if(class_weighting=='average'):
                    if(issparse(data)==True):
                        gradW = data.T.dot((dscores1 + C*dscores2))
                    else:
                        gradW = np.dot((dscores1 + C*dscores2).transpose(),data)
                        gradW = gradW.transpose()
                    gradW = (0.5/N)*gradW
                else:
                    sample_weights_b=sample_weights_batch[batch_num]                
                    if(issparse(data)==True):
                        gradW = (data.multiply(sample_weights_b[:,None])).T.dot((dscores1 + C*dscores2))
                        gradW = np.dot((dscores1 + C*dscores2).transpose(),data*sample_weights_b[:,None])
                    else:
                        gradW = np.dot((dscores1 + C*dscores2).transpose(),data*sample_weights_b[:,None])
                        gradW = gradW.transpose()
                    gradW=(0.5/numClasses)*gradW
                        
                if(np.sum(gradW**2)>G_clip_threshold):#gradient clipping
                    gradW = G_clip_threshold*gradW/np.sum(gradW**2)
                    
                if(update_type=='sgd'):
                    W = W - eta*gradW
                elif(update_type=='momentum'):
                    v = mu * v - eta * gradW # integrate velocity
                    W += v # integrate position
                elif(update_type=='nesterov'):
                    v_prev[:] = v[:] # back this up
                    v = mu * v - eta * gradW # velocity update stays the same
                    W += -mu * v_prev + (1 + mu) * v # position update changes form
                elif(update_type=='adagrad'):
                    cache += gradW**2
                    W += - eta1* gradW / (np.sqrt(cache) + eps)
                elif(update_type=='rmsprop'):
                    cache = decay_rate * cache + (1 - decay_rate) * gradW**2
                    W += - eta1 * gradW / (np.sqrt(cache) + eps)
                elif(update_type=='adam'):
                    m = beta1*m + (1-beta1)*gradW
                    mt = m / (1-beta1**(iter1+1))
                    v = beta2*v + (1-beta2)*(gradW**2)
                    vt = v / (1-beta2**(iter1+1))
                    W += - eta1 * mt / (np.sqrt(vt) + eps)           
                else:
                    W = W - eta*gradW
                    
                if(reg_type == 'M'):
                    gradW1= np.zeros(W.shape)
                    gradW2= np.zeros(W.shape)
                    for i in range(numClasses):
                        w=W[0:(M-1),i]
                        if(K_plus is not None):
                            gradW1[0:(M-1),i]=((E*0.5)/(numClasses))*2*np.dot(K_plus,w)
                        if(K_minus is not None):
                            gradW2[0:(M-1),i]=((E*0.5)/(numClasses))*temp_mat[:,i]
                    if(update_type == 'adam'):
                        W += -(gradW1-gradW2)*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -(gradW1-gradW2)*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -(gradW1-gradW2)*(eta)
                        
                if(reg_type == 'ISTA'):
                    if(update_type == 'adam'):
                        idx_plus =  W > D*(eta1/(np.sqrt(vt) + eps))
                        idx_minus = W < -D*(eta1/(np.sqrt(vt) + eps))
                        idx_zero = np.abs(W) < D*(eta1/(np.sqrt(vt) + eps))
                        W[idx_plus] = W[idx_plus] - D*(eta1/(np.sqrt(vt[idx_plus]) + eps))
                        W[idx_minus] = W[idx_minus] + D*(eta1/(np.sqrt(vt[idx_minus]) + eps))
                        W[idx_zero] = 0.0
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        idx_plus =  W > D*(eta1/(np.sqrt(cache) + eps))
                        idx_minus = W < -D*(eta1/(np.sqrt(cache) + eps))
                        idx_zero = np.abs(W) < D*(eta1/(np.sqrt(cache) + eps))
                        W[idx_plus] = W[idx_plus] - D*(eta1/(np.sqrt(cache[idx_plus]) + eps))
                        W[idx_minus] = W[idx_minus] + D*(eta1/(np.sqrt(cache[idx_minus]) + eps))
                        W[idx_zero] = 0.0
                    else:
                        idx_plus =  W > D*(eta)
                        idx_minus = W < -D*(eta)
                        idx_zero = np.abs(W) < D*(eta)
                        W[idx_plus] = W[idx_plus] - D*(eta)
                        W[idx_minus] = W[idx_minus] + D*(eta)
                        W[idx_zero] = 0.0

                        
                if(reg_type=='l2'):
                    if(update_type == 'adam'):
                        W += -D*W*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -D*W*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -D*W*(eta)  
                
                if(reg_type=='en'):
                    if(update_type == 'adam'):
                        W += -D*(1.0-E)*W*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -D*(1.0-E)*W*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -D*W*(eta)  
                    
                if(reg_type=='l1' or reg_type == 'M'):
                    if(update_type=='adam'):
                        u = u + D*(eta1/(np.sqrt(vt) + eps))
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        u = u + D*(eta1/(np.sqrt(cache) + eps))
                    else:
                        u = u + D*eta
                    z[:] = W[:]
                    idx_plus = W>0
                    idx_minus = W<0
                    
                    W_temp = np.zeros(W.shape)
                    if(update_type=='adam' or update_type == 'adagrad' or update_type =='rmsprop'):
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u[idx_plus]+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u[idx_minus]-q[idx_minus]))                    
                    else:
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u-q[idx_minus]))
                    
                    W[idx_plus]=W_temp[idx_plus]
                    W[idx_minus]=W_temp[idx_minus]
                    q=q+(W-z)
                    
                if(reg_type=='en'):
                    if(update_type=='adam'):
                        u = u + D*E*(eta1/(np.sqrt(vt) + eps))
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        u = u + D*E*(eta1/(np.sqrt(cache) + eps))                    
                    else:
                        u = u + D*E*eta
                    z[:] = W[:]
                    idx_plus = W>0
                    idx_minus = W<0
                    
                    W_temp = np.zeros(W.shape)
                    if(update_type=='adam' or update_type == 'adagrad' or update_type =='rmsprop'):
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u[idx_plus]+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u[idx_minus]-q[idx_minus]))                    
                    else:
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u-q[idx_minus]))
                    W[idx_plus]=W_temp[idx_plus]
                    W[idx_minus]=W_temp[idx_minus]
                    q=q+(W-z)
                
                if(np.sum(W**2)>W_clip_threshold):#gradient clipping
                    W = W_clip_threshold*W/np.sum(W**2)
            
            if(iter1%iter_eval==0):                    
                #once the W are calculated for each epoch we calculate the scores
                scores=xTrain.dot(W)
                N=scores.shape[0]
                correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
                mat = (scores.transpose()-correct_scores.transpose()).transpose() 
                mat = mat+1.0
                mat[range(N),np.array(yTrain,dtype='int32')] = 0.0

                scores1  = np.zeros(scores.shape)
                scores1[:] = scores[:]
                scores1[range(N),np.array(yTrain,dtype='int32')] = -np.inf
                max_scores = np.max(scores1,axis =1)
                mat1 = 1 - correct_scores + max_scores
                
                f=0.0
                if(reg_type=='l2'):
                    f += D*0.5*np.sum(W**2) 
                if(reg_type=='l1'):
                    f += D*np.sum(np.abs(W))
                if(reg_type=='en'):
                    f += D*0.5*(1-E)*np.sum(W**2)  +  D*E*np.sum(np.abs(W))
                  
                if(class_weighting=='average'):
                    f1 = C*0.5*np.sum(scores**2) + 0.5*np.sum(mat1**2)
                    f += (1.0/N)*f1 
                else:
                    f1 = C*0.5*np.sum((scores**2)*sample_weights[:,None]) + 0.5*np.sum((mat1**2)*sample_weights[:,None])
                    f+= (1.0/numClasses)*f1 
                    
                for i in range(numClasses):
                    #first term in objective function for margin
                    if(K_plus is not None):
                        w = W[0:(M-1),i]
                        f2 = np.dot(np.dot(K_plus,w),w)
                        f += ((0.5*E)/(numClasses))*f2  
                        #the second term in the objective function for margin
                    if(K_minus is not None):
                        f3 = np.dot(temp_mat[:,i],w)
                        f += -((0.5*E)/(numClasses))*f3
                if(verbose == True):        
                    print('iter1=%d, f=%0.3f'%(iter1,f))
                fvals[iter1]=f
                rel_error=np.abs(f_prev-f)/np.abs(f_prev)
                error_tol1 = 1e-02
                maxW = np.max(np.abs(W[:-1,:]),axis = 0)
                max_W = np.ones(W[:-1,].shape)*maxW
                W[:-1,][np.abs(W)[:-1,]<error_tol1*max_W]=0.0
                
                if(f<f_best):
                    f_best=f
                    W_best[:]=W[:]
                    maxW = np.max(np.abs(W_best[:-1,:]),axis = 0)
                    max_W = np.ones(W_best[:-1,].shape)*maxW
                    W_best[:-1,][np.abs(W_best)[:-1,]<error_tol1*max_W]=0.0
                    iter_best=iter1
                else:
                    break
                f_prev=f      
 
#            eta=eta_zero/np.power((iter1+1),1)
            
        fvals[iter1]=-1
        max_W1 = np.max(np.abs(W_best[:-1,:]),axis = 0)
        max_W1 = np.ones(W_best[:-1,].shape)*max_W1        
        W1_best=np.zeros(W_best.shape)
        W2=np.zeros(W_best.shape)
        W2[:]=W_best[:]   
        train_acc_best=0.0
        for l1_ratio in [1e-02,5e-02,1e-01,2e-01,3e-01]:
            W2[:]=W_best[:]
            W2[:-1,][np.abs(W2)[:-1,]<l1_ratio*max_W1]=0.0
            scores=xTrain.dot(W2)
            pred = np.argmax(scores,axis=1)
            train_acc = np.sum(pred == yTrain)*100/yTrain.shape[0]
#            print('l1_ratio =%0.3f, train_acc=%0.3f'%(l1_ratio,train_acc))
            if(train_acc>=train_acc_best):
                train_acc_best=train_acc-0.5
                W1_best[:]=W2[:]               
        return W1_best,f_best,iter_best,fvals
        
    def predict(self,data, xTrain, W,  me, std, subset):
        """
        Returns the predicted labels
        Parameters:
        ----------
        data: np.array (n_samples1, n_features)
            test data
        xTrain: np.array (n_samples2, n_features)
            train data
        W: np.array 
            learnt weight matrix
        me: np.array (n_features,) or (n_subset,)
            mean of train data features
        std: np.array (n_features,) or (n_subset,)
            standard deviation of train data features
        subset: list
            subset of train data to be used
        Returns:
        -------
        labels: np.array (n_samples1,)
            labels of  the train data
        scores: np.array (n_samples1, numClasses)
            scores of each sample
        """
        util = utils()
        kernel_type = self.kernel_type
        gamma = self.gamma
        levels = self.levels
        n_components = self.n_components
        compress_type = self.compress_type
        N = data.shape[0]  
        M = data.shape[1]
        label = np.zeros((N,))
        feature_indices = np.array(range(M))
        sample_indices = np.array(range(xTrain.shape[0]))
        if(compress_type == 'zero_one'):
            xTrain = util.zero_one_normalization(xTrain)
            data = util.zero_one_normalization(data)
        elif(compress_type == 'sigmoid'):
            xTrain = util.logsig(xTrain)
            data = util.logsig(data)
        elif(compress_type == 'saturate'):
            xTrain = util.saturate_fcn1(xTrain)
            data = util.saturate_fcn1(data)
        else:
            raise ValueError('wrong compress_type selected!')
           

        X1, X2 = util.select_(data, xTrain, kernel_type, subset, feature_indices, sample_indices)
        
        X1 = util.quantize(X1,levels)
        X1 = util.dequantize(X1,levels)
#        X1 = util.onehot_minibatch(X1,levels)
#        X1 = util.tempcode_minibatch(X1,levels)
        if(X2 is not None):
            X2 = util.quantize(X2, levels)
            X2 = util.dequantize(X2, levels)            
#            X2 = util.onehot_minibatch(X2, levels)
#            X2 = util.tempcode_minibatch(X2, levels)
        data1 = util.kernel_transform(X1 = X1, X2 = X2, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
        
        if(kernel_type != 'linear_primal'):
            centering = True
        else:
            centering = False
        
        if(centering == True):
            data1 = util.normalize_(data1,me,std)
            
        data1 = util.add_bias(data1)                    
        scores = data1.dot(W)
        label = np.argmax(scores,axis=1) 
        return label,scores   
                        
    def accuracy_classifier(self,actual_label,found_labels):
        acc=np.divide(np.sum(actual_label==found_labels)*100.0 , actual_label.shape[0],dtype='float64')
        return acc