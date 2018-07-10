# T-EFS-LS-MCM
Repository for non-Mercer multiclass EFS-LS-MCM presented in IJCNN 2018

## Abstract
This paper extends the idea of Least Squares Minimal Complexity Machines (LS-MCMs) to non-Mercer kernels. There are no efficient solvers for LS-SVMs with non-Mercer kernels for large scale datasets. Here, we propose two variants of our novel multiclass loss LS-MCMs. Firstly, with an L1 regularizer and secondly, with an explicit margin regularizer along with L1 norm in the Empirical Feature Space (EFS). Both these methods can be scaled to large datasets with the use of "prototype vectors" selected from the dataset. The first optimization algorithm can be solved efficiently using Stochastic Gradient Descent (SGD) directly as the problem remains convex in the parameter space. The second problem however, is solved using difference of convex functions (DC) programming with SGD due to the non-convex nature of margin regularizer. Our method also obtains a sparse solution as opposed to the one obtained using LS-SVMs which tend to be non-sparse.

## Code
The code is written in Python 3.6 and requires the following packages
* sklearn,
* scipy,
* numpy, 
* pandas

All packages except can be found in Anaconda python installation.

## Examples
To see a template for running T-EFS-LS-MCM, refer 
1) MCM_SGD_2.m

## Citation
If you use the code please cite the following papers using the bibtex entry:

```
@article{sharma2017large,
  title={Large-Scale Minimal Complexity Machines Using Explicit Feature Maps},
  author={Sharma, Mayank and Soman, Sumit and Pant, Himanshu and others},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume={47},
  number={10},
  pages={2653--2662},
  year={2017},
  publisher={IEEE}
}

@article{sharma2018ultra,
  title={Ultra-Sparse Classifiers Through Minimizing the VC Dimension in the Empirical Feature Space},
  author={Sharma, Mayank and Soman, Sumit and Pant, Himanshu and others},
  journal={Neural Processing Letters},
  pages={1--33},
  year={2018},
  publisher={Springer}
}

@inproceedings{sharma2018large,
	title={Non-Mercer Large Scale Multiclass Least Squares Minimal Complexity Machines.},
	author={Sharma, Mayank and Soman, Sumit and Pant, Himanshu and others},
	booktitle={IJCNN},
	year={2018}
}
```

## Research Paper
The papers for the same are available at:

* http://ieeexplore.ieee.org/abstract/document/7942005/
* https://link.springer.com/article/10.1007/s11063-018-9793-9
