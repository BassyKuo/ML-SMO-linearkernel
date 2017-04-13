# ML-SMO-linearkernel
The implement of SMO-algorithm in machine learning course.

# Execution Version
Python 3.5.2

# Requirements
* Numpy >= 1.12.1

# Usage
```
$ python3 main_v2.py -h
```

For example,
```
$ python3 main_v2.py --trainset messidor_features_training.csv --testset messidor_features_testing.csv --c 0.7 --n 5 --tol 0.001
```

# Reference
1. [Sequential Minimal Optimization- A Fast Algorithm for Training Support Vector Machines](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. [Improvements to platt's SMO algorithm for SVM classifier design](http://web.cs.iastate.edu/~honavar/keerthi-svm.pdf)
3. [CS 229, Autumn 2009 | The Simplified SMO Algorithm](http://cs229.stanford.edu/materials/smo.pdf)
