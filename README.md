# ML_algorithms
The human body contains a wide variety of microbiome that provide up to 99% of the genetic material present in our bodies, which play an important role in regulating host metabolism and immune system development. This repository includes several machine learning method using the rRNA sequences of gut microbiome to predict disease.


## Five algorithm
 - Logistic Regression
 - NaiveBayes(Gaussian, Bernoulli, Multinomial)
 - RandomForest
 - SVM
 - XGB
These purpose of using these algorithm is to provide the baseline to analyse the performance of deep learning pipeline developed.


## Some design and consideration
 1. To deal with sequences data, the files end with "Single" indicate only taking the last timestep as input and the files end with "Flatten" indicate reshaping from 3 dimesion array(sizes, features, time points) to 2 dimesion(sizes, features * time points)
 2. Since the dataset size is small, cross-validation is used to better
    utilized the data. More concertly, the code using nested
    cross-validation with grid search to find out the best
    hypeparameters. [Link to My blog of cross-validation](https://blog.csdn.net/Aren8/article/details/115479877)
 3. The evalution metric includes f1_score, auc and roc curve.
 
 ## Performance

 
 
