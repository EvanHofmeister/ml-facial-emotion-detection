# **Breast Cancer Classification**
<center><img src = "references/images/BreastCancer.jpg" width="900" height="350"/></center>

# **Project Overview**
This project focuses on classifying breast cancer tumors as either benign or malignant. Utilizing the Breast Cancer Wisconsin (Diagnostic) dataset, various machine learning models are applied and contrasted.

### Data Source
[The Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), available from the UCI Machine Learning Repository, consists of features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. It contains 569 instances, each with 30 real-valued features, providing data on the characteristics of the cell nuclei. The objective of the dataset is to classify whether the breast cancer is benign or malignant.

There are 32 rows in the dataset where the first two are the index and target. The remaining 30 can be divided into groups of 10 where:
* 3-12: Mean of `a-j`
* 13-22: SE of `a-j`
* 23-32: Worst (defined for each metric) of `a-j`

### Methodology
* Data Preprocessing: The dataset is cleaned and normalized and analyzed to capture any relationships between features/targets.
* Feature Engineering: Key features are selected based on a survey of methods.
* Model Building: I experiment with several machine learning algorithms, including Logistic Regression, Decision Trees, Random Forest, XGBoost, and ANN.
* Evaluation: Models are evaluated based on accuracy, precision, and recall metrics to determine the most effective approach.

### Results
The best results were achieved with an optimized Random Forest model, yielding an AUC of `0.986` and an F1 score of `0.968`. This work not only demonstrates the potential of machine learning in healthcare but also provided hands-on experience with Scikit-learn and Flask, expanding my experience with both model development and application deployment.

### Instructions for setting up project and installing dependencies