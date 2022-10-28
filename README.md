# Credit-card-Fraud-Detection-
About Credit Card Fraud Detection<br />
In this machine learning project, we solve the problem of detecting credit card fraud transactions using machine numpy, scikit learn, and few other python libraries. We overcome the problem by creating a binary classifier and experimenting with various machine learning techniques to see which fits better.

# Understanding and Defining Fraud
Credit card fraud is any dishonest act and behaviour to obtain information without the proper authorization from the account holder for financial gain. Among different ways of frauds, Skimming is the most common one, which is the way of duplicating of information located on the magnetic strip of the card. Apart from this, the other ways are: <br />

Manipulation/alteration of genuine cards  <br />
Creation of counterfeit cards <br />
Stolen/lost credit cards <br />
Fraudulent telemarketing <br />

#  Credit Card Fraud Dataset
The dataset consists of 31 parameters. Due to confidentiality issues, 28 of the features are the result of the PCA transformation. “Time’ and “Amount” are the only aspects that were not modified with PCA.
There are a total of 284,807 transactions with only 492 of them being fraud. So, the label distribution suffers from imbalance issues.

# Tools and Libraries used
We use the following libraries and frameworks in credit card fraud detection project.
Python – 3.x <br />
Numpy – 1.19.2 <br />
Scikit-learn – 0.24.1 <br />
Matplotlib – 3.3.4 <br />
Imblearn – 0.8.0 <br />
Collections, Itertools <br />

# Steps to Develop Credit Card Fraud Classifier in Machine Learning
Our approach to building the classifier is discussed in the steps:

1-Perform Exploratory Data Analysis (EDA) on our dataset <br />
2-Apply different Machine Learning algorithms to our dataset <br />
3-Train and Evaluate our models on the dataset and pick the best one.
#  Project Pipeline
The project pipeline can be briefly summarized in the following four steps:<br />

1-Data Understanding: Here, we need to load the data and understand the features present in it. This would help us choose the features that we will need for your final model.<br />
2-Exploratory data analytics (EDA): Normally, in this step, we need to perform univariate and bivariate analyses of the data, followed by feature transformations, if necessary. For the current data set, because Gaussian variables are used, we do not need to perform Z-scaling. However, you can check if there is any skewness in the data and try to mitigate it, as it might cause problems during the model-building phase.<br />
3-Train/Test Split: Now we are familiar with the train/test split, which we can perform in order to check the performance of our models with unseen data. Here, for validation, we can use the k-fold cross-validation method. We need to choose an appropriate k value so that the minority class is correctly represented in the test folds.<br />
4-Model-Building/Hyperparameter Tuning: This is the final step at which we can try different models and fine-tune their hyperparameters until we get the desired level of performance on the given dataset. We should try and see if we get a better model by the various sampling techniques.<br />
Model Evaluation: We need to evaluate the models using appropriate evaluation metrics. Note that since the data is imbalanced it is is more important to identify which are fraudulent transactions accurately than the non-fraudulent. We need to choose an appropriate evaluation metric which reflects this business goal.
# Algorithms
# 1-Logistic regression <br />
Logistic regression is an example of supervised learning. It is used to calculate or predict the probability of a binary (yes/no) event occurring <br />
Training data accuracy : 91% <br />
Test data accuracy : 92% <br />
# 2-Decision Tree Classifier <br />
type of Supervised Machine Learning (that is you explain what the input is and what the corresponding output is in the training data) where the data is continuously split according to a certain parameter <br />
Hyper prameter :max_depth=6 <br />
Training data accuracy : 98% <br />
Test data accuracy : 92% <br />
# 3-K-Nearest Neighbors <br />
The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. <br />
Hyper prameter :n=5 <br />
Training data accuracy : 75% <br />
Test data accuracy : 66% <br />
# 4-Random Forest Classifier model <br />
Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time  <br />
Hyper prameter :max_depth=7 <br />
Training data accuracy : 97% <br />
Test data accuracy : 93% <br />
# 5-Gaussian Naive Bayes
Training data accuracy : 85% <br />
Test data accuracy : 86% <br />
# 6-support vector machines
Training data accuracy : 54% <br />
Test data accuracy : 61% <br />
# the best model for this probblem according to the  accuracuy :
Random Forest Classifier model
