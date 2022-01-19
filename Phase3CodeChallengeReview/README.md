
# Phase 3 Review

![review guy](https://media.giphy.com/media/3krrjoL0vHRaWqwU3k/giphy.gif)

# TOC 

1. [Gradient Descent](#grad_desc)
2. [Logistic Regression](#logistic)
3. [Confusion Matrix](#con_mat)
4. [Accuracy/Precision/Recall/F1](#more_metric)
5. [auc_roc](#auc_roc)
3. [Algos](#algos)


```python
from src.student_caller import one_random_student
from src.student_list import quanggang
```

<a id='grad_desc'></a>

## Gradient Descent

Question: What is a loss function? (Explain it in terms of the relationship between true and predicted values) 



```python
one_random_student(quanggang)
```

Question: What loss functions do we know and what types of data work best with each?


```python
one_random_student(quanggang)

```

To solidify our knowledge of gradient descent, we will use Sklearn's stochastic gradient descent algorithm for regression [SGDRegressor](https://scikit-learn.org/stable/modules/sgd.html#regression).   Sklearn classifiers share many methods and parameters, such as fit/predict, but some have useful additions.  SGDRegressor has a new method called partial_fit, which will allow us to inspect the calculated coefficients after each step of gradient descent.  
We will use the diabetes dataset for this task.  

```python
from sklearn.datasets import load_diabetes
import numpy as np

data = load_diabetes(as_frame=True)
X = data['data']
y = data['target']
```


```python
X.shape
```


```python
X.head()
```


```python
from sklearn.linear_model import SGDRegressor
```


```python
# Instantiate a SGDRegressor object and run partial fit on X and y. For now, pass the argument `penalty=None`
```


```python
one_random_student(quanggang)
```


```python
# Inspect the coefficient array
```


```python
one_random_student(quanggang)
```


```python
# Import mean_squared_error from metrics, and pass in the true ys, an array of predictions
# and the agrument squared = False
```


```python
one_random_student(quanggang)
```


```python
# Repeat the partial fit. Inspect, RMSE, coefficients.
```


```python
one_random_student(quanggang)
```

Pick a coefficient, and explain the gradient descent update.

Together, let's plot the trajectory of one coefficient against the loss. 

```python
# code
```
Compare that to a full fit of the SGDRegressor.

```python
# code
```

<a id='logistic'></a>

# Logistic Regression and Modeling

What type of target do we feed the logistic regression model?


```python
one_random_student(quanggang)
```


```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
X = data['data']
y = data['target']
```


```python
# Perform a train-test split
```


```python
one_random_student(quanggang)
```

Question: What is the purpose of train/test split?  



```python
one_random_student(quanggang)
```

Question: Why should we never fit to the test portion of our dataset?


```python
one_random_student(quanggang)
```


```python
# Scale the training set using a standard scaler
ss = None
X_train_scaled = None
```


```python
one_random_student(quanggang)
```


```python
X_train_scaled.head()
```

Question: Why is scaling our data important? For part of your answer, relate to one of the advantages of logistic regression over another classifier.


```python
# fit model with logistic regression to the appropriate portion of our dataset
```


```python
one_random_student(quanggang)
```

Now that we have fit our classifier, the object `lr` has been filled up with information about the best fit parameters.  Take a look at the coefficients held in the `lr` object.  Interpret what their magnitudes mean.


```python
# Inspect the .coef_ attribute of lr and interpret
```


```python
one_random_student(quanggang)
```

Logistic regression has a predict method just like linear regression.  Use the predict method to generate a set of predictions (y_hat_train) for the training set.


```python
# use predict to generate a set of predictions
y_hat_train = None
```


```python
one_random_student(quanggang)

```

<a id='con_mat'></a>

### Confusion Matrix

Confusion matrices are a great way to visualize the performance of our classifiers. 

Question: What does a good confusion matrix look like?


```python
one_random_student(quanggang)
```


```python
# create a confusion matrix for our logistic regression model fit on the scaled training data
```


```python
one_random_student(quanggang)
```

<a id='more_metrics'></a>

## Accuracy/Precision/Recall/F_1 Score

We have a bunch of additional metrics, most of which we can figure out from the CM

Question: Define accuracy. What is the accuracy score of our classifier?


```python
# Confirm accuracy in code
```


```python
one_random_student(quanggang)

```

Question: Why might accuracy fail to be a good representation of the quality of a classifier?


```python

one_random_student(quanggang)
```

Question: Define recall. What is the recall score of our classifier?


```python
# Confirm recall in code
```


```python

one_random_student(quanggang)
```

Question: Define precision? What is the precision score of our classifier?


```python
# Confirm precision in code
```


```python

one_random_student(quanggang)
```

Question: Define f1 score? What is the f1 score score of our classifier?


```python

one_random_student(quanggang)
```

<a id='auc_roc'></a>

## Auc_Roc

The AUC_ROC curve can't be deduced from the confusion matrix.  Describe what the AUC_ROC curve shows. 
Look [here](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) for some nice visualizations of AUC_ROC.
Describe the AUC_ROC curve.  What does a good AUC_ROC curve look like? What is a good AUC_ROC score?

```python

one_random_student(quanggang)
```

One of the advantages of logistic regression is that it generates a set of probabilities associated with each prediction.  What is the default threshold?  How would decrease or increasing your threshold affect true positive and false positive rates?


For our scaled X_train, generate an array of probabilities associated with the probability of the positive class.


```python
# your code here
```


```python

one_random_student(quanggang)
```

Now, using those probabilities, create two arrays, one which converts the probabilities to label predictions using the default threshold, and one using a threshold of .4.  How does it affect our metrics?


```python
# Plot the AUC_ROC curve for our classifier
```

<a id='algos'></a>

# More Algorithms

Much of the sklearn syntax is shared across classifiers and regressors.  Fit, predict, score, and more are methods associated with all sklearn classifiers.  They work differently under the hood. KNN's fit method simply stores the training set in memory. Logistic regressions .fit() does the hard work of calculating coefficients. 

![lazy_george](https://media.giphy.com/media/8TJK6prvRXF6g/giphy.gif)

However, each algo also has specific parameters and methods associated with it.  For example, decision trees have feature importances and logistic has coefficients. KNN has n_neighbors and decision trees has max_depth.


Getting to know the algo's and their associated properties is an important area of study. 

That being said, you now are getting to the point that no matter which algorithm you choose, you can run the code to create a model as long as you have the data in the correct shape. Most importantly, the target is the appropriate form (continuous/categorical) and is isolated from the predictors.

Here are the algos we know so far. 
 - Linear Regression
 - Lasso/Ridge Regression
 - Logistic Regression
 - Naive-Bayes
 - KNN
 - Decision Trees
 
> Note that KNN and decision trees also have regression classes in sklearn.


Here are two datasets from seaborn and sklearn.  Let's work through the process of creating simple models for each.


```python
import seaborn as sns
penguins = sns.load_dataset('penguins')
penguins.head()
```
Question: What algorithm would be appropriate based on the target

```python
# split target from predictors
```


```python
one_random_student(quanggang)
```
For the first simple model, let's just use the numeric predictors.

```python
one_random_student(quanggang)
```


```python
# isolate numeric predictors
```


```python
one_random_student(quanggang)
```


```python
# Scale appropriately

```


```python
one_random_student(quanggang)
```


```python
# instantiate appropriate model and fit to appropriate part of data.

```


```python
one_random_student(quanggang)
```


```python
# Create a set of predictions

y_hat_train = None
y_hat_test = None

```


```python
one_random_student(quanggang)
```


```python
# Create and analyze appropriate metrics
```


```python
one_random_student(quanggang)
```


```python
from sklearn.datasets import load_boston
data = load_boston()
X = pd.DataFrame(data['data'], columns = data['feature_names'])
y = data['target']
```
Question: What algorithm would be appropriate based on the target?

```python
one_random_student(quanggang)
```


```python
# split target from predictors
```


```python
one_random_student(quanggang)
```
For the first simple model, let's just use the numeric predictors.

```python
# isolate numeric predictors
```


```python
one_random_student(quanggang)
```


```python
# Scale appropriately

```


```python
one_random_student(quanggang)
```


```python
# instantiate appropriate model and fit to appropriate part of data.

```


```python
one_random_student(quanggang)
```


```python
# Create a set of predictions

y_hat_train = None
y_hat_test = None

```


```python
one_random_student(quanggang)
```


```python
# Create and analyze appropriate metrics
```


```python
one_random_student(quanggang)
```
