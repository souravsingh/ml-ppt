#### Machine Learning
by Sourav Singh

---

#### What is Machine Learning?

Machine Learning is a sub-domain of Artificial Intelligence that deals with the ability to learn and improve 
from experience without any explicit programming.

---

#### Various Types of algorithms in Machine Learning

The various types of algorithms in Machine Learning are-

* Supervised learning: These algorithms make use of labelled data to predict certain properties. They can be divided into two types:
    * Classification, used for classifying data.
    * Prediction, used to prediction of a certain value.


* Unsupervised learning: These algorithms make use of unlabelled data to obtain patterns.

* Semi-Supervised learning: These algorithms come in between supervised and unsupervised learning algorithms.

* Reinforcement Learning: These types of algorithms aim to use the observations from the environment to take actions that either minimize the risk factor.

---

#### Working behind Machine Learning algorithms

Most supervised learning algorithms require a target variable from which classification can be done. 
We can show this through a classification example shown below.
(The example is adapted from scikit-learn).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import Iris data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target # We make use of Y for training our algorithm

h = .02  # step size in the mesh
logreg = linear_model.LogisticRegression(C=1e5) # We make use of Logistic Regression for classification

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(16, 9))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
```
The Python code generates the plot as shown in next slide-

---

<img src="https://cocalc.com/aa1f3474-17ab-4a1d-bf06-3ce11eadc790/raw/.smc/jupyter/blobs/a.png?sha1=4ba608ea3cbb2d287d617764d65b83a48fba4146&attempts=0">

As we can see above, the datapoints(coloured in light-blue, brown and olive color) are divided into their respecitve boundaries.
so we were able to classify the datapoints into their respective boundaries.
---

#### Applications of Machine Learning

Machine Learning has been used for various field like Astronomy, Medicine, Physics etc. A few of the applications are-

* Classifying stars in a picture based on brightness and size.
* Classifiying whether a mail is a spam or not.
* Detecting the sentiment behind a tweet or a facebook post.
* Detecting type of peptides or RNA sequences based on certain properties.

---
