# XGBoost
 contain sample codes and tuning examples

From Miguel Fierro on LinkedIn

## Gradient Boosting Decision Trees

When I'm doing machine learning on tabular data, my go-to algorithm is Gradient Boosting Decision Trees (GBDT), in particular, I tend to use LightGBM in its GPU or Spark implementation. Here is a deep dive on GBDT.

Gradient boosting is a machine learning technique that produces a prediction model in the form of an ensemble of weak classifiers. One of the most popular types of gradient boosting is gradient boosted decision trees (GBDT), which internally is made up of an ensemble of weak decision trees.

Boosting is a technique that builds strong classifiers by sequentially emsambling weak ones. First, a model is trained on the data. Then a second model tries to correct the errors found in the first model. This process is repeated until the error is reduced to a limit or a maximum number of models is added.

In ADA boost, we take the data points that had a bad performance and they are weighted more in the next model.

In gradient boosting, we take the residual, which is the difference between the true labels and the labels predicted by the model.

To train the GBDT, we want to minimize a loss function composed of the residual and a regularization term, to reduce the complexity of the model.

This loss is not easy to compute, one way is via a Taylor expansion. A Taylor expansion is a way to approximate a complex function based on its derivative (gradient or Jacobian) and second derivate (Hessian). In particular, the hessian is not easy to compute, and this is one of the reasons finding the right split in gradient boosting trees can be difficult.

To find the optimal split, instead of computing the Gini or the entropy (like in decision trees), GBDT computes a gain that has into account the gradient, the hessian, and other terms.

When to use GBDT:
In tabular data, almost every time. This is always the first algorithm I try.


## Random Forest

Random Forest is a machine learning model that can be used for general classification or regression problems. It is an improvement on top of Decision Trees and helps to overcome the overfitting of these models.

In particular, Random Forest uses a technique called bagging, short for bootstrap aggregation. The data is randomly divided into a number of bags, and on each bag, we put around 60% of the data.

Then we train a decision tree on each bag and compute the ensemble of these models. The training of Random Forest is done in parallel.

An interesting detail is that when we build the subsample data to generate the bags, we add the data with replacement (this means that the same data can go to the same bag).

Adding data with replacement ensures that we don't run out of data points, and that each bag is large enough to be a good representation of the full dataset.

Bagging is actually independent of the algorithm, even though it is used in Random Forest, it can also be used in Gradient Boosting Decision Trees or other machine learning algorithms.

In terms of training, a Decision Tree is trained on each of the bags. For finding the optimal split, we use the same measures of purity that are used in individual Decision Trees, the Gini index, or the entropy.

Gini index is a measure of the variance of data points on each node. If we have a lot of data points belonging to the wrong class, the Gini index is high, if all the data points belong to the same class, Gini is zero.

The other way of computing the optimal split is using entropy. Entropy takes a probabilistic approach and treats each node as a probability distribution of the data. It measures the uncertainty of the node, if there are a lot of data points of the wrong class, the node doesn't have a lot of certainty and the entropy is high. However, if all the data points belong to the same class, the node has complete certainty about the class of the data, and the entropy is zero.

After all the trees have been trained on each bag, we create an ensemble of all trees. This is a majority vote, we get a prediction for each tree and get the most voted class.

Using ensembles is a very good way of improving generalization and reducing overfitting.

When to use Random Forest:
For tabular data, Random Forests are a good solution to try together with Gradient Boosted Decision Trees (GBDT). As a rule of thumb, if the data has a very complex signal, GBDT tend to perform better. But in general, my advice is to try both of them and compare.
