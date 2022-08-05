# XGBoost
 contain sample codes and tuning examples

From Miguel Fierro on LinkedIn

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
