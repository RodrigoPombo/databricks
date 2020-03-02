// Databricks notebook source
// MAGIC %python
// MAGIC import os
// MAGIC import warnings
// MAGIC import sys
// MAGIC 
// MAGIC import pandas as pd
// MAGIC import numpy as np
// MAGIC from itertools import cycle
// MAGIC import matplotlib.pyplot as plt
// MAGIC from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
// MAGIC from sklearn.model_selection import train_test_split
// MAGIC from sklearn.linear_model import ElasticNet
// MAGIC from sklearn.linear_model import lasso_path, enet_path
// MAGIC from sklearn import datasets
// MAGIC 
// MAGIC # Import mlflow
// MAGIC import mlflow
// MAGIC import mlflow.sklearn
// MAGIC 
// MAGIC # Load Diabetes datasets
// MAGIC diabetes = datasets.load_diabetes()
// MAGIC X = diabetes.data
// MAGIC y = diabetes.target
// MAGIC 
// MAGIC # Create pandas DataFrame for sklearn ElasticNet linear_model
// MAGIC Y = np.array([y]).transpose()
// MAGIC d = np.concatenate((X, Y), axis=1)
// MAGIC cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
// MAGIC data = pd.DataFrame(d, columns=cols)

// COMMAND ----------

// MAGIC %python
// MAGIC def plot_enet_descent_path(X, y, l1_ratio):
// MAGIC     # Compute paths
// MAGIC     eps = 5e-3  # the smaller it is the longer is the path
// MAGIC 
// MAGIC     # Reference the global image variable
// MAGIC     global image
// MAGIC     
// MAGIC     print("Computing regularization path using ElasticNet.")
// MAGIC     alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio, fit_intercept=False)
// MAGIC 
// MAGIC     # Display results
// MAGIC     fig = plt.figure(1)
// MAGIC     ax = plt.gca()
// MAGIC 
// MAGIC     colors = cycle(['b', 'r', 'g', 'c', 'k'])
// MAGIC     neg_log_alphas_enet = -np.log10(alphas_enet)
// MAGIC     for coef_e, c in zip(coefs_enet, colors):
// MAGIC         l1 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)
// MAGIC 
// MAGIC     plt.xlabel('-Log(alpha)')
// MAGIC     plt.ylabel('coefficients')
// MAGIC     title = 'ElasticNet Path by alpha for l1_ratio = ' + str(l1_ratio)
// MAGIC     plt.title(title)
// MAGIC     plt.axis('tight')
// MAGIC 
// MAGIC     # Display images
// MAGIC     image = fig
// MAGIC     
// MAGIC     # Save figure
// MAGIC     fig.savefig("ElasticNet-paths.png")
// MAGIC 
// MAGIC     # Close plot
// MAGIC     plt.close(fig)
// MAGIC 
// MAGIC     # Return images
// MAGIC     return image    

// COMMAND ----------

// MAGIC %python
// MAGIC def train_diabetes(data, in_alpha, in_l1_ratio):
// MAGIC   # Evaluate metrics
// MAGIC   def eval_metrics(actual, pred):
// MAGIC       rmse = np.sqrt(mean_squared_error(actual, pred))
// MAGIC       mae = mean_absolute_error(actual, pred)
// MAGIC       r2 = r2_score(actual, pred)
// MAGIC       return rmse, mae, r2
// MAGIC 
// MAGIC   warnings.filterwarnings("ignore")
// MAGIC   np.random.seed(40)
// MAGIC 
// MAGIC   # Split the data into training and test sets. (0.75, 0.25) split.
// MAGIC   train, test = train_test_split(data)
// MAGIC 
// MAGIC   # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
// MAGIC   train_x = train.drop(["progression"], axis=1)
// MAGIC   test_x = test.drop(["progression"], axis=1)
// MAGIC   train_y = train[["progression"]]
// MAGIC   test_y = test[["progression"]]
// MAGIC 
// MAGIC   if float(in_alpha) is None:
// MAGIC     alpha = 0.05
// MAGIC   else:
// MAGIC     alpha = float(in_alpha)
// MAGIC     
// MAGIC   if float(in_l1_ratio) is None:
// MAGIC     l1_ratio = 0.05
// MAGIC   else:
// MAGIC     l1_ratio = float(in_l1_ratio)
// MAGIC   
// MAGIC   # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
// MAGIC   with mlflow.start_run():
// MAGIC     lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
// MAGIC     lr.fit(train_x, train_y)
// MAGIC 
// MAGIC     predicted_qualities = lr.predict(test_x)
// MAGIC 
// MAGIC     (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
// MAGIC 
// MAGIC     # Print out ElasticNet model metrics
// MAGIC     print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
// MAGIC     print("  RMSE: %s" % rmse)
// MAGIC     print("  MAE: %s" % mae)
// MAGIC     print("  R2: %s" % r2)
// MAGIC 
// MAGIC     # Set tracking_URI first and then reset it back to not specifying port
// MAGIC     # Note, we had specified this in an earlier cell
// MAGIC     #mlflow.set_tracking_uri(mlflow_tracking_URI)
// MAGIC 
// MAGIC     # Log mlflow attributes for mlflow UI
// MAGIC     mlflow.log_param("alpha", alpha)
// MAGIC     mlflow.log_param("l1_ratio", l1_ratio)
// MAGIC     mlflow.log_metric("rmse", rmse)
// MAGIC     mlflow.log_metric("r2", r2)
// MAGIC     mlflow.log_metric("mae", mae)
// MAGIC     mlflow.sklearn.log_model(lr, "model")
// MAGIC     modelpath = "/dbfs/mlflow/test_diabetes/model-%f-%f" % (alpha, l1_ratio)
// MAGIC     mlflow.sklearn.save_model(lr, modelpath)
// MAGIC     
// MAGIC     # Call plot_enet_descent_path
// MAGIC     image = plot_enet_descent_path(X, y, l1_ratio)
// MAGIC     
// MAGIC     # Log artifacts (output files)
// MAGIC     mlflow.log_artifact("ElasticNet-paths.png")

// COMMAND ----------

// MAGIC %fs rm -r dbfs:/mlflow/test_diabetes

// COMMAND ----------

// MAGIC %python
// MAGIC train_diabetes(data, 0.2, 0.01)

// COMMAND ----------

// MAGIC %python
// MAGIC display(image)