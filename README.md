Mall Customer Segmentation Using K-Means Clustering
Project Overview
This project uses unsupervised machine learning to group mall
customers based on their annual income and spending score. The K-Means
clustering algorithm identifies customer segments that can help a
business better understand shopping behavior and plan targeted marketing
strategies.
Objectives
- Explore the customer dataset using Exploratory Data Analysis (EDA).
- Check for missing values, duplicate rows, and unique values.
- Visualize customer feature distributions and investigate potential
  outliers.
- Examine correlations between numerical features.
- Scale the selected features before clustering.
- Use the Elbow Method to help choose the number of clusters.
- Train a K-Means model and evaluate the resulting clusters.
- Visualize clusters and their centroids.
- Save the dataset with assigned cluster labels.
- Predict the cluster for a new customer.
Dataset
The notebook reads a file named Mall_Customers.csv.

The analysis uses these features for clustering:
- Annual Income (k$) --- annual income, expressed in thousands of
  dollars.
- Spending Score (1-100) --- a spending score on a scale from 1 to
  100.
The notebook also explores Age during EDA. CustomerID is an
identifier and is not used as a clustering feature.

Dataset requirement: Place Mall_Customers.csv in the same
directory as the notebook before running it. If you obtained the dataset
from an external source, add its source and license here.
Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
Workflow
1. Load the data with Pandas.
2. Check data quality by inspecting missing values, duplicate rows,
   and unique values.
3. Perform EDA using histograms, boxplots, and a correlation
   heatmap.
4. Select features: annual income and spending score.
5. Standardize features using StandardScaler so differences in
   scale have less influence on distance calculations.
6. Choose a cluster count by plotting inertia (WCSS) for values of
   K from 1 to 10 with the Elbow Method.
7. Train K-Means with five clusters, k-means++ initialization,
   random_state=42, n_init=20, and max_iter=500.
8. Evaluate clustering using inertia and the silhouette score.
9. Visualize results with customer points and cluster centroids.
10. Save results to Mall_Customers_KMeans_Groups.csv.
11. Predict a new customer's cluster using the already-fitted scaler
    and K-Means model.
Installation
Python 3.10 or later is a reasonable starting point. Install the
required packages:
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
How to Run
1. Clone or download this repository.
2. Place Mall_Customers.csv beside the notebook.
3. Install the dependencies listed above.
4. Start Jupyter:
   jupyter notebook
5. Open ASS4(2).ipynb (or the notebook's renamed file in the
   repository).
6. Run the notebook cells in order.
The notebook creates Mall_Customers_KMeans_Groups.csv after the
clustering analysis.
Predict a New Customer's Cluster
The notebook demonstrates prediction using annual income and spending
score. It scales the new customer's input with the same fitted
StandardScaler and then calls the trained K-Means model's predict()
method.

Example inputs in the notebook:
- Annual income: 80 (meaning $80,000 when using the dataset's k$\
  convention)
- Spending score: 75
Cluster numbers are labels, not rankings. For example, Cluster 4 is
not inherently better than Cluster 1; interpret each cluster by
examining its income and spending characteristics.
Evaluation
The notebook reports:
- Inertia (WCSS): the sum of squared distances from samples to
  their assigned cluster centers. Lower values generally occur as the
  number of clusters increases, so inertia should be considered
  alongside the Elbow Method.
- Silhouette score: measures how well-separated the clusters are.
  Higher values generally indicate better-defined clusters; the score
  should be interpreted in context.
The notebook uses five clusters in its final model. This is a chosen
setting in the notebook, not proof that five clusters are universally
optimal. Review the Elbow plot and silhouette score when assessing the
choice.
Output
- Mall_Customers_KMeans_Groups.csv --- original customer records
  with an additional Cluster column.
- Plots for feature distributions, outlier exploration, correlations,
  the Elbow Method, and customer clusters.
Limitations
- K-Means is sensitive to feature selection, scaling, initialization,
  and the chosen number of clusters.
- Cluster labels do not have an inherent business meaning; profile
  each group before naming it.
- This project uses two features for clustering, so the resulting
  segments do not represent every aspect of customer behavior.
- The notebook's prediction examples require the fitted scaler and
  K-Means model to have been created by running the earlier cells.
Future Improvements
- Compare several values of K using both the Elbow Method and
  silhouette score.
- Profile each cluster with summary statistics and descriptive
  business labels.
- Try alternative clustering algorithms, such as DBSCAN, and compare
  their results.
- Add a reusable prediction script or a small interactive interface.
Author
Abhishek babu
If you publish this project, include the dataset source and license
information where applicable, and verify that the notebook runs from a
clean environment before sharing it.
