# Heart Disease Data Science and Predictive Modeling
An end-to-end data science and machine learning project developed in Python to analyze, cluster, and build predictive models on clinical healthcare datasets.

## Tech Stack and Libraries
* Language: Python
* Data Manipulation: Pandas, NumPy
* Machine Learning (Supervised): Scikit-learn (Logistic Regression, Random Forest, Decision Tree)
* Machine Learning (Unsupervised): Scikit-learn (K-Means, PCA), MLxtend (Apriori Algorithm)
* Data Visualization: Matplotlib, Seaborn

## Project Architecture and Workflow
The architecture is split into modular components capturing the entire data lifecycle:

### 1. Data Preprocessing and Analytics (preprocessing.py)
* Implemented data cleaning techniques including outlier removal via the Interquartile Range (IQR) method.
* Handled missing values using statistical imputation (median/mode) and transformed continuous features through standard scaling (StandardScaler).
* Conducted Pearson correlation mapping to isolate critical features affecting cardiovascular states.

### 2. Supervised Predictive Modeling (predictive_models.py)
* Developed and benchmarked three core predictive models: Logistic Regression, Random Forest, and Decision Trees to analyze patient heart disease status.
* Evaluated performance limitations through Confusion Matrices, addressing data imbalance challenges where high accuracy metrics can mask critical Recall and Precision variations in clinical diagnostics.

### 3. Unsupervised Clustering and Pattern Mining (clustering_and_mining.py)
* Applied K-Means Clustering to segment patients into discrete risk-profiles based on high-dimensional data, utilizing Silhouette Scores and Inertia to optimize cluster locations.
* Compressed high-dimensional spaces using Principal Component Analysis (PCA) for dense structural data reduction and 2D visualization.
* Executed the Apriori Algorithm to mine multi-factor association rules, capturing hidden behavioral and clinical hazard chains (such as combinations of High Blood Pressure, Cholesterol flags, and Family History).
