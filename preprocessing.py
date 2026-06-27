import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Leximi i datasetit
df_original = pd.read_csv("heart_disease.csv")
# Kopja per preprocessing
df = df_original.copy()
print("DATASET PARA PREPROCESSING")
print(df.head())
# MISSING VALUES
print("\nMISSING VALUES")
print(df.isnull().sum())
# DUPLICATE VALUES
print("\nDUPLICATE VALUES")
print(df.duplicated().sum())
# Largimi i duplicate rows
df = df.drop_duplicates()
print("\nSHAPE PAS LARGIMIT TE DUPLIKATEVE")
print(df.shape)
# IDENTIFIKIMI I KOLONAVE
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns
# MISSING VALUES HANDLING
# Numeric columns i zevendeson me  median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns  i zevendeson mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMISSING VALUES PAS PREPROCESSING")

print(df.isnull().sum())

# VISUALIZATIONS PARA PREPROCESSING

# HISTOGRAMS
df[num_cols].hist(figsize=(15, 12))
plt.suptitle("Histogramet Para Preprocessing")
plt.show()
# HEATMAP PARA PREPROCESSING
plt.figure(figsize=(12, 10))
sns.heatmap(
    df[num_cols].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Heatmap Para Preprocessing")
plt.show()
# BOXPLOT PARA OUTLIERS
plt.figure(figsize=(15, 7))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=90)
plt.title("Boxplot Para Largimit te Outliers")
plt.show()

# OUTLIER DETECTION - IQR METHOD
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# BOXPLOT PAS OUTLIERS
plt.figure(figsize=(15, 7))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=90)
plt.title("Boxplot Pas Largimit te Outliers")
plt.show()

# ENCODING

df = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=True
)

print("\nDATA PAS ONE HOT ENCODING")
print(df.head())
# STANDARDIZATION
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("\nDATA PAS STANDARDIZIMIT")
print(df.head())
# HISTOGRAMS PAS STANDARDIZATION
df[num_cols].hist(figsize=(15, 12))
plt.suptitle("Histogramet Pas Standardizimit")
plt.show()
# HEATMAP PAS PREPROCESSING
plt.figure(figsize=(12, 10))
sns.heatmap(
    df[num_cols].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Heatmap Pas Preprocessing")
plt.show()

# COMPARISON CHARTS

plt.figure(figsize=(18, 7))
# PARA
plt.subplot(1, 2, 1)
sns.boxplot(data=df_original.select_dtypes(include=['int64', 'float64']))
plt.xticks(rotation=90)
plt.title("Para Preprocessing")
# PAS
plt.subplot(1, 2, 2)
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=90)
plt.title("Pas Preprocessing")
plt.show()

# FINAL MISSING VALUES CHECK

print("\nFINAL MISSING VALUES")
print(df.isnull().sum())

# PCA VISUALIZATION

target_cols = [col for col in df.columns if "Heart" in col]
if len(target_cols) > 0:
    features = df.drop(columns=target_cols)
else:
    features = df.copy()
pca = PCA(n_components=2)

pca_result = pca.fit_transform(features)
pca_df = pd.DataFrame(
    data=pca_result,
    columns=['PCA1', 'PCA2']
)

plt.figure(figsize=(10, 7))

plt.scatter(
    pca_df['PCA1'],
    pca_df['PCA2']
)

plt.title("PCA Visualization")

plt.xlabel("PCA1")

plt.ylabel("PCA2")

plt.show()

# FINAL DATASET

print("\nFINAL DATASET")

print(df.shape)

print(df.head())

# SAVE FINAL DATASET

df.to_csv(
    "heart_cleaned.csv",
    index=False
)