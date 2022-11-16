import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

dataframe = pd.read_csv("C:/Users/12245/Downloads/CC GENERAL.csv")
dataframe.info()

print(dataframe.head())

print(dataframe.describe())

print(dataframe.head())


dataframe.corr().style.background_gradient(cmap="Greens")

dataframe.fillna(dataframe.mean(), inplace=True)
dataframe.isnull().any()

x = dataframe.iloc[:,0:-1]
y = dataframe.iloc[:,-1]


scaler = preprocessing.StandardScaler()
print(scaler.fit(x))
X_scaled_array = scaler.transform(x)
X_scaled_df = pd.DataFrame(X_scaled_array, columns = x.columns)

X_normalized = preprocessing.normalize(X_scaled_df)
# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)

pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(X_normalized)
principalDf = pd.DataFrame(data = principalComponents, columns = ['P1', 'P2'])

finalDf = pd.concat([principalDf, dataframe[['TENURE']]], axis = 1)
print(finalDf.head())

plt.figure(figsize=(7,7))
plt.scatter(finalDf['P1'],finalDf['P2'],c=finalDf['TENURE'],cmap='prism', s =5)
plt.xlabel('pc1')
plt.ylabel('pc2')

ac2 = AgglomerativeClustering(n_clusters=2)

# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(principalDf['P1'], principalDf['P2'],
            c=ac2.fit_predict(principalDf), cmap='rainbow')
plt.show()

ac3 = AgglomerativeClustering(n_clusters=3)

# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(principalDf['P1'], principalDf['P2'],
            c=ac3.fit_predict(principalDf), cmap='rainbow')
plt.show()

ac4 = AgglomerativeClustering(n_clusters=4)

# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(principalDf['P1'], principalDf['P2'],
            c=ac4.fit_predict(principalDf), cmap='rainbow')
plt.show()

ac5 = AgglomerativeClustering(n_clusters=5)

# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(principalDf['P1'], principalDf['P2'],
            c=ac5.fit_predict(principalDf), cmap='rainbow')
plt.show()

k = [2, 3, 4, 5]

# Appending the silhouette scores of the different models to the list
silhouette_scores = []
silhouette_scores.append(
    silhouette_score(principalDf, ac2.fit_predict(principalDf)))
silhouette_scores.append(
    silhouette_score(principalDf, ac3.fit_predict(principalDf)))
silhouette_scores.append(
    silhouette_score(principalDf, ac4.fit_predict(principalDf)))
silhouette_scores.append(
    silhouette_score(principalDf, ac5.fit_predict(principalDf)))

# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize=20)
plt.ylabel('S(i)', fontsize=20)
plt.show()