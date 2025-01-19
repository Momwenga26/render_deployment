from flask import Flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

app = Flask(__name__)
df = pd.read_csv("LG_Customer_Data.csv")
df.head()
df.info()
df.describe()
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Scale numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Split the data into training and test datasets
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

# Step 3: Model Development
# For this example, we'll use K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train)

# Assign clusters to the original data based on the full dataset for evaluation purposes
df['Cluster'] = kmeans.predict(scaled_features)


# Step 4: Evaluate Model Performance
silhouette_avg = silhouette_score(X_train, kmeans.labels_)
calinski_harabasz = calinski_harabasz_score(X_train, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')
print(f'Calinski-Harabasz Index: {calinski_harabasz}')

# Step 4.1: K-Validation
k_values = range(2, 11)  # Testing k from 2 to 10
silhouette_scores = []
calinski_harabasz_scores = []

for k in k_values:
    kmeans_test = KMeans(n_clusters=k, random_state=42)
    kmeans_test.fit(X_train)
    labels = kmeans_test.labels_
    silhouette_scores.append(silhouette_score(X_train, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X_train, labels))

# Plot the metrics for different k values
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, silhouette_scores, marker='o', label='Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_values, calinski_harabasz_scores, marker='o', color='r', label='Calinski-Harabasz Index')
plt.title('Calinski-Harabasz Index vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Index')
plt.legend()

plt.tight_layout()
plt.show()

# Step 5: Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=scaled_features[:, 1], y=scaled_features[:, 2],
    hue=df['Cluster'], palette='viridis', legend='full'
)
plt.title('Customer Clusters')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.show()

# Step 6: Save the model for deployment
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler have been saved for deployment.")

