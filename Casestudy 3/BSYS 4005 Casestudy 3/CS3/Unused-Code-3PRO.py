import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- STEP 1: LOAD & CLEAN ---
# Load data
df = pd.read_csv('SantaClaraListings.csv')

# Keep only the "Value" indicators + Location for plotting
keep_cols = [
    'id', 'price', 'review_scores_rating', 'number_of_reviews', 
    'reviews_per_month', 'availability_365', 'latitude', 'longitude', 
    'neighbourhood_cleansed'
]

data = df[keep_cols].copy()

# CLEANING:
# 1. Fix Price (Remove $ and ,)
data['price'] = data['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

# 2. Drop "Dead" Listings (0 reviews means we can't judge quality)
data = data[data['number_of_reviews'] > 0]

# 3. Drop missing values (Simplicity first)
data.dropna(inplace=True)

# 4. Filter Extremes (Remove >$1000/night mansions and <$20 errors)
data = data[(data['price'] < 1000) & (data['price'] > 20)]

print(f"Working with {len(data)} active listings.")

# --- STEP 2: PREPARE FOR AI ---
# We cluster on these specific "Performance" metrics
# We DO NOT cluster on Lat/Lon (we want to find the pattern first, then see where it lives)
cluster_features = ['price', 'review_scores_rating', 'reviews_per_month', 'availability_365']

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[cluster_features])

# --- STEP 3: RUN K-MEANS ---
# K=3 is usually perfect for "Low/Med/High" or "Good/Bad/Value" analysis
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(data_scaled)

# Assign clusters back to the main dataframe
data['Cluster'] = clusters

# --- STEP 4: ANALYZE THE GROUPS ---
# Group by Cluster and calculate the average of our key metrics
summary = data.groupby('Cluster')[cluster_features].mean().round(2)
# Add a "Count" column so we know how big each group is
summary['count'] = data['Cluster'].value_counts()

print("\n--- CLUSTER SUMMARY (Copy this table for your report) ---")
print(summary)

# --- STEP 5: VISUALIZE (The "Money Shot") ---

# Plot 1: Price vs. Rating (colored by Cluster)
# This proves your segmentation works: You should see distinct blobs.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='price', y='review_scores_rating', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('Segmentation: Price vs. Quality')
plt.xlabel('Nightly Price ($)')
plt.ylabel('Review Rating (0-5)')
plt.axhline(y=4.5, color='r', linestyle='--', label='High Quality Cutoff')
plt.legend()
plt.show()

# Plot 2: The Map (Where are these clusters?)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data, x='longitude', y='latitude', hue='Cluster', palette='viridis', s=40, alpha=0.8)
plt.title('Map of Santa Clara Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()

# --- STEP 6: EXPORT ---
# Save the results so you can open in Excel and filter for specific neighborhoods
data.to_csv('santa_clara_clustered.csv', index=False)