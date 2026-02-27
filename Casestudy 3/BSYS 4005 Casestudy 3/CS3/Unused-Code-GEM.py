import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD & CLEAN (The boring part) ---
print("Loading data...")
df = pd.read_csv('SantaClaraListings.csv')

# We only care about Money and Satisfaction
data = df[['id', 'price', 'review_scores_rating', 'number_of_reviews', 'latitude', 'longitude']].copy()

# Remove the '$' and ',' from price so the computer can read it
data['price'] = data['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

# Drop rows where data is missing or empty (listings with 0 reviews tell us nothing)
data = data.dropna()
data = data[data['number_of_reviews'] > 0] 
data = data[data['price'] < 1000] # Ignore $5000/night mansions

print(f"Data cleaned. Analyzing {len(data)} active listings.")

# --- 2. THE AI MODEL (Clustering) ---
# We are grouping them based on PRICE and RATING only.
features = data[['price', 'review_scores_rating']]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features)

# Find 3 groups (Low, Medium, High)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# --- 3. GENERATE THE IMAGES (Save these for your report) ---

# IMAGE 1: The Segmentation (Shows the groups clearly)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['price'], y=data['review_scores_rating'], hue=data['Cluster'], palette='viridis', s=50)
plt.title('The 3 Segments of Santa Clara Market')
plt.xlabel('Nightly Price ($)')
plt.ylabel('Guest Rating (0-5)')
plt.grid(True, alpha=0.3)
plt.savefig('segmentation_chart.png') # Saves to your folder
print("\n[Generated 'segmentation_chart.png' for your report]")

# IMAGE 2: The Map (Where should we buy?)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['longitude'], y=data['latitude'], hue=data['Cluster'], palette='viridis', s=20)
plt.title('Location of Investment Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('location_map.png') # Saves to your folder
print("[Generated 'location_map.png' for your report]")

# --- 4. THE HUMAN-READABLE REPORT OUTPUT ---
# This calculates the averages for you
summary = data.groupby('Cluster')[['price', 'review_scores_rating', 'number_of_reviews']].mean().round(2)
count = data['Cluster'].value_counts()

print("\n" + "="*50)
print("   COPY THIS TABLE INTO YOUR WORD DOC   ")
print("="*50)
print(summary)
print("\nCluster Counts (Size of market):")
print(count)
print("="*50)

# Automatic Insight Generator (Read this to write your 'Insights')
print("\n--- QUICK INSIGHTS FOR HIMANISH ---")
for i in summary.index:
    avg_price = summary.loc[i, 'price']
    avg_rating = summary.loc[i, 'review_scores_rating']
    print(f"Cluster {i}: Average Price is ${avg_price} with a Rating of {avg_rating}/5.0")