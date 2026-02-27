# BSYS 4005 APAI | Case Study 3: Segmentation - Santa Clara AirBnbs
# Group 3: Riya, Nilan, Randy, Himanish
# → 
import numpy as np #math
import pandas as pd #data
import matplotlib.pyplot as plt #plot
import seaborn as sns #visual 
from sklearn.cluster import KMeans #clustering
from sklearn.preprocessing import StandardScaler #feature scale
from sklearn.metrics import silhouette_score

print("Loading data...")
print("...")
df = pd.read_csv('SantaClaraListings.csv')

# --- 1. Setup & Clean
# columns selection-only relevant to bp ones
data = df[[
    'id',
    'price',
    'review_scores_rating',
    'number_of_reviews',
    'latitude',
    'longitude'
]].copy()

# convert price from $123.00 to numeric 123.00
data['price'] = (
    data['price']
    .astype(str)                    # ensure text format
    .str.replace(r'[$,]', '', regex=True)  # remove symbols
    .astype(float)                  # convert to number
)
#removing the missing data rows
data = data.dropna()
#removing listings with no review (kinda useless to us)
data = data[data['number_of_reviews'] > 0]
# high priced ones = not relevant to business problem (bp)
data = data[data['price'] <1000]
print(f"Cleaning Finished - Analyzing {len(data)} listings!")

# --- 2. Feature "Engineering" 
#log price - smoothing the extreme differences in price (AI suggestion for this)
data['log_price'] = np.log1p(data['price'])
# Demand score = how active/popular a listing is
data['demand_score'] = np.log1p(data['number_of_reviews'])
# Value efficiency = rating relative to price
data['value_score'] = data['review_scores_rating'] / data['price']

# --- 3. Clustering Model
# features representing cost + quality + demand
features = data[[
    'log_price', 'review_scores_rating', 'demand_score', 'value_score']]

#scaling - too many metrics, easy to overshadow smth like ratings as the scales only 1-5
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ! The KMeans Clustering Model (with 3 market segments)
kmeans = KMeans(n_clusters=3, random_state=27, n_init=17)
# Assign each listing to a cluster
data['Cluster'] = kmeans.fit_predict(scaled_features)

score = silhouette_score(scaled_features, data['Cluster'])
print(f"\nModel Performance (Silhouette Score): {score:.3f}")
print("(Score > 0.4 indicates strong structure. > 0.5 is excellent.)")

# --- 4. Visualizer
plt.figure(figsize=(12, 8)) #how big is it
# price vs demand colored by clusted
sns.scatterplot(
    x=data['price'],
    y=data['demand_score'],
    hue=data['Segment_Name'],
    palette='viridis', #tried mako but thats just invis lol
    s=150,
    alpha=0.8,
    edgecolor='w'
)

plt.title('Price vs Demand (Investment Landscape)')
plt.xlabel('Pricing ($)')
plt.ylabel('Demand Score (Review Activity)')
plt.grid(True, alpha=0.5)
# Save image for report
plt.savefig('CS3-SegmentationChart.png')
print("\nSaved Segmentation Chart.png")

# --- 4.1 Visual - Map
plt.figure(figsize=(10, 8))

# Map cluster locations
sns.scatterplot(
    x=data['longitude'],
    y=data['latitude'],
    hue=data['Segment_Name'],
    palette='viridis',
    s=60
)
plt.title('Location of Market Segments')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('CS-LocationMap.png')
print("Saved: Location Map.png")

# --- 5 Summary Table - AI Suggestion

# average characteristics of each cluster
summary = data.groupby('Cluster')[[
    'price',
    'review_scores_rating',
    'number_of_reviews',
    'value_score'
]].mean().round(2)
# listings per cluster
counts = data['Cluster'].value_counts()
print("\n" + "="*60)
print("CLUSTER SUMMARY")
print("="*60)
print(summary)
print("\nCluster Sizes:")
print(counts)
print("="*60)

# --- 5.1 "Interpretation Guide" AI Suggestion

print("Investment Interpretation Guide")
for i in summary.index:
    row = summary.loc[i]
    print(f"\nCluster {i}:")
    print(f"Average price: ${row['price']}")
    print(f"Average rating: {row['review_scores_rating']}")
    print(f"Demand level (reviews): {row['number_of_reviews']}")
    print(f"Value efficiency: {row['value_score']:.4f}")

print("\n Succesful.")

# "Legend/Naming"
cluster_names = {
    0: 'Budget / High-Volume',
    1: 'Luxury / Premium',
    2: 'Mid-Tier / Stable'
}
data['Segment_Name'] = data['Cluster'].map(cluster_names)
