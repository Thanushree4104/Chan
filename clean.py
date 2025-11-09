import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load merged dataset again
file_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\merged_libs_dataset.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns if any
df = df.select_dtypes(include=['float64', 'int64'])

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Redo PCA
pca = PCA().fit(X_scaled)

# Plot
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_.cumsum(),
         marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('LIBS PCA Explained Variance')
plt.grid(True)
plt.show()
