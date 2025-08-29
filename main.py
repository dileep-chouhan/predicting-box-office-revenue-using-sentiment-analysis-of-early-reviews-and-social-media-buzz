import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'MovieTitle': [f'Movie {i}' for i in range(1, num_movies + 1)],
    'EarlyReviewSentiment': np.random.rand(num_movies) * 2 - 1, # -1 to 1 range
    'SocialMediaBuzz': np.random.randint(0, 1000, size=num_movies),
    'BoxOfficeRevenue': np.random.randint(100000, 10000000, size=num_movies)
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data.  
# We could add more realistic data generation in a real-world scenario.
# Example of adding a combined sentiment score:
df['CombinedSentiment'] = (df['EarlyReviewSentiment'] + (df['SocialMediaBuzz']/1000)) /2
# --- 3. Analysis ---
# Simple correlation analysis:
correlation_matrix = df[['EarlyReviewSentiment', 'SocialMediaBuzz', 'BoxOfficeRevenue', 'CombinedSentiment']].corr()
print("Correlation Matrix:\n", correlation_matrix)
# --- 4. Visualization ---
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Sentiment, Buzz, and Box Office Revenue')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
# --- 5. (Illustrative) Predictive Modeling (Simple Linear Regression) ---
# This is a simplified example; a real-world model would be more complex.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df[['CombinedSentiment', 'SocialMediaBuzz']]
y = df['BoxOfficeRevenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# (Not shown: Model evaluation metrics like R-squared, MSE etc.)
# --- 6. Example of Sentiment Analysis on a sample review (Illustrative) ---
sample_review = "This movie was absolutely fantastic!  A must-see."
analysis = TextBlob(sample_review)
polarity = analysis.sentiment.polarity
print(f"\nSentiment Analysis of Sample Review: Polarity = {polarity}") #positive polarity