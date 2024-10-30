# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the dataset with the correct encoding
data = pd.read_csv('C:\\Users\\LC\\Downloads\\DS_Case_Study_beer-ratings_2020 (2)\\DS_Case_Study_beer-ratings_2020\\train.csv', encoding='utf-8')

# Initial data exploration
print(data.info())
print(data.describe())

# Exploratory Data Analysis (EDA)
sns.histplot(data['beer/ABV'], kde=True)
plt.show()

# Preprocessing & Feature Engineering
data['review_text_length'] = data['review/text'].fillna("").apply(lambda x: len(str(x)))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=500)
# Fill NaN values in 'review/text' with an empty string
data['review/text'] = data['review/text'].fillna("")
tfidf_matrix = vectorizer.fit_transform(data['review/text'])

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Concatenate original DataFrame with TF-IDF features
X = pd.concat([data.drop(columns=['review/overall', 'review/text']), tfidf_df], axis=1)

# Check the data types of the features
print(X.dtypes)

# Check for any string types or object types in the features
non_numeric_columns = X.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

# If 'beer/name' or similar exists, drop it
X = X.drop(columns=non_numeric_columns, errors='ignore')  # Drop non-numeric columns

y = data['review/overall']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))
print("Shape of X after dropping non-numeric columns:", X.shape)
print(X.head())

