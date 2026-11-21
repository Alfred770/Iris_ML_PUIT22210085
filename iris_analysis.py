import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

#load the iris dataset
iris_data= load_iris(as_frame=True)
df= iris_data.frame
df["species"] = df["target"].map(dict(enumerate(iris_data.target_names)))

df=df.drop(columns=["target"])

# Rename columns for easier analysis
df = df.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width"
})


#Display the first five records
print(df.head())

#Display the number of rows and columns
print("Dataset shape:", df.shape)

#Display all column names
print("Column names:", df.columns)

#Display information about the dataset
df.info()

#Check for missing values
print(df.isnull().sum())

#Display statistical information
print(df.describe())

#Display all column names
print(df.columns.tolist())

#Count the number of flowers in each species
print(df['species'].value_counts())

# Count the number of flowers in each species
species_count = df['species'].value_counts()

# Create a bar chart
plt.figure(figsize=(8, 5))
species_count.plot(kind='bar')

# Add chart labels and title
plt.title('Distribution of Iris Flower Species')
plt.xlabel('Flower Species')
plt.ylabel('Number of Flowers')

# Display the graph
plt.show()

# Create a pair plot
sns.pairplot(
    df,
    vars=[
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width'
    ],
    hue='species'
)

# Display the graph
plt.show()

# Select the four original numerical measurements
numerical_data = df[
    ['sepal_length', 'sepal_width',
     'petal_length', 'petal_width']
]

# Calculate correlation
correlation = numerical_data.corr()

# Create heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

# Add title
plt.title('Correlation Heatmap of Iris Dataset')

# Display graph
plt.show()

#Check for missing values
print("Missing values:")
print(df.isnull().sum())

#Check for duplicate records
print("Number of duplicates:", df.duplicated().sum())

#Check data types
print("Data types:")
print(df.dtypes)

#Check minimum and maximum values 
print(df.describe())

#Display duplicate records
print(df[df.duplicated(keep=False)])

#Create a cleaned copy
df_clean= df.drop_duplicates().copy()

#Compare original and cleaned datasets
print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", df_clean.shape)

#Verify duplicates are removed
print("Remaining duplicates:", df_clean.duplicated().sum())

from sklearn.preprocessing import LabelEncoder

# Encode flower species
encoder = LabelEncoder()

df_clean['species_encoded'] = encoder.fit_transform(
    df_clean['species']
)

print(df_clean[['species', 'species_encoded']].drop_duplicates())

# Select features
X = df_clean[
    ['sepal_length', 'sepal_width',
     'petal_length', 'petal_width']
]

# Select target
y = df_clean['species_encoded']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

# Import Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("Model training completed successfully!")

# Make predictions using testing data
y_pred = model.predict(X_test)

print("Predictions completed successfully!")
print(y_pred)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')

plt.show()

from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Convert actual labels into binary format
y_test_binary = label_binarize(y_test, classes=[0, 1, 2])

# Get prediction probabilities
y_prob = model.predict_proba(X_test)

# Calculate multiclass ROC-AUC
roc_auc = roc_auc_score(
    y_test_binary,
    y_prob,
    multi_class="ovr",
    average="macro"
)

print("ROC-AUC Score:", roc_auc)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix Values:")
print(cm)



import joblib

joblib.dump(model, "iris_model.pkl")
joblib.dump(encoder, "iris_encoder.pkl")

print("Model and encoder saved successfully!")