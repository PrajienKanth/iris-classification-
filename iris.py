import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset from a CSV file
# Replace 'path_to_your_file.csv' with the actual path to your Iris dataset file
iris_data = pd.read_csv('iris.csv')

# Remove the 'Id' column if present
if 'Id' in iris_data.columns:
    iris_data.drop('Id', axis=1, inplace=True)

# Split the dataset into features and target
X = iris_data.drop('Species', axis=1)  # Features (sepal length, sepal width, petal length, petal width)
y = iris_data['Species']  # Target (species)

# Train the model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Function to classify user input
def classify_user_input():
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    user_input = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(user_input)
    print(f"The predicted species is: {prediction[0]}")

# Continuous classification loop
while True:
    classify_user_input()
    user_choice = input("Do you want to classify another flower? (yes/no): ").lower()
    if user_choice != 'yes':
        break
