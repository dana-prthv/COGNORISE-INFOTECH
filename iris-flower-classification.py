import pandas as pd
from sklearn.model_selection import train_test_split


# Load the dataset
iris_data = pd.read_csv('C:\\Users\\world\\OneDrive\\Desktop\\INTERNSHIPS\\IRIS.csv')


# Display the first few rows of the dataset
print(iris_data.head())
print("\n\n\n")

# Check for missing values
print(iris_data.info())
print("\n\n\n")

# Statistical summary
print(iris_data.describe())
print("\n\n\n")

# Separate features and labels
X = iris_data.drop('species', axis=1)  # Features (sepal length, sepal width, petal length, petal width)
y = iris_data['species']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Model Training and Evaluation
from sklearn.tree import DecisionTreeClassifier


# Initialize the classifier
clf = DecisionTreeClassifier()


# Train the model
clf.fit(X_train, y_train)


# Predict on the test set
predictions = clf.predict(X_test)


# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, predictions))
print("\n\n\n")
print("Classification Report:\n", classification_report(y_test, predictions))


# Code snippet after model evaluation

# Unique classes predicted by the classifier
predicted_classes = clf.classes_

print("\n\n\n")

# Print the names of the classified groups of species
print("Names of Classified Groups of Species:")
for class_name in predicted_classes:
    print(class_name)


# Assuming 'predictions' contains the predicted species
# Create a DataFrame to hold the predicted values and their corresponding IDs
predicted_data = pd.DataFrame({'Predicted_Species': predictions, 'Actual_Species': y_test})

# Group by Predicted_Species and count occurrences, displaying the counts for each species
species_counts = predicted_data.groupby('Predicted_Species').size()

# Display the counts for each species
print("\n\n\n")
print("Species Counts in Classified Groups:")
for species_id, count in species_counts.items():
    print(f"Species ID: {species_id}, Count: {count}")






