import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Loading the dataset
titanic_data = pd.read_csv('C:\\Users\\world\\OneDrive\\Desktop\\INTERNSHIPS\\titanic.csv')

print(titanic_data)
print("\n\n\n\n")
print(titanic_data.info())
print("\n\n\n\n")

# Dropping irrelevant columns or those with many missing values
titanic_data.drop(['Cabin', 'Ticket', 'PassengerId', 'Name'], axis=1, inplace=True)

# Handling missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Impute missing values in 'Fare' with mean
imputer = SimpleImputer(strategy='mean')
titanic_data['Fare'] = imputer.fit_transform(titanic_data[['Fare']])

# Impute categorical 'Embarked' missing values with most frequent value
imputer = SimpleImputer(strategy='most_frequent')


#Filling missing values in 'Embarked' with the most frequent value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)




# Converting categorical variables into dummy/indicator variables
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Separating features and target variable
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initializing SimpleImputer to fill missing values with the mean (or any desired strategy)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Continuing with model training using X_train_imputed and X_test_imputed


# Initializing and training the model (Logistic Regression used here)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)



# Assuming 'predictions' contains the predicted values (0 or 1 for not survived and survived)
# Assuming 'y_test' contains the true labels
# Creating a DataFrame to hold the predicted values and their corresponding true labels
predicted_data = pd.DataFrame({'Predicted': predictions, 'Actual': y_test})

# Grouping by 'Predicted' values and count occurrences of each category
survival_counts = predicted_data.groupby('Predicted').size()

# Plotting the bar chart
plt.figure(figsize=(6, 6))
survival_counts.plot(kind='bar', color=['red', 'green'])
plt.xlabel('Survival Prediction')
plt.ylabel('Count')
plt.title('Predicted Survival Counts (0: Not Survived, 1: Survived)')
plt.xticks([0, 1], ['Not Survived', 'Survived'], rotation=0)
plt.show()
