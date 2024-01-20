import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



# Load the data from car.csv
filename = 'C:\\Users\\world\\OneDrive\\Desktop\\INTERNSHIPS\\car.csv'
data = pd.read_csv(filename)

# Selecting relevant columns for training
selected_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation','wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype','cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke','compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']

data_selected = data[selected_columns]

# Encoding categorical variables
encoder = OneHotEncoder()
data_encoded = encoder.fit_transform(data_selected[['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem', 'cylindernumber']])
data_processed = pd.concat([pd.DataFrame(data_encoded.toarray()), data_selected.drop(columns=['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem', 'cylindernumber'])], axis=1)
data_processed.columns = data_processed.columns.astype(str)




# Splitting the dataset into training and testing sets
X = data_processed.drop('price', axis=1)
y = data_processed['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model (using Decision Tree Regressor)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predicting prices for the first 5 records
first_5_records = X.head(5)
predicted_prices = model.predict(first_5_records)
actual_prices = y.head(5)

# Plotting the actual vs. predicted prices
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), actual_prices, marker='o', label='Actual Prices')
plt.plot(range(1, 6), predicted_prices, marker='x', label='Predicted Prices')
plt.xlabel('Car Record')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Car Prices')
plt.legend()
plt.xticks(range(1, 6))
plt.grid()
plt.show()
