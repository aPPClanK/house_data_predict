#From Kaggle course


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
file_path = 'train.csv'

home_data = pd.read_csv(file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
print(X.describe())
print(y.describe())
# Split into validation and training data
for test_s in [0.2, 0.3, 0.35]:
    for random_s in [1,2,3]:
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=random_s, test_size=test_s)

        # Specify DecisionTree Model
        dt_model = DecisionTreeRegressor(random_state=1)
        # Fit Model
        dt_model.fit(train_X, train_y)

        # Make validation predictions and calculate mean absolute error
        val_predictions = dt_model.predict(val_X)
        val_mae = mean_absolute_error(val_predictions, val_y)
        print("Validation with DecisionTree: {:,.0f} (test_size: {}, random_state: {})".format(val_mae, test_s,random_s), end="\t")

        # Specify RandomForest Model
        rf_model = RandomForestRegressor(random_state=1)
        rf_model.fit(train_X,train_y)
        
        # Make validation predictions and calculate mean absolute error
        val_rf_model_predictions = rf_model.predict(val_X)
        val_rf_model_mae = mean_absolute_error(val_rf_model_predictions,val_y)
        print("Validation wit RandomForest: {:,.0f} (test_size: {}, random_state: {})".format(val_rf_model_mae, test_s,random_s))
    print("\n")

