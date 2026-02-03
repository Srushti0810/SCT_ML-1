from google.colab import files
import zipfile

# Upload file
uploaded = files.upload()

# Extract zip
with zipfile.ZipFile("house-prices-advanced-regression-techniques (1).zip", "r") as zip_ref:
    zip_ref.extractall("house_data")
import pandas as pd

df = pd.read_csv("house_data/train.csv")
df.head()
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

     

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

import pandas as pd

# Example: 2000 sqft, 3 bedrooms, 2 bathrooms
example = [[2000, 3, 2]]
example_df = pd.DataFrame(example, columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predicted_price = model.predict(example_df)
print("Predicted Price:", predicted_price[0])
