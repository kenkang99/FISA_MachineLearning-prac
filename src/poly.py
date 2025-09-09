import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

selected_features = ['X_43', 'X_25', 'X_42', 'X_46', 'X_18', 'X_26', 
                     'X_02', 'X_05', 'X_28', 'X_22', 'X_47', 'X_37', 
                     'X_51', 'X_11', 'X_33', 'X_30', 'X_01', 'X_38', 
                     'X_40', 'X_14', 'X_41', 'X_36', 'X_07']

train = pd.read_csv('data/train.csv')

train_x = train[selected_features]
train_y = train['target']

#print(train_y.unique())
test = pd.read_csv('data/test.csv')

test_x = test[selected_features]


degree = 23
poly = PolynomialFeatures(degree=degree, include_bias=False)

train_x_poly = poly.fit_transform(train_x)
test_x_poly = poly.transform(test_x)

model = LinearRegression()
model.fit(train_x_poly, train_y) 

preds = model.predict(test_x_poly)
preds_int = np.clip(np.round(preds), 0, 20).astype(int)

submission = pd.read_csv('data/sample_submission.csv')

submission['target'] = preds_int

submission.to_csv('data/poly_submit.csv', index=False, encoding='utf-8-sig')