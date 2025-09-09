import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

selected_features = ['X_43', 'X_25', 'X_42', 'X_46', 'X_18', 'X_26', 
                     'X_02', 'X_05', 'X_28', 'X_22', 'X_47', 'X_37', 
                     'X_51', 'X_11', 'X_33', 'X_30', 'X_01', 'X_38', 
                     'X_40', 'X_14', 'X_41', 'X_36', 'X_07']

train = pd.read_csv('data/train.csv')

train_x = train[selected_features].to_numpy()
train_y = train['target'].to_numpy()

test = pd.read_csv('data/test.csv')

test_x = test[selected_features].to_numpy()

scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

k=5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(train_x_scaled, train_y)

preds = model.predict(test_x_scaled)
#preds_int = np.clip(np.round(preds), 0, 20).astype(int)

submission = pd.read_csv('data/sample_submission.csv')

submission['target'] = preds

submission.to_csv('data/knn_submit.csv', index=False, encoding='utf-8-sig')

# 모델 + 스케일러 둘 다 저장해야 함
joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")