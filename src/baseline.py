import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('data/train.csv')

train_x = train.drop(columns=['ID', 'target'])
train_y = train['target']

model = RandomForestClassifier(random_state=42)
model.fit(train_x, train_y) 

test = pd.read_csv('data/test.csv')

test_x = test.drop(columns=['ID'])
preds = model.predict(test_x)

submission = pd.read_csv('data/sample_submission.csv')

submission['target'] = preds
submission

submission.to_csv('data/baseline_submit_gyemoo.csv', index=False, encoding='utf-8-sig')