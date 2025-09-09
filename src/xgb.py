import os, json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

os.makedirs('data/models_xgb', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# --- 데이터 + 라벨 매핑 ---
train = pd.read_csv('data/train.csv'); test = pd.read_csv('data/test.csv')
y_raw = train['target'].astype(int).values
classes = np.sort(np.unique(y_raw))
class2id = {c:i for i,c in enumerate(classes)}
id2class = classes
y = np.array([class2id[v] for v in y_raw], dtype=np.int32)
X = train.drop(columns=['ID','target']).astype(np.float32)
X_test = test.drop(columns=['ID']).astype(np.float32)

# --- 전처리(분위수 클리핑) + 메타 저장 ---
q_lo, q_hi = X.quantile(0.005), X.quantile(0.995)
X = X.clip(q_lo, q_hi, axis=1)
X_test = X_test.clip(q_lo, q_hi, axis=1)

with open('data/models_xgb/meta.json','w') as f:
    json.dump({
        'classes': classes.tolist(),
        'q_lo': q_lo.to_dict(),
        'q_hi': q_hi.to_dict(),
        'feature_order': X.columns.tolist()
    }, f)

# --- 파라미터/학습 ---
params = {
    'objective':'multi:softprob','num_class':len(classes),'eval_metric':'mlogloss',
    'tree_method':'gpu_hist','predictor':'gpu_predictor',
    'max_depth':6,'min_child_weight':1,'gamma':0.0,
    'subsample':1.0,'colsample_bytree':1.0,
    'lambda':0.0,'alpha':0.0,'eta':0.1,'verbosity':0
}
num_boost_round = 4000
patience = 200

seeds = [42, 2025, 1337]
oof_prob  = np.zeros((len(X), len(classes)), dtype=np.float32)
test_prob = np.zeros((len(X_test), len(classes)), dtype=np.float32)
all_logs = []

for seed in seeds:
    params['seed'] = seed
    # 시드마다 다른 폴드 분할(중요)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        dtr   = xgb.DMatrix(X.iloc[tr], label=y[tr])
        dva   = xgb.DMatrix(X.iloc[va], label=y[va])
        dtest = xgb.DMatrix(X_test)

        evals_result = {}
        es_cb = xgb.callback.EarlyStopping(rounds=patience, save_best=True, maximize=False)

        bst = xgb.train(
            params, dtr, num_boost_round,
            evals=[(dva, 'valid')],
            callbacks=[es_cb],
            evals_result=evals_result
        )

        # 모델 저장 (seed+fold 포함해 덮어쓰기 방지)
        bst.save_model(f'data/models_xgb/xgb_seed{seed}_fold{fold}.json')

        # OOF / TEST 누적 (시드 평균)
        oof_prob[va]  += bst.predict(dva,   iteration_range=(0, bst.best_iteration+1)) / len(seeds)
        test_prob     += bst.predict(dtest, iteration_range=(0, bst.best_iteration+1)) / (len(seeds) * skf.n_splits)

        # 로그 축적
        hist = pd.DataFrame({
            'seed': seed,
            'fold': fold,
            'iter': np.arange(len(evals_result['valid']['mlogloss'])),
            'valid_mlogloss': evals_result['valid']['mlogloss'],
            'best_iteration': bst.best_iteration,
            'best_score': bst.best_score
        })
        all_logs.append(hist)

# 로그 저장
log_df = pd.concat(all_logs, ignore_index=True)
log_df.to_csv('logs/xgb_train_log.csv', index=False, encoding='utf-8-sig')

# OOF 점수 & 제출 저장
print('OOF macro F1:', f1_score(y, oof_prob.argmax(1), average='macro'))

sub = pd.read_csv('data/sample_submission.csv')[['ID']]
# 원라벨 역매핑이 필요하면: pred_lab = id2class[test_prob.argmax(1)]
pred_lab = id2class[test_prob.argmax(1)]
submission = sub.merge(pd.DataFrame({'ID': test['ID'], 'target': pred_lab}), on='ID', how='left')
submission.to_csv('data/xgb_oof_seedavg_submit.csv', index=False, encoding='utf-8-sig')

print('Saved models to data/models_xgb/, logs to logs/xgb_train_log.csv, and submission to data/xgb_oof_seedavg_submit.csv')
