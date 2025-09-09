import os, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# 디렉토리
os.makedirs('data/models_lgbm', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# --- 데이터/라벨 ---
train = pd.read_csv('data/train.csv'); test = pd.read_csv('data/test.csv')
y_raw = train['target'].astype(int).values
classes = np.sort(np.unique(y_raw))
class2id = {c:i for i,c in enumerate(classes)}
id2class = classes
y = np.array([class2id[v] for v in y_raw], dtype=np.int32)

X      = train.drop(columns=['ID','target']).astype(np.float32)
X_test = test.drop(columns=['ID']).astype(np.float32)

# --- 전처리 + 메타 저장 ---
q_lo, q_hi = X.quantile(0.005), X.quantile(0.995)
X      = X.clip(q_lo, q_hi, axis=1)
X_test = X_test.clip(q_lo, q_hi, axis=1)

with open('data/models_lgbm/meta.json','w') as f:
    json.dump({
        'classes': classes.tolist(),
        'q_lo': q_lo.to_dict(),
        'q_hi': q_hi.to_dict(),
        'feature_order': X.columns.tolist()
    }, f)

# --- 파라미터 ---
params_base = {
    'objective':'multiclass','num_class':len(classes),'metric':'multi_logloss',
    'device':'gpu','boosting_type':'gbdt',
    'learning_rate':0.1,'num_leaves':63,'max_depth':-1,
    'min_data_in_leaf':20,'min_sum_hessian_in_leaf':1e-3,
    'feature_fraction':1.0,'bagging_fraction':1.0,'bagging_freq':0,
    'lambda_l1':0.0,'lambda_l2':0.0,'verbosity':-1,'num_threads':-1,
}

num_boost_round = 4000
early_stopping_rounds = 200
seeds = [42, 2025, 1337]

oof_prob  = np.zeros((len(X), len(classes)), dtype=np.float32)
test_prob = np.zeros((len(X_test), len(classes)), dtype=np.float32)
all_logs = []
gpu_ok = True

for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr])
        dva = lgb.Dataset(X.iloc[va], label=y[va])

        params = params_base.copy()
        params['seed'] = seed
        if not gpu_ok:
            params['device'] = 'cpu'

        eval_rec = {}
        try:
            model = lgb.train(
                params, dtr, num_boost_round,
                valid_sets=[dva], valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.record_evaluation(eval_rec),   # 에폭별 로그 기록
                    # lgb.log_evaluation(0),           # 콘솔 로그 완전 끄기 원하면 주석 해제
                ],
            )
        except Exception as e:
            if gpu_ok:
                print(f"[Fold {fold}] GPU 실패 → CPU로 재시도: {e}")
                gpu_ok = False
                params['device'] = 'cpu'
                model = lgb.train(
                    params, dtr, num_boost_round,
                    valid_sets=[dva], valid_names=['valid'],
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds, verbose=False),
                        lgb.record_evaluation(eval_rec),
                    ],
                )
            else:
                raise

        # 모델 저장(시드+폴드로 파일명 고유화, best_iteration만 저장)
        model.save_model(f'data/models_lgbm/lgbm_seed{seed}_fold{fold}.txt',
                         num_iteration=model.best_iteration)

        # OOF/TEST 누적(시드 평균 & 폴드 평균)
        oof_prob[va]  += model.predict(X.iloc[va], num_iteration=model.best_iteration) / len(seeds)
        test_prob     += model.predict(X_test,      num_iteration=model.best_iteration) / (len(seeds) * skf.n_splits)

        # 에폭별 로그 축적
        valid_name = 'valid'
        # metric 이름을 안전하게 가져오기 (혹시 metric을 바꿀 때 대비)
        metric_name = next(iter(eval_rec[valid_name].keys()))
        vals = eval_rec[valid_name][metric_name]
        n = len(vals)

        hist = pd.DataFrame({
            'seed':  [seed] * n,
            'fold':  [fold] * n,
            'iter':  np.arange(n),
            f'{valid_name}_{metric_name}': vals,
        })

        # best_iteration / best_score 는 스칼라 → float로 꺼내서 채우기
        best_score_val = float(model.best_score[valid_name][metric_name])
        hist['best_iteration'] = int(model.best_iteration)
        hist['best_score']     = best_score_val

        all_logs.append(hist)

# 로그 저장
log_df = pd.concat(all_logs, ignore_index=True)
log_df.to_csv('logs/lgbm_train_log.csv', index=False, encoding='utf-8-sig')

# OOF 점수
print('OOF macro F1:', f1_score(y, oof_prob.argmax(1), average='macro'))

# 제출 저장(원라벨 역매핑)
sub = pd.read_csv('data/sample_submission.csv')[['ID']]
pred_lab = id2class[test_prob.argmax(1)]
submission = sub.merge(pd.DataFrame({'ID': test['ID'], 'target': pred_lab}), on='ID', how='left')
submission.to_csv('data/lgbm_oof_seedavg_submit.csv', index=False, encoding='utf-8-sig')

print('Saved models → data/models_lgbm/')
print('Saved logs   → logs/lgbm_train_log.csv')
print('Saved submit → data/lgbm_oof_seedavg_submit.csv')
