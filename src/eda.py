import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import entropy

train = pd.read_csv('data/train.csv')

#train.info() # 결측치 없고, 다 float64

train_x = train.drop(columns=['ID', 'target'])
train_y = train['target']

## 다중공선성 확인
# x_const = add_constant(train_x)

# vif = pd.DataFrame()
# vif["Feature"] = train_x.columns
# vif["VIF"] = [variance_inflation_factor(x_const.values, i+1) for i in range(len(train_x.columns))]

# # VIF 높은 상위 10개만 보기
# vif_sorted = vif.sort_values("VIF", ascending=False).head(10)
# print(vif_sorted)


## 상관계수 확인
# corr = train.corr(numeric_only=True)

# # 절대 상관계수가 0.8 이상인 변수들만 필터링
# high_corr = corr[(corr.abs() > 0.8) & (corr.abs() < 1.0)]

# # 상관관계 있는 변수쌍만 추출
# pairs = []
# for i in range(len(high_corr.columns)):
#     for j in range(i):
#         if abs(high_corr.iloc[i,j]) > 0.8:
#             pairs.append((high_corr.index[i], high_corr.columns[j], high_corr.iloc[i,j]))

# df = pd.DataFrame(pairs, columns=["Feature1","Feature2","Correlation"])
# print(df)


## 분포 확인(쿨백-라이블러)
def to_distribution(series, bins=50):
    hist, bin_edges = np.histogram(series.dropna(), bins=bins, density=True)
    hist = hist + 1e-9  # 0 방지
    hist = hist / hist.sum()  # 정규화
    return hist

# 3. KL divergence 계산
features = train_x.columns
n = len(features)
kl_matrix = np.zeros((n, n))

for i in range(n):
    p = to_distribution(train_x[features[i]])
    for j in range(n):
        q = to_distribution(train_x[features[j]])
        # 대칭 KL (Jensen-Shannon을 써도 됨)
        kl_pq = entropy(p, q)
        kl_qp = entropy(q, p)
        kl_matrix[i, j] = (kl_pq + kl_qp) / 2

# 4. Heatmap 시각화
# plt.figure(figsize=(14,10))
# sns.heatmap(kl_matrix, xticklabels=features, yticklabels=features, cmap="viridis")
# plt.title("Symmetric KL Divergence between Features")
# plt.show()

threshold = 10  # 예: KL 값이 10 이상인 경우만 추출

pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):  # 대각선 제외
        if kl_matrix[i, j] > threshold:
            pairs.append((features[i], features[j], kl_matrix[i, j]))

kl_pairs = pd.DataFrame(pairs, columns=["Feature1", "Feature2", "KL_value"])
#kl_pairs.sort_values("KL_value", ascending=False, inplace=True)

#print(kl_pairs.head(30))  # 상위 20개만 확인

selected_features = set(kl_pairs["Feature1"]).union(set(kl_pairs["Feature2"]))
print("선택된 변수 개수:", len(selected_features))
print("선택된 변수:", selected_features)