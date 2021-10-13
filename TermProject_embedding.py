import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
import xgboost as xgb
import lightgbm as lgb
import re
import optuna

sns.set_theme(style="darkgrid")

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

#괄호와 괄호안내용 제거
print('아파트 괄호 제거 중')
regex = "\(.*\)|\s-\s.*"
for i in tqdm(range(len(train_df))):
   train_df.at[i, 'apt'] = re.sub(regex, '', train_df.at[i, 'apt'])
for i in tqdm(range(len(test_df))):
   test_df.at[i, 'apt'] = re.sub(regex, '', test_df.at[i, 'apt'])   
print('괄호 제거 완료')

# top 10 시공사 아파트 여부를 나타내는 컬럼 생성
train_df['top10'] = 0
test_df['top10'] = 0
top10 = ['자이', '푸르지오', '더샵', '롯데캐슬', '이편한|e편한|e-편한',
         '힐스테이트', '아이파크', '래미안', 'sk|SK|에스케이', '데시앙']

train_df['apt'] = train_df['apt'].fillna('others')
# top 10 시공사면 1, 아니면 0
for i, brand in enumerate(top10):
    train_df.loc[train_df['apt'].str.contains(brand), 'top10'] = 1
    test_df.loc[test_df['apt'].str.contains(brand), 'top10'] = 1

print('apt 임베딩')
vector_size = 256
epochs = 100
apt_emb = FastText.load(f'apt_ft_{vector_size}_sg_{epochs}.model')
# apt_emb = Word2Vec.load(f'apt_w2v_{vector_size}_sg_{epochs}.model')
train_df['apt_embedded'] = train_df['apt'].apply(lambda x: apt_emb.wv[x])

# test 시작 거래연월인 인덱스 저장
test_start = train_df.loc[train_df['transaction_year_month'] == 201701, 'transaction_year_month'].index[0]

# 완공연도에서 최소연도를 뺌으로써 완공연도 라벨인코딩
print('변환전\n', train_df['year_of_completion'].unique()[:5])
train_df['year_of_completion'] = train_df['year_of_completion'] - train_df['year_of_completion'].min()
test_df['year_of_completion'] = test_df['year_of_completion'] - test_df['year_of_completion'].min()
print('변환후\n', train_df['year_of_completion'].unique()[:5])

# 연월 증가하는 순으로 라벨 인코딩
print('train 변환전\n', train_df['transaction_year_month'].unique()[:5])
print('test 변환전\n', test_df['transaction_year_month'].unique()[:5])
le = LabelEncoder()
train_df['transaction_year_month'] = le.fit_transform(train_df['transaction_year_month'])
# test는 다음과 같이 처리
test_df['transaction_year_month'] = test_df['transaction_year_month'] - test_df['transaction_year_month'].min() + train_df.at[test_start, 'transaction_year_month']
print('train 변환후\n', train_df['transaction_year_month'].unique()[:5])
print('test 변환후\n', test_df['transaction_year_month'].unique()[:5])

# 필요없는 열 제거
train_df = train_df.drop(['jibun', 'transaction_date', 'addr_kr'], axis=1)
test_df = test_df.drop(['jibun', 'transaction_date', 'addr_kr'], axis=1)


seoul_set = set(train_df.loc[train_df['city']=='서울특별시', 'dong'])
busan_set = set(train_df.loc[train_df['city']=='부산광역시', 'dong'])
same_dong = seoul_set & busan_set 
for d in same_dong:
    train_df.loc[(train_df['city']=='서울특별시') & (train_df['dong']==d), 'dong'] = '서울' + d
    train_df.loc[(train_df['city']=='부산광역시') & (train_df['dong']==d), 'dong'] = '부산' + d
    test_df.loc[(test_df['city']=='서울특별시') & (test_df['dong']==d), 'dong'] = '서울' + d
    test_df.loc[(test_df['city']=='부산광역시') & (test_df['dong']==d), 'dong'] = '부산' + d
    

seoul_set = set(train_df.loc[train_df['city']=='서울특별시', 'dong'])
busan_set = set(train_df.loc[train_df['city']=='부산광역시', 'dong'])
same_dong = seoul_set & busan_set

print('dong 임베딩')
vector_size = 256
epochs = 100
# dong_emb = Word2Vec.load(f'dong_w2v_{vector_size}_sg_{epochs}.model')
dong_emb = FastText.load(f'dong_ft_{vector_size}_sg_{epochs}.model')
train_df['dong_embedded'] = train_df['dong'].apply(lambda x: dong_emb.wv[x])
train_df.head()

# 최소값이 -4이므로 4를 더해서 음수를 없애고 순서형범주처리
print('변환전\n', train_df['floor'].values[:5])
train_df['floor'] = train_df['floor'].map(lambda x: x+4)
test_df['floor'] = test_df['floor'].map(lambda x: x+1)
print('변환후\n', train_df['floor'].values[:5])

# 가격 로그 변환 후 원래 가격 따로 저장
train_df['log_price'] = np.log1p(train_df['transaction_real_price'])
real_price = train_df['transaction_real_price'] # 원래 가격
train_df.drop('transaction_real_price', axis=1, inplace=True)

# 면적 로그 변환 후 원래 면적 따로 저장
train_df['log_area'] = np.log1p(train_df['exclusive_use_area'])
test_df['log_area'] = np.log1p(test_df['exclusive_use_area'])
area = train_df['exclusive_use_area'] # 원래 가격
train_df.drop('exclusive_use_area', axis=1, inplace=True)
test_df.drop('exclusive_use_area', axis=1, inplace=True)

drop_col = ['transaction_id', 'apartment_id', 'dong', 'apt']

train_df['city'] = train_df['city'].map(lambda x: 1 if x == '서울특별시' else 0)
test_df['city'] = test_df['city'].map(lambda x: 1 if x == '서울특별시' else 0)

train_df.drop(drop_col, axis=1, inplace=True)
test_df.drop(drop_col, axis=1, inplace=True)

emb_apt = np.array(train_df['apt_embedded'].to_numpy().tolist())
emb_dong = np.array(train_df['dong_embedded'].to_numpy().tolist())

X_common = train_df.drop(['apt_embedded', 'dong_embedded'], axis=1)
embedding = np.concatenate([emb_apt, emb_dong], axis=1)
train_X = np.concatenate([X_common.drop('log_price', axis=1).to_numpy(), embedding], axis=1)
train_y = train_df['log_price'].to_numpy()

def RMSE(y, y_pred):
    rmse = mean_squared_error(y, y_pred) ** 0.5
    return rmse

cut = int(len(train_df)*0.8)
h_train_X = train_X[:cut]
h_train_y = train_y[:cut]
h_valid_X = train_X[cut:]
h_valid_y = train_y[cut:]

print(h_train_X.shape, h_train_y.shape, h_valid_X.shape, h_valid_y.shape)

from optuna.samplers import TPESampler

sampler = TPESampler(seed=10)

def objective(trial):
    dtrain = lgb.Dataset(h_train_X, label=h_train_y)
    dtest = lgb.Dataset(h_valid_X, label=h_valid_y)

    param = {
        'objective': 'regression', # 회귀
        'verbose': -1,
        'num_threads': 6,
        'device': 'gpu',
        'metric': 'rmse', 
        'max_depth': trial.suggest_int('max_depth',3, 15),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
    }

    model = lgb.LGBMRegressor(**param)
    lgb_model = model.fit(h_train_X, h_train_y, eval_set=[(h_valid_X, h_valid_y)], verbose=0, early_stopping_rounds=25)
    rmse = RMSE(h_valid_y, lgb_model.predict(h_valid_X))
    return rmse
        
study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
study_lgb.optimize(objective, n_trials=100)

trial = study_lgb.best_trial
trial_params = trial.params
print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))