# Домашнее задание по курсу «Машинное обучение в экономике»
# Влияние подключения малого бизнеса к маркетплейсу на месячную выручку фирмы
# Романов Алексей, Галкин Евгений
# https://github.com/banana-hedge-fund/ml-econ-marketplace-hw
#
# Python 3.10+. Установка зависимостей:
#   pip install numpy pandas scikit-learn scipy matplotlib statsmodels doubleml pgmpy catboost
# Запуск: python hw.py  (результаты появятся в папках figures/ и tables/)

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# папки для результатов
figures = Path('figures')
tables = Path('tables')
figures.mkdir(exist_ok=True)
tables.mkdir(exist_ok=True)


def save_table(table, name):
    table.to_csv(tables / name, index=False)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


def make_pipe(model, scale):
    # для kNN и логрегрессии нужно масштабирование, для деревьев нет
    if scale:
        return Pipeline([('scaler', StandardScaler()), ('model', model)])
    return Pipeline([('model', model)])


# ===========================================================================
# РАЗДЕЛ 2. Генерация данных, описательные статистики, разбиение выборки
# ===========================================================================

# параметры процесса генерации данных (см. раздел 2.1)
params = dict(gamma0=-1.2, gamma_z=0.9, gamma_u=0.7, gamma_u2=0.1,
              d1=-0.25, d2=0.35, d3=0.6, d4=0.2, d5=-0.2,
              b0=50, b1=40, b2=15, b3=30, b4=25, b5=20, b6=3, sigma=25,
              t0=35, t1=12, t2=-15, t3=-10, t4=3)


def simulate(n, rng, p=params):
    # U - ненаблюдаемое качество менеджмента (источник эндогенности)
    U = rng.standard_normal(n)
    age = np.exp(rng.normal(1.6, 0.5, n))
    size = np.ceil(np.exp(rng.normal(1.2, 0.6, n))).astype(int)
    online = (rng.random(n) < 1 / (1 + np.exp(-(-0.2 + 0.8 * U)))).astype(int)
    city = (rng.random(n) < 0.5).astype(int)
    Z = (rng.random(n) < 0.5).astype(int)

    # уравнение подключения и потенциальные исходы воздействия
    g = (p['d1'] * np.log(age) + p['d2'] * np.sqrt(size) + p['d3'] * online
         + p['d4'] * city + p['d5'] * online * city)
    index = p['gamma0'] + p['gamma_u'] * U + p['gamma_u2'] * U ** 2 + g + rng.standard_normal(n)
    D0 = (index >= 0).astype(int)
    D1 = (index + p['gamma_z'] >= 0).astype(int)
    D = np.where(Z == 1, D1, D0)

    # уравнение выручки и потенциальные исходы
    mu = (p['b0'] + p['b1'] * np.sqrt(size) + p['b2'] * np.log(age) + p['b3'] * online
          + p['b4'] * city + p['b5'] * U + p['b6'] * U ** 2)
    tau = p['t0'] + p['t1'] * U + p['t2'] * online + p['t3'] * city + p['t4'] * np.log(age)
    noise = rng.standard_normal(n)
    Y0 = mu + p['sigma'] * noise
    Y1 = mu + tau + p['sigma'] * noise
    Y = np.where(D == 1, Y1, Y0)

    return pd.DataFrame(dict(Y=Y, D=D, Z=Z, age=age, size=size, online=online, city=city,
                             U=U, D0=D0, D1=D1, Y0=Y0, Y1=Y1, tau=tau))


data = simulate(10000, np.random.default_rng(20260606))
observed = ['Y', 'D', 'Z', 'age', 'size', 'online', 'city']   # без U
controls = ['age', 'size', 'online', 'city']

# 2.2 описательные статистики
continuous = ['Y', 'age', 'size']
binary = ['D', 'Z', 'online', 'city']

desc_cont = []
for col in continuous:
    x = data[col]
    desc_cont.append([col, round(x.mean(), 2), round(x.std(), 2),
                      round(x.median(), 2), round(x.min(), 2), round(x.max(), 2)])
save_table(pd.DataFrame(desc_cont, columns=['Переменная', 'Среднее', 'Ст.откл.', 'Медиана', 'Мин', 'Макс']),
           'desc_cont.csv')

desc_bin = []
for col in binary:
    desc_bin.append([col, round(data[col].mean(), 3), int(data[col].sum())])
save_table(pd.DataFrame(desc_bin, columns=['Переменная', 'Доля единиц', 'Кол-во единиц']), 'desc_bin.csv')

# корреляционная матрица наблюдаемых переменных
corr = data[observed].corr()
corr.to_csv(tables / 'corr.csv')
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(observed)), observed, rotation=45, ha='right')
plt.yticks(range(len(observed)), observed)
for i in range(len(observed)):
    for j in range(len(observed)):
        plt.text(j, i, round(corr.iloc[i, j], 2), ha='center', va='center', fontsize=8)
plt.title('Корреляционная матрица наблюдаемых переменных')
plt.tight_layout()
plt.savefig(figures / 'corr.png', dpi=150)
plt.close()

# 2.3 разбиение на обучающую и тестовую выборки (25% тест)
all_index = np.arange(len(data))
train_idx, test_idx = train_test_split(all_index, test_size=0.25, random_state=42, stratify=data['D'])
print('Раздел 2: N =', len(data), ', train =', len(train_idx), ', test =', len(test_idx))


# ===========================================================================
# РАЗДЕЛ 3. Классификация переменной воздействия D (признаки без Y и без U)
# ===========================================================================

clf_features = ['Z', 'age', 'size', 'online', 'city']
X_clf = data[clf_features].values
y_clf = data['D'].values
X_train, X_test = X_clf[train_idx], X_clf[test_idx]
y_train, y_test = y_clf[train_idx], y_clf[test_idx]
cv5 = StratifiedKFold(5, shuffle=True, random_state=1)

# модели и сетки гиперпараметров: (модель, сетка, нужно ли масштабирование)
classifiers = {
    'LogReg': (LogisticRegression(max_iter=1000), {'model__C': [0.1, 1, 10]}, True),
    'kNN': (KNeighborsClassifier(), {'model__n_neighbors': [15, 31, 51]}, True),
    'NaiveBayes': (GaussianNB(), {'model__var_smoothing': [1e-9, 1e-7]}, True),
    'RandomForest': (RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                     {'model__max_depth': [None, 12], 'model__min_samples_leaf': [1, 10]}, False),
    'GradBoost': (GradientBoostingClassifier(random_state=0),
                  {'model__n_estimators': [200], 'model__max_depth': [2, 3], 'model__learning_rate': [0.1]}, False),
}

# 3.2 точность по умолчанию и после подбора гиперпараметров (по accuracy)
clf_rows = []
best_models = {}
test_proba = {}
for name, (model, grid, scale) in classifiers.items():
    base = make_pipe(model, scale)
    base.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, base.predict(X_train))
    acc_test = accuracy_score(y_test, base.predict(X_test))
    acc_cv = cross_val_score(base, X_train, y_train, cv=cv5, scoring='accuracy').mean()

    search = GridSearchCV(make_pipe(model, scale), grid, cv=cv5, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    best_models[name] = best
    test_proba[name] = best.predict_proba(X_test)[:, 1]

    best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    clf_rows.append([name, round(acc_train, 3), round(acc_cv, 3), round(acc_test, 3),
                     str(best_params), round(search.best_score_, 3),
                     round(accuracy_score(y_test, best.predict(X_test)), 3)])
save_table(pd.DataFrame(clf_rows, columns=['Метод', 'Acc train', 'Acc CV', 'Acc test',
                                           'Лучшие гиперпарам.', 'CV acc (тюн.)', 'Acc test (тюн.)']),
           'clf_tuning.csv')

# 3.3 тот же подбор, но по альтернативному критерию AUC
auc_rows = []
for name, (model, grid, scale) in classifiers.items():
    search = GridSearchCV(make_pipe(model, scale), grid, cv=cv5, scoring='roc_auc', n_jobs=-1)
    search.fit(X_train, y_train)
    best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    auc_test_value = roc_auc_score(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])
    auc_rows.append([name, str(best_params), round(search.best_score_, 3), round(auc_test_value, 3)])
save_table(pd.DataFrame(auc_rows, columns=['Метод', 'Лучшие гиперпарам.(AUC)', 'CV AUC', 'AUC test']),
           'clf_auc_tuning.csv')

# 3.3 (повышенная сложность) свой критерий качества - ожидаемая прибыль таргетинга
gain, cost = 10.0, 3.0
thresholds_grid = np.linspace(0.05, 0.95, 91)


def profit_linear(y_true, proba, t):
    pred = proba >= t
    tp = np.sum(pred & (y_true == 1))
    fp = np.sum(pred & (y_true == 0))
    return (gain - cost) * tp - cost * fp


def profit_criterion(y_true, proba):
    proba = np.asarray(proba)
    if proba.ndim > 1:
        proba = proba[:, -1]
    return max(profit_linear(y_true, proba, t) for t in thresholds_grid)


profit_scorer = make_scorer(profit_criterion, response_method='predict_proba')
rf_grid = {'model__max_depth': [None, 12], 'model__min_samples_leaf': [1, 10]}
search_acc = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1), False),
                          rf_grid, cv=cv5, scoring='accuracy', n_jobs=-1)
search_acc.fit(X_train, y_train)
search_profit = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1), False),
                             rf_grid, cv=cv5, scoring=profit_scorer, n_jobs=-1)
search_profit.fit(X_train, y_train)
crit_rows = [
    ['accuracy', str({k.replace('model__', ''): v for k, v in search_acc.best_params_.items()}),
     round(search_acc.best_score_, 3)],
    ['собств. критерий (прибыль)', str({k.replace('model__', ''): v for k, v in search_profit.best_params_.items()}),
     round(search_profit.best_score_, 1)],
]
save_table(pd.DataFrame(crit_rows, columns=['Критерий тюнинга', 'Лучшие гиперпарам. RF', 'Значение CV-критерия']),
           'custom_criterion.csv')

# 3.2 (повышенная сложность) подбор случайного леса по OOB-ошибке
oob_rows = []
for depth in [None, 6, 12]:
    for leaf in [1, 5, 20]:
        rf = RandomForestClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                                    oob_score=True, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        oob_rows.append([str(depth), leaf, round(rf.oob_score_, 4),
                         round(accuracy_score(y_test, rf.predict(X_test)), 4)])
save_table(pd.DataFrame(oob_rows, columns=['max_depth', 'min_samples_leaf', 'OOB acc', 'Test acc']), 'rf_oob.csv')

# 3.10 (повышенная сложность) дополнительный метод вне scikit-learn - CatBoost
from catboost import CatBoostClassifier

catboost = CatBoostClassifier(verbose=0, random_seed=0)
catboost.fit(X_train, y_train)
catboost_grid = {'depth': [4, 6], 'iterations': [300], 'learning_rate': [0.05, 0.1]}
catboost_search = GridSearchCV(CatBoostClassifier(verbose=0, random_seed=0), catboost_grid,
                               cv=3, scoring='accuracy', n_jobs=-1)
catboost_search.fit(X_train, y_train)
test_proba['CatBoost'] = catboost_search.best_estimator_.predict_proba(X_test)[:, 1]
catboost_row = [['CatBoost', round(accuracy_score(y_test, catboost.predict(X_test)), 3),
                 str(catboost_search.best_params_), round(catboost_search.best_score_, 3),
                 round(accuracy_score(y_test, catboost_search.best_estimator_.predict(X_test)), 3),
                 round(roc_auc_score(y_test, catboost_search.best_estimator_.predict_proba(X_test)[:, 1]), 3)]]
save_table(pd.DataFrame(catboost_row, columns=['Метод', 'Acc test(деф.)', 'Лучшие гиперпарам.',
                                               'CV acc', 'Acc test(тюн.)', 'AUC test']), 'catboost.csv')

# 3.4 ROC-кривые и AUC на тесте
plt.figure(figsize=(6, 5))
auc_test = {}
for name, proba in test_proba.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_test[name] = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=name + ' (AUC=' + str(round(auc_test[name], 3)) + ')')
plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-кривые классификаторов (тест)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(figures / 'roc_clf.png', dpi=150)
plt.close()
save_table(pd.DataFrame([[k, round(v, 3)] for k, v in auc_test.items()], columns=['Метод', 'AUC test']), 'clf_auc.csv')

# 3.5-3.6 матрица ошибок и пороги для лучшего по AUC классификатора
best_clf_name = max(auc_test, key=auc_test.get)
best_proba = test_proba[best_clf_name]
conf = confusion_matrix(y_test, (best_proba >= 0.5).astype(int))
pd.DataFrame(conf, index=['факт 0', 'факт 1'], columns=['прогноз 0', 'прогноз 1']).to_csv(tables / 'confusion.csv')

threshold_rows = []
for t in [0.3, 0.5, 0.7]:
    pred = (best_proba >= t).astype(int)
    threshold_rows.append([t, round(accuracy_score(y_test, pred), 3),
                           round(precision_score(y_test, pred, zero_division=0), 3),
                           round(recall_score(y_test, pred), 3)])
save_table(pd.DataFrame(threshold_rows, columns=['Порог', 'ACC', 'Precision', 'Recall']), 'thresholds.csv')


# 3.7 прибыль: порог подбираем на train, прибыль считаем на test (линейная и нелинейная)
def profit_nonlinear(y_true, proba, t):
    pred = proba >= t
    return gain * np.sqrt(proba[pred & (y_true == 1)]).sum() - cost * pred.sum()


profit_rows = []
for name, model in best_models.items():
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    profits_lin = [profit_linear(y_train, proba_train, t) for t in thresholds_grid]
    t_lin = thresholds_grid[int(np.argmax(profits_lin))]
    profits_nl = [profit_nonlinear(y_train, proba_train, t) for t in thresholds_grid]
    t_nl = thresholds_grid[int(np.argmax(profits_nl))]
    profit_rows.append([name, round(t_lin, 3), round(profit_linear(y_test, proba_test, t_lin), 1),
                        round(t_nl, 3), round(profit_nonlinear(y_test, proba_test, t_nl), 1)])
save_table(pd.DataFrame(profit_rows, columns=['Метод', 'Порог(лин.)', 'Прибыль test(лин.)',
                                              'Порог(нелин.)', 'Прибыль test(нелин.)']), 'profit.csv')

# 3.8 DAG и байесовская сеть (наш экспертный граф против выученного)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC

discrete = data.copy()
for col in ['age', 'size']:
    q1, q2 = np.quantile(discrete[col].iloc[train_idx], [1 / 3, 2 / 3])
    discrete[col + '_bin'] = np.digitize(discrete[col], [q1, q2])
bn_cols = ['D', 'Z', 'online', 'city', 'age_bin', 'size_bin']
bn_features = ['Z', 'online', 'city', 'age_bin', 'size_bin']
bn_train = discrete[bn_cols].astype(int).iloc[train_idx].reset_index(drop=True)
bn_test = discrete[bn_cols].astype(int).iloc[test_idx].reset_index(drop=True)

our_dag = DiscreteBayesianNetwork([(f, 'D') for f in bn_features])
our_dag.fit(bn_train)
acc_our = accuracy_score(bn_test['D'], our_dag.predict(bn_test[bn_features], n_jobs=1)['D'].values)

learned_structure = HillClimbSearch(bn_train).estimate(scoring_method=BIC(bn_train), max_indegree=3, show_progress=False)
learned_edges = list(learned_structure.edges())
learned_dag = DiscreteBayesianNetwork(learned_edges if learned_edges else [(bn_features[0], 'D')])
for node in bn_cols:
    if node not in learned_dag.nodes():
        learned_dag.add_node(node)
learned_dag.fit(bn_train)
acc_learned = accuracy_score(bn_test['D'], learned_dag.predict(bn_test[[c for c in bn_cols if c != 'D']], n_jobs=1)['D'].values)

bn_proba = our_dag.predict_probability(bn_test[bn_features])
proba_cols = [c for c in bn_proba.columns if str(c).endswith('1')]
auc_bn = roc_auc_score(bn_test['D'], bn_proba[proba_cols[0]].values if proba_cols else bn_proba.iloc[:, -1].values)
save_table(pd.DataFrame([['Наш экспертный DAG', round(acc_our, 3), '-'],
                         ['Выученный DAG (HC+BIC)', round(acc_learned, 3), str(learned_edges)],
                         ['Байес-сеть (AUC, наш DAG)', '-', round(auc_bn, 3)]],
                        columns=['Модель', 'Acc test', 'Доп.']), 'bayesnet.csv')

# 3.9 лучший и худший классификаторы по AUC на тесте
best_clf = max(auc_test, key=auc_test.get)
worst_clf = min(auc_test, key=auc_test.get)
print('Раздел 3: лучший =', best_clf, ', худший =', worst_clf, ', AUC =', {k: round(v, 3) for k, v in auc_test.items()})


# ===========================================================================
# РАЗДЕЛ 4. Регрессия: прогноз выручки Y (признаки без переменной воздействия D)
# ===========================================================================

X_reg = data[controls].values
y_reg = data['Y'].values
Xr_train, Xr_test = X_reg[train_idx], X_reg[test_idx]
yr_train, yr_test = y_reg[train_idx], y_reg[test_idx]
cv4 = KFold(4, shuffle=True, random_state=1)

regressors = {
    'OLS': (LinearRegression(), {}, False),
    'kNN': (KNeighborsRegressor(), {'model__n_neighbors': [10, 25, 50]}, True),
    'RandomForest': (RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
                     {'model__max_depth': [None, 12], 'model__min_samples_leaf': [1, 20]}, False),
    'GradBoost': (GradientBoostingRegressor(random_state=0),
                  {'model__n_estimators': [300], 'model__max_depth': [2, 3], 'model__learning_rate': [0.1]}, False),
}

# 4.2 RMSE и MAPE до и после подбора гиперпараметров
reg_rows = []
best_regressors = {}
for name, (model, grid, scale) in regressors.items():
    base = make_pipe(model, scale)
    base.fit(Xr_train, yr_train)
    rmse_train = rmse(yr_train, base.predict(Xr_train))
    rmse_test = rmse(yr_test, base.predict(Xr_test))
    rmse_cv = -cross_val_score(base, Xr_train, yr_train, cv=cv4, scoring='neg_root_mean_squared_error').mean()
    if grid:
        search = GridSearchCV(make_pipe(model, scale), grid, cv=cv4,
                              scoring='neg_root_mean_squared_error', n_jobs=-1)
        search.fit(Xr_train, yr_train)
        best = search.best_estimator_
        best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
        rmse_cv_tuned = -search.best_score_
    else:
        best = base
        best_params = {'-': '-'}
        rmse_cv_tuned = rmse_cv
    best_regressors[name] = best
    reg_rows.append([name, round(rmse_train, 2), round(rmse_cv, 2), round(rmse_test, 2),
                     round(mape(yr_test, base.predict(Xr_test)), 3), str(best_params),
                     round(rmse_cv_tuned, 2), round(rmse(yr_test, best.predict(Xr_test)), 2),
                     round(mape(yr_test, best.predict(Xr_test)), 3)])
save_table(pd.DataFrame(reg_rows, columns=['Метод', 'RMSE train', 'RMSE CV', 'RMSE test', 'MAPE test',
                                           'Лучшие гиперпарам.', 'RMSE CV(тюн.)', 'RMSE test(тюн.)',
                                           'MAPE test(тюн.)']), 'reg_tuning.csv')

# 4.2 (повышенная сложность) число деревьев бустинга по OOB против CV
gb = GradientBoostingRegressor(n_estimators=600, subsample=0.8, max_depth=3, learning_rate=0.05, random_state=0)
gb.fit(Xr_train, yr_train)
n_oob = int(np.argmax(np.cumsum(gb.oob_improvement_)) + 1)
gb_oob = GradientBoostingRegressor(n_estimators=n_oob, subsample=0.8, max_depth=3, learning_rate=0.05, random_state=0)
gb_oob.fit(Xr_train, yr_train)
save_table(pd.DataFrame([['CV (тюнинг)', '-', round(rmse(yr_test, best_regressors['GradBoost'].predict(Xr_test)), 2)],
                         ['OOB (subsample=0.8)', 'n_estimators=' + str(n_oob),
                          round(rmse(yr_test, gb_oob.predict(Xr_test)), 2)]],
                        columns=['Критерий', 'Параметры бустинга', 'RMSE test']), 'gb_oob.csv')

# 4.4 (повышенная сложность) дополнительный метод вне scikit-learn - CatBoostRegressor
from catboost import CatBoostRegressor

catboost_reg_grid = {'depth': [6], 'iterations': [400], 'learning_rate': [0.1]}
catboost_reg = GridSearchCV(CatBoostRegressor(verbose=0, random_seed=0), catboost_reg_grid,
                            cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
catboost_reg.fit(Xr_train, yr_train)
save_table(pd.DataFrame([['CatBoost', str(catboost_reg.best_params_),
                          round(rmse(yr_test, catboost_reg.best_estimator_.predict(Xr_test)), 2),
                          round(mape(yr_test, catboost_reg.best_estimator_.predict(Xr_test)), 3)]],
                        columns=['Метод', 'Лучшие гиперпарам.', 'RMSE test', 'MAPE test']), 'catboost_reg.csv')

# 4.3 лучшая и худшая регрессии по RMSE на CV
reg_cv_scores = {row[0]: row[6] for row in reg_rows}
best_reg = min(reg_cv_scores, key=reg_cv_scores.get)
worst_reg = max(reg_cv_scores, key=reg_cv_scores.get)
print('Раздел 4: лучшая =', best_reg, ', худшая =', worst_reg)


# ===========================================================================
# РАЗДЕЛ 5. Эффекты воздействия (вся выборка целиком)
# ===========================================================================
from doubleml import DoubleMLData, DoubleMLIRM, DoubleMLIIVM
from scipy.stats import spearmanr


def base_gbr():
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0)


def base_logit():
    return LogisticRegression(max_iter=1000)


# 5.2 истинные эффекты по потенциальным исходам на большой выборке
big = simulate(1_000_000, np.random.default_rng(777))
true_ATE = big['tau'].mean()
compliers = (big.D0 == 0) & (big.D1 == 1)
true_LATE = big.loc[compliers, 'tau'].mean()

sample = big.sample(100000, random_state=1)
plt.figure(figsize=(6, 4))
plt.hist(sample['tau'], bins=60, density=True, alpha=0.7, color='steelblue')
plt.axvline(true_ATE, color='red', ls='--', label='ATE=' + str(round(true_ATE, 1)))
plt.axvline(true_LATE, color='green', ls='--', label='LATE=' + str(round(true_LATE, 1)))
plt.xlabel('Индивидуальный эффект подключения, тыс. руб.')
plt.ylabel('Плотность')
plt.title('Истинное распределение эффектов воздействия (CATE)')
plt.legend()
plt.tight_layout()
plt.savefig(figures / 'cate_true.png', dpi=150)
plt.close()
big.groupby(['online', 'city'])['tau'].mean().round(2).to_csv(tables / 'cate_true_groups.csv')

X = data[controls].values
D = data['D'].values
Y = data['Y'].values
Z = data['Z'].values
tau = data['tau'].values

# 5.3 наивная разница средних + 5.4 ATE разными методами
ate = {}
ate['Наивная разница средних'] = Y[D == 1].mean() - Y[D == 0].mean()
ate['OLS (линейная корректировка)'] = sm.OLS(Y, sm.add_constant(np.column_stack([D, X]))).fit().params[1]

outcome_model = base_gbr()
outcome_model.fit(np.column_stack([D, X]), Y)
mu1 = outcome_model.predict(np.column_stack([np.ones_like(D), X]))
mu0 = outcome_model.predict(np.column_stack([np.zeros_like(D), X]))
ate['Условные мат. ожидания (g-computation)'] = (mu1 - mu0).mean()

propensity = LogisticRegression(max_iter=1000).fit(X, D).predict_proba(X)[:, 1]
propensity = np.clip(propensity, 0.02, 0.98)   # чтобы не было весов 1/0
ate['IPW (взвешивание на склонность)'] = (D * Y / propensity - (1 - D) * Y / (1 - propensity)).mean()
ate['AIPW (двойная устойчивость)'] = (mu1 - mu0 + D * (Y - mu1) / propensity
                                      - (1 - D) * (Y - mu0) / (1 - propensity)).mean()

dml_data = DoubleMLData(data[['Y', 'D'] + controls].copy(), y_col='Y', d_cols='D', x_cols=controls)
irm = DoubleMLIRM(dml_data, ml_g=base_gbr(), ml_m=base_logit(), n_folds=5)
irm.fit()
ate['Двойное машинное обучение (DML, без IV)'] = float(irm.coef[0])

ate_rows = [[name, round(value, 2), round(value - true_ATE, 2)] for name, value in ate.items()]
ate_rows.append(['Истинный ATE (супервыборка)', round(true_ATE, 2), 0.0])
save_table(pd.DataFrame(ate_rows, columns=['Метод оценки ATE', 'Оценка', 'Смещение от истинного ATE']), 'ate.csv')

# 5.5 условные средние эффекты (CATE)
cate = {}
ols_inter = sm.OLS(Y, sm.add_constant(np.column_stack([D, X, D[:, None] * X]))).fit().params
cate['OLS-взаимодействия'] = ols_inter[1] + X @ ols_inter[2 + len(controls):]

s_learner = base_gbr()
s_learner.fit(np.column_stack([D, X]), Y)
cate['S-learner'] = (s_learner.predict(np.column_stack([np.ones_like(D), X]))
                     - s_learner.predict(np.column_stack([np.zeros_like(D), X])))

model_treated = base_gbr()
model_control = base_gbr()
model_treated.fit(X[D == 1], Y[D == 1])
model_control.fit(X[D == 0], Y[D == 0])
cate['T-learner'] = model_treated.predict(X) - model_control.predict(X)

transformed = (D - propensity) / (propensity * (1 - propensity)) * Y
cate['Трансформация классов'] = base_gbr().fit(X, transformed).predict(X)

imputed_treated = base_gbr().fit(X[D == 1], Y[D == 1] - model_control.predict(X[D == 1])).predict(X)
imputed_control = base_gbr().fit(X[D == 0], model_treated.predict(X[D == 0]) - Y[D == 0]).predict(X)
cate['X-learner'] = propensity * imputed_control + (1 - propensity) * imputed_treated

cate_rows = []
for name, values in cate.items():
    cate_rows.append([name, round(float(np.mean(values)), 2), round(float(rmse(values, tau)), 2),
                      round(float(spearmanr(values, tau).correlation), 3)])
cate_rows.append(['Истинный CATE (tau)', round(tau.mean(), 2), 0.0, 1.0])
save_table(pd.DataFrame(cate_rows, columns=['Метод CATE', 'Среднее', 'RMSE к истинному CATE',
                                            'Ранг. корр. (Spearman)']), 'cate.csv')

cate_groups = data.copy()
cate_groups['cate_x'] = cate['X-learner']
cate_groups.groupby(['online', 'city']).agg(истинный=('tau', 'mean'),
                                            Xlearner=('cate_x', 'mean')).round(2).to_csv(tables / 'cate_groups_cmp.csv')

plt.figure(figsize=(6, 4))
plt.hist(tau, bins=50, density=True, alpha=0.5, label='истинный CATE', color='green')
plt.hist(cate['X-learner'], bins=50, density=True, alpha=0.5, label='X-learner', color='orange')
plt.xlabel('Эффект подключения, тыс. руб.')
plt.ylabel('Плотность')
plt.legend()
plt.title('Оценённый (X-learner) и истинный CATE')
plt.tight_layout()
plt.savefig(figures / 'cate_est.png', dpi=150)
plt.close()

# 5.6 LATE: Wald, DML без инструмента (смещён) и DML с инструментом
wald = (Y[Z == 1].mean() - Y[Z == 0].mean()) / (D[Z == 1].mean() - D[Z == 0].mean())
dml_data_iv = DoubleMLData(data[['Y', 'D', 'Z'] + controls].copy(), y_col='Y', d_cols='D',
                           x_cols=controls, z_cols='Z')
iivm = DoubleMLIIVM(dml_data_iv, ml_g=base_gbr(), ml_m=base_logit(), ml_r=base_logit(), n_folds=5)
iivm.fit()
late_dml = float(iivm.coef[0])
save_table(pd.DataFrame([['Wald / 2SLS (без ковариат)', round(wald, 2), round(wald - true_LATE, 2)],
                         ['DML без IV (смещён, не LATE)', round(float(irm.coef[0]), 2),
                          round(float(irm.coef[0]) - true_LATE, 2)],
                         ['DML с IV', round(late_dml, 2), round(late_dml - true_LATE, 2)],
                         ['Истинный LATE (супервыборка)', round(true_LATE, 2), 0.0]],
                        columns=['Метод оценки LATE', 'Оценка', 'Смещение от истинного LATE']), 'late.csv')


# 5.7 устойчивость: лучшие против слабых вспомогательных моделей
def nuisance_reg(kind):
    if kind == 'best':
        return base_gbr()
    return KNeighborsRegressor(n_neighbors=50)


def nuisance_clf(kind):
    if kind == 'best':
        return base_logit()
    return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=10, random_state=0, n_jobs=-1)


robust_rows = []
for kind in ['best', 'worst']:
    irm_k = DoubleMLIRM(dml_data, ml_g=nuisance_reg(kind), ml_m=nuisance_clf(kind), n_folds=5)
    ate_k = float(irm_k.fit().coef[0])
    iivm_k = DoubleMLIIVM(dml_data_iv, ml_g=nuisance_reg(kind), ml_m=nuisance_clf(kind),
                          ml_r=nuisance_clf(kind), n_folds=5)
    late_k = float(iivm_k.fit().coef[0])
    robust_rows.append([kind, round(ate_k, 2), round(ate_k - true_ATE, 2),
                        round(late_k, 2), round(late_k - true_LATE, 2)])
save_table(pd.DataFrame(robust_rows, columns=['Качество ML-моделей', 'DML без IV (ATE)', 'Смещение ATE',
                                              'DML с IV (LATE)', 'Смещение LATE']), 'late_robust.csv')

# 5.6 (повышенная сложность) параметрическая модель эндогенного переключения (MLE), аналог switchSelection.
# Уравнение отбора с инструментом Z и два уравнения выручки для D=0 и D=1; совместная нормальность
# ошибок с корреляциями rho0, rho1 между отбором и исходами корректирует эндогенность.
from scipy.optimize import minimize
from scipy.stats import norm

Xs = np.column_stack([np.ones(len(data)), np.log(data['age']), np.sqrt(data['size']), data['online'], data['city']])
Ws = np.column_stack([Xs, data['Z'].values])   # уравнение отбора дополнительно содержит инструмент Z
kf = Xs.shape[1]
kw = Ws.shape[1]


def switch_neg_loglik(theta):
    gamma = theta[:kw]
    beta0 = theta[kw:kw + kf]
    beta1 = theta[kw + kf:kw + 2 * kf]
    sigma0 = np.exp(theta[kw + 2 * kf])
    sigma1 = np.exp(theta[kw + 2 * kf + 1])
    rho0 = np.tanh(theta[kw + 2 * kf + 2])
    rho1 = np.tanh(theta[kw + 2 * kf + 3])
    index = Ws @ gamma
    e1 = (Y - Xs @ beta1) / sigma1
    e0 = (Y - Xs @ beta0) / sigma0
    ll1 = norm.logpdf(e1) - np.log(sigma1) + norm.logcdf((index + rho1 * e1) / np.sqrt(1 - rho1 ** 2))
    ll0 = norm.logpdf(e0) - np.log(sigma0) + norm.logcdf(-(index + rho0 * e0) / np.sqrt(1 - rho0 ** 2))
    loglik = np.where(D == 1, ll1, ll0)
    if not np.all(np.isfinite(loglik)):
        return 1e10
    return -loglik.sum()


# стартовые значения - МНК по двум режимам
start_beta1 = np.linalg.lstsq(Xs[D == 1], Y[D == 1], rcond=None)[0]
start_beta0 = np.linalg.lstsq(Xs[D == 0], Y[D == 0], rcond=None)[0]
start = np.concatenate([np.zeros(kw), start_beta0, start_beta1, [np.log(25), np.log(25), 0, 0]])
result = minimize(switch_neg_loglik, start, method='Nelder-Mead',
                  options={'maxiter': 60000, 'maxfev': 60000, 'xatol': 1e-4, 'fatol': 1e-3})
theta = result.x
beta0 = theta[kw:kw + kf]
beta1 = theta[kw + kf:kw + 2 * kf]
rho0 = np.tanh(theta[kw + 2 * kf + 2])
rho1 = np.tanh(theta[kw + 2 * kf + 3])
ate_switch = float((Xs @ (beta1 - beta0)).mean())
save_table(pd.DataFrame([['Методы на наблюдаемых (DML без IV)', round(float(irm.coef[0]), 2),
                          round(float(irm.coef[0]) - true_ATE, 2)],
                         ['Параметрическое эндог. переключение (MLE)', round(ate_switch, 2),
                          round(ate_switch - true_ATE, 2)],
                         ['Двойное машинное обучение с IV', round(late_dml, 2), round(late_dml - true_ATE, 2)],
                         ['Истинный ATE', round(true_ATE, 2), 0.0]],
                        columns=['Метод', 'Оценка эффекта', 'Смещение от истинного ATE']), 'switch_param.csv')

json.dump({'true_ATE': true_ATE, 'true_LATE': true_LATE, 'compliers': float(compliers.mean())},
          open(tables / 'true_effects.json', 'w'))
print('Раздел 5: истинный ATE =', round(true_ATE, 2), ', LATE =', round(true_LATE, 2))
print('  наивная =', round(ate['Наивная разница средних'], 2),
      ', DML без IV =', round(float(irm.coef[0]), 2),
      ', DML с IV =', round(late_dml, 2),
      ', параметрическая =', round(ate_switch, 2))
print('Готово: результаты в папках figures/ и tables/')
