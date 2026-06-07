# -*- coding: utf-8 -*-
# ============================================================================
# Домашнее задание по курсу «Машинное обучение в экономике», 2025-2026.
# Тема: влияние подключения малого бизнеса к маркетплейсу на месячную выручку.
# Один файл воспроизводит все результаты. Таблицы (CSV) сохраняются в ./tables,
# рисунки (PNG) в ./figures. Нумерация блоков соответствует заданию.
# Запуск:  python3 hw.py
# ============================================================================
import os, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, mean_squared_error,
                             mean_absolute_percentage_error, make_scorer)
import statsmodels.api as sm

FAST = os.environ.get('FAST') == '1'
FIG = Path('figures'); TAB = Path('tables'); FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
def savetab(df, name): df.to_csv(TAB/name, index=False)
def pipe(est, scale): return Pipeline([('sc', StandardScaler()), ('m', est)]) if scale else Pipeline([('m', est)])
rmse = lambda a, b: mean_squared_error(a, b) ** 0.5

# ============================================================================
# РАЗДЕЛ 2. Генерация данных (DGP), описательные статистики, разбиение выборки
# ============================================================================
P = dict(gamma0=-1.2, gamma_z=0.9, gamma_u=0.7, gamma_u2=0.1,
         d1=-0.25, d2=0.35, d3=0.6, d4=0.2, d5=-0.2,
         b0=50, b1=40, b2=15, b3=30, b4=25, b5=20, b6=3, sigma=25,
         t0=35, t1=12, t2=-15, t3=-10, t4=3)

def simulate(n, rng, p=P):
    """Процесс генерации данных по разделу 2.1 (U -- ненаблюдаемое качество менеджмента)."""
    U = rng.standard_normal(n)
    age = np.exp(rng.normal(1.6, 0.5, n))
    size = np.ceil(np.exp(rng.normal(1.2, 0.6, n))).astype(int)
    online = (rng.random(n) < 1/(1+np.exp(-(-0.2 + 0.8*U)))).astype(int)
    city = (rng.random(n) < 0.5).astype(int)
    Z = (rng.random(n) < 0.5).astype(int)
    g = p['d1']*np.log(age) + p['d2']*np.sqrt(size) + p['d3']*online + p['d4']*city + p['d5']*online*city
    base = p['gamma0'] + p['gamma_u']*U + p['gamma_u2']*U**2 + g + rng.standard_normal(n)
    D0 = (base >= 0).astype(int)              # потенциальные исходы воздействия
    D1 = (base + p['gamma_z'] >= 0).astype(int)
    D = np.where(Z == 1, D1, D0)
    mu = (p['b0'] + p['b1']*np.sqrt(size) + p['b2']*np.log(age) + p['b3']*online
          + p['b4']*city + p['b5']*U + p['b6']*U**2)
    tau = p['t0'] + p['t1']*U + p['t2']*online + p['t3']*city + p['t4']*np.log(age)
    xi = rng.standard_normal(n)
    Y0 = mu + p['sigma']*xi; Y1 = mu + tau + p['sigma']*xi
    Y = np.where(D == 1, Y1, Y0)
    return pd.DataFrame(dict(Y=Y, D=D, Z=Z, age=age, size=size, online=online, city=city,
                            U=U, D0=D0, D1=D1, Y0=Y0, Y1=Y1, tau=tau))

N_MAIN = 1500 if FAST else 10000
df = simulate(N_MAIN, np.random.default_rng(20260606))
obs = ['Y', 'D', 'Z', 'age', 'size', 'online', 'city']           # наблюдаемые (U исключена)
ctrl = ['age', 'size', 'online', 'city']

# 2.2 описательные статистики и корреляционная матрица
cont, binv = ['Y', 'age', 'size'], ['D', 'Z', 'online', 'city']
savetab(pd.DataFrame([[c, f'{df[c].mean():.2f}', f'{df[c].std():.2f}', f'{df[c].median():.2f}',
                       f'{df[c].min():.2f}', f'{df[c].max():.2f}'] for c in cont],
                     columns=['Переменная', 'Среднее', 'Ст.откл.', 'Медиана', 'Мин', 'Макс']), 'desc_cont.csv')
savetab(pd.DataFrame([[c, f'{df[c].mean():.3f}', int(df[c].sum())] for c in binv],
                     columns=['Переменная', 'Доля единиц', 'Кол-во единиц']), 'desc_bin.csv')
corr = df[obs].corr(); corr.to_csv(TAB/'corr.csv')
fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(obs))); ax.set_xticklabels(obs, rotation=45, ha='right')
ax.set_yticks(range(len(obs))); ax.set_yticklabels(obs)
for i in range(len(obs)):
    for j in range(len(obs)): ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
fig.colorbar(im); plt.title('Корреляционная матрица наблюдаемых переменных'); plt.tight_layout()
plt.savefig(FIG/'corr.png', dpi=150); plt.close()

# 2.3 разбиение на обучающую (75%) и тестовую (25%) выборки
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.25, random_state=42, stratify=df['D'])
print(f'[Р2] N={len(df)}; train={len(tr)} test={len(te)}; '
      f'доли D/Z/online/city={[round(df[c].mean(),3) for c in binv]}', flush=True)

# ============================================================================
# РАЗДЕЛ 3. Классификация переменной воздействия D (признаки без Y и без U)
# ============================================================================
fC = ['Z', 'age', 'size', 'online', 'city']
Xc, yc = df[fC].values, df['D'].values
Xtr, Xte, ytr, yte = Xc[tr], Xc[te], yc[tr], yc[te]
cvc = StratifiedKFold(5, shuffle=True, random_state=1)
clf_models = {
 'LogReg': (LogisticRegression(max_iter=1000), {'m__C': [0.1, 1, 10]}, True),
 'kNN': (KNeighborsClassifier(), {'m__n_neighbors': [15, 31, 51]}, True),
 'NaiveBayes': (GaussianNB(), {'m__var_smoothing': [1e-9, 1e-7]}, True),
 'RandomForest': (RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                  {'m__max_depth': [None, 12], 'm__min_samples_leaf': [1, 10]}, False),
 'GradBoost': (GradientBoostingClassifier(random_state=0),
               {'m__n_estimators': [200], 'm__max_depth': [2, 3], 'm__learning_rate': [0.1]}, False),
}
# 3.2 точность по умолчанию и после тюнинга (accuracy)
rows, tuned_c, proba_c = [], {}, {}
for name, (est, grid, sc) in clf_models.items():
    base = pipe(est, sc).fit(Xtr, ytr)
    a_tr, a_te = accuracy_score(ytr, base.predict(Xtr)), accuracy_score(yte, base.predict(Xte))
    a_cv = cross_val_score(base, Xtr, ytr, cv=cvc, scoring='accuracy').mean()
    gs = GridSearchCV(pipe(est, sc), grid, cv=cvc, scoring='accuracy', n_jobs=-1).fit(Xtr, ytr)
    bst = gs.best_estimator_; tuned_c[name] = bst; proba_c[name] = bst.predict_proba(Xte)[:, 1]
    bp = {k.replace('m__', ''): v for k, v in gs.best_params_.items()}
    rows.append([name, round(a_tr, 3), round(a_cv, 3), round(a_te, 3), str(bp),
                 round(gs.best_score_, 3), round(accuracy_score(yte, bst.predict(Xte)), 3)])
savetab(pd.DataFrame(rows, columns=['Метод', 'Acc train', 'Acc CV', 'Acc test', 'Лучшие гиперпарам.',
                                    'CV acc (тюн.)', 'Acc test (тюн.)']), 'clf_tuning.csv')

# 3.3 альтернативный критерий качества: тюнинг по AUC
rows = []
for name, (est, grid, sc) in clf_models.items():
    gs = GridSearchCV(pipe(est, sc), grid, cv=cvc, scoring='roc_auc', n_jobs=-1).fit(Xtr, ytr)
    rows.append([name, str({k.replace('m__', ''): v for k, v in gs.best_params_.items()}),
                 round(gs.best_score_, 3), round(roc_auc_score(yte, gs.best_estimator_.predict_proba(Xte)[:, 1]), 3)])
savetab(pd.DataFrame(rows, columns=['Метод', 'Лучшие гиперпарам.(AUC)', 'CV AUC', 'AUC test']), 'clf_auc_tuning.csv')

# 3.3 (повыш. сложность) собственный критерий: ожидаемая прибыль таргетинга
G, C = 10.0, 3.0; thr_grid = np.linspace(0.05, 0.95, 91)
def profit_lin(yt, p, t): pr = p >= t; return (G-C)*np.sum(pr & (yt == 1)) - C*np.sum(pr & (yt == 0))
def profit_score(yt, p):
    p = np.asarray(p); p = p[:, -1] if p.ndim > 1 else p
    return max(profit_lin(yt, p, t) for t in thr_grid)
prof_scorer = make_scorer(profit_score, response_method='predict_proba')
gs_acc = GridSearchCV(pipe(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1), False),
                      {'m__max_depth': [None, 12], 'm__min_samples_leaf': [1, 10]}, cv=cvc, scoring='accuracy', n_jobs=-1).fit(Xtr, ytr)
gs_prof = GridSearchCV(pipe(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1), False),
                       {'m__max_depth': [None, 12], 'm__min_samples_leaf': [1, 10]}, cv=cvc, scoring=prof_scorer, n_jobs=-1).fit(Xtr, ytr)
savetab(pd.DataFrame([['accuracy', str({k.replace('m__', ''): v for k, v in gs_acc.best_params_.items()}), round(gs_acc.best_score_, 3)],
                      ['собств. критерий (прибыль)', str({k.replace('m__', ''): v for k, v in gs_prof.best_params_.items()}), round(gs_prof.best_score_, 1)]],
                     columns=['Критерий тюнинга', 'Лучшие гиперпарам. RF', 'Значение CV-критерия']), 'custom_criterion.csv')

# 3.2 (повыш. сложность) OOB-тюнинг случайного леса против кросс-валидации
rows = []
for md in [None, 6, 12]:
    for ms in [1, 5, 20]:
        rf = RandomForestClassifier(n_estimators=300, max_depth=md, min_samples_leaf=ms,
                                    oob_score=True, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        rows.append([str(md), ms, round(rf.oob_score_, 4), round(accuracy_score(yte, rf.predict(Xte)), 4)])
savetab(pd.DataFrame(rows, columns=['max_depth', 'min_samples_leaf', 'OOB acc', 'Test acc']), 'rf_oob.csv')

# 3.10 (повыш. сложность) доп. метод вне scikit-learn: CatBoost
from catboost import CatBoostClassifier
cb = CatBoostClassifier(verbose=0, random_seed=0).fit(Xtr, ytr)
cbg = {'depth': [4, 6], 'iterations': [300], 'learning_rate': [0.05, 0.1]} if not FAST else {'iterations': [200]}
cbgs = GridSearchCV(CatBoostClassifier(verbose=0, random_seed=0), cbg, cv=3, scoring='accuracy', n_jobs=-1).fit(Xtr, ytr)
proba_c['CatBoost'] = cbgs.best_estimator_.predict_proba(Xte)[:, 1]
savetab(pd.DataFrame([['CatBoost', round(accuracy_score(yte, cb.predict(Xte)), 3), str(cbgs.best_params_),
                       round(cbgs.best_score_, 3), round(accuracy_score(yte, cbgs.best_estimator_.predict(Xte)), 3),
                       round(roc_auc_score(yte, cbgs.best_estimator_.predict_proba(Xte)[:, 1]), 3)]],
                     columns=['Метод', 'Acc test(деф.)', 'Лучшие гиперпарам.', 'CV acc', 'Acc test(тюн.)', 'AUC test']), 'catboost.csv')

# 3.4 ROC-кривые и AUC на тесте
plt.figure(figsize=(6, 5)); auc_c = {}
for name, p in proba_c.items():
    fpr, tpr, _ = roc_curve(yte, p); auc_c[name] = roc_auc_score(yte, p)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_c[name]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=.8); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC-кривые классификаторов (тест)'); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(FIG/'roc_clf.png', dpi=150); plt.close()
savetab(pd.DataFrame([[k, round(v, 3)] for k, v in auc_c.items()], columns=['Метод', 'AUC test']), 'clf_auc.csv')

# 3.5-3.6 матрица ошибок и пороги для лучшего по AUC классификатора
best_c = max(auc_c, key=auc_c.get); pbest = proba_c[best_c]
cm = confusion_matrix(yte, (pbest >= 0.5).astype(int))
pd.DataFrame(cm, index=['факт 0', 'факт 1'], columns=['прогноз 0', 'прогноз 1']).to_csv(TAB/'confusion.csv')
savetab(pd.DataFrame([[t, round(accuracy_score(yte, (pbest >= t).astype(int)), 3),
                       round(precision_score(yte, (pbest >= t).astype(int), zero_division=0), 3),
                       round(recall_score(yte, (pbest >= t).astype(int)), 3)] for t in [0.3, 0.5, 0.7]],
                     columns=['Порог', 'ACC', 'Precision', 'Recall']), 'thresholds.csv')

# 3.7 прибыль: оптимальный порог на train, прибыль на test (линейная и нелинейная функции)
def profit_nl(yt, p, t): pr = p >= t; return G*np.sqrt(p[pr & (yt == 1)]).sum() - C*pr.sum()
rows = []
for name, m in tuned_c.items():
    ptr, pte = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]
    tl = thr_grid[int(np.argmax([profit_lin(ytr, ptr, t) for t in thr_grid]))]
    tn = thr_grid[int(np.argmax([profit_nl(ytr, ptr, t) for t in thr_grid]))]
    rows.append([name, round(tl, 3), round(profit_lin(yte, pte, tl), 1), round(tn, 3), round(profit_nl(yte, pte, tn), 1)])
savetab(pd.DataFrame(rows, columns=['Метод', 'Порог(лин.)', 'Прибыль test(лин.)', 'Порог(нелин.)', 'Прибыль test(нелин.)']), 'profit.csv')

# 3.8 DAG и байесовская сеть (наш экспертный DAG против выученного)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC
dd = df.copy()
for c in ['age', 'size']:
    qs = np.quantile(dd[c].iloc[tr], [1/3, 2/3]); dd[c+'_b'] = np.digitize(dd[c], qs)
cols = ['D', 'Z', 'online', 'city', 'age_b', 'size_b']; fB = ['Z', 'online', 'city', 'age_b', 'size_b']
Dtr, Dte = dd[cols].astype(int).iloc[tr].reset_index(drop=True), dd[cols].astype(int).iloc[te].reset_index(drop=True)
our = DiscreteBayesianNetwork([(f, 'D') for f in fB]); our.fit(Dtr)
acc_our = accuracy_score(Dte['D'], our.predict(Dte[fB], n_jobs=1)['D'].values)
learned = HillClimbSearch(Dtr).estimate(scoring_method=BIC(Dtr), max_indegree=3, show_progress=False)
edges = list(learned.edges()); lbn = DiscreteBayesianNetwork(edges if edges else [(fB[0], 'D')])
for n in cols:
    if n not in lbn.nodes(): lbn.add_node(n)
lbn.fit(Dtr)
acc_learn = accuracy_score(Dte['D'], lbn.predict(Dte[[c for c in cols if c != 'D']], n_jobs=1)['D'].values)
proba_bn = our.predict_probability(Dte[fB]); c1 = [c for c in proba_bn.columns if str(c).endswith('1')]
auc_bn = roc_auc_score(Dte['D'], proba_bn[c1[0]].values if c1 else proba_bn.iloc[:, -1].values)
savetab(pd.DataFrame([['Наш экспертный DAG', round(acc_our, 3), '-'],
                      ['Выученный DAG (HC+BIC)', round(acc_learn, 3), str(edges)],
                      ['Байес-сеть (AUC, наш DAG)', '-', round(auc_bn, 3)]],
                     columns=['Модель', 'Acc test', 'Доп.']), 'bayesnet.csv')

# 3.9 лучший и худший классификаторы по AUC на тесте
best_clf, worst_clf = max(auc_c, key=auc_c.get), min(auc_c, key=auc_c.get)
auc_round = {k: round(v, 3) for k, v in auc_c.items()}
print(f'[Р3] AUC={auc_round}; лучший={best_clf}, худший={worst_clf}; '
      f'DAG: наш={acc_our:.3f}, выученный={acc_learn:.3f}', flush=True)

# ============================================================================
# РАЗДЕЛ 4. Регрессия: прогноз выручки Y (признаки без переменной воздействия D)
# ============================================================================
Xr = df[ctrl].values; yr = df['Y'].values
Xrtr, Xrte, yrtr, yrte = Xr[tr], Xr[te], yr[tr], yr[te]
cvr = KFold(4, shuffle=True, random_state=1)
mape = lambda a, b: mean_absolute_percentage_error(a, b)
reg_models = {
 'OLS': (LinearRegression(), {}, False),
 'kNN': (KNeighborsRegressor(), {'m__n_neighbors': [10, 25, 50]}, True),
 'RandomForest': (RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
                  {'m__max_depth': [None, 12], 'm__min_samples_leaf': [1, 20]}, False),
 'GradBoost': (GradientBoostingRegressor(random_state=0),
               {'m__n_estimators': [300], 'm__max_depth': [2, 3], 'm__learning_rate': [0.1]}, False),
}
# 4.2 RMSE/MAPE до и после тюнинга
rows, tuned_r = [], {}
for name, (est, grid, sc) in reg_models.items():
    base = pipe(est, sc).fit(Xrtr, yrtr)
    r_tr, r_te = rmse(yrtr, base.predict(Xrtr)), rmse(yrte, base.predict(Xrte))
    r_cv = -cross_val_score(base, Xrtr, yrtr, cv=cvr, scoring='neg_root_mean_squared_error').mean()
    if grid:
        gs = GridSearchCV(pipe(est, sc), grid, cv=cvr, scoring='neg_root_mean_squared_error', n_jobs=-1).fit(Xrtr, yrtr)
        bst = gs.best_estimator_; bp = {k.replace('m__', ''): v for k, v in gs.best_params_.items()}; cvb = -gs.best_score_
    else:
        bst, bp, cvb = base, {'-': '-'}, r_cv
    tuned_r[name] = bst
    rows.append([name, round(r_tr, 2), round(r_cv, 2), round(r_te, 2), round(mape(yrte, base.predict(Xrte)), 3),
                 str(bp), round(cvb, 2), round(rmse(yrte, bst.predict(Xrte)), 2), round(mape(yrte, bst.predict(Xrte)), 3)])
savetab(pd.DataFrame(rows, columns=['Метод', 'RMSE train', 'RMSE CV', 'RMSE test', 'MAPE test', 'Лучшие гиперпарам.',
                                    'RMSE CV(тюн.)', 'RMSE test(тюн.)', 'MAPE test(тюн.)']), 'reg_tuning.csv')

# 4.2 (повыш. сложность) OOB-подход для бустинга (subsample<1) против CV
gb = GradientBoostingRegressor(n_estimators=600, subsample=0.8, max_depth=3, learning_rate=0.05, random_state=0).fit(Xrtr, yrtr)
n_oob = int(np.argmax(np.cumsum(gb.oob_improvement_)) + 1)
gb_oob = GradientBoostingRegressor(n_estimators=n_oob, subsample=0.8, max_depth=3, learning_rate=0.05, random_state=0).fit(Xrtr, yrtr)
savetab(pd.DataFrame([['CV (тюнинг)', '-', round(rmse(yrte, tuned_r['GradBoost'].predict(Xrte)), 2)],
                      ['OOB (subsample=0.8)', f'n_estimators={n_oob}', round(rmse(yrte, gb_oob.predict(Xrte)), 2)]],
                     columns=['Критерий', 'Параметры бустинга', 'RMSE test']), 'gb_oob.csv')

# 4.4 (повыш. сложность) доп. метод вне scikit-learn: CatBoostRegressor
from catboost import CatBoostRegressor
cbr_g = {'depth': [6], 'iterations': [400], 'learning_rate': [0.1]} if not FAST else {'iterations': [200]}
cbr = GridSearchCV(CatBoostRegressor(verbose=0, random_seed=0), cbr_g, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1).fit(Xrtr, yrtr)
savetab(pd.DataFrame([['CatBoost', str(cbr.best_params_), round(rmse(yrte, cbr.best_estimator_.predict(Xrte)), 2),
                       round(mape(yrte, cbr.best_estimator_.predict(Xrte)), 3)]],
                     columns=['Метод', 'Лучшие гиперпарам.', 'RMSE test', 'MAPE test']), 'catboost_reg.csv')

# 4.3 лучшая и худшая регрессии по RMSE на CV
reg_cv = {r[0]: r[6] for r in rows}; best_reg, worst_reg = min(reg_cv, key=reg_cv.get), max(reg_cv, key=reg_cv.get)
rmse_round = {r[0]: r[7] for r in rows}
print(f'[Р4] RMSE_test={rmse_round}; лучшая={best_reg}, худшая={worst_reg}', flush=True)

# ============================================================================
# РАЗДЕЛ 5. Эффекты воздействия (вся выборка целиком)
# ============================================================================
# 5.2 истинные эффекты по потенциальным исходам на супервыборке
n_big = 200000 if FAST else 1_000_000
big = simulate(n_big, np.random.default_rng(777))
true_ATE = big['tau'].mean()
comp = (big.D0 == 0) & (big.D1 == 1); true_LATE = big.loc[comp, 'tau'].mean()
samp = big.sample(min(100000, n_big), random_state=1)
plt.figure(figsize=(6, 4)); plt.hist(samp['tau'], bins=60, density=True, alpha=.7, color='steelblue')
plt.axvline(true_ATE, color='red', ls='--', label=f'ATE={true_ATE:.1f}')
plt.axvline(true_LATE, color='green', ls='--', label=f'LATE={true_LATE:.1f}')
plt.xlabel('Индивидуальный эффект подключения, тыс. руб.'); plt.ylabel('Плотность')
plt.title('Истинное распределение эффектов воздействия (CATE)'); plt.legend(); plt.tight_layout()
plt.savefig(FIG/'cate_true.png', dpi=150); plt.close()
big.groupby(['online', 'city'])['tau'].mean().round(2).to_csv(TAB/'cate_true_groups.csv')

X, D, Y, Z = df[ctrl].values, df['D'].values, df['Y'].values, df['Z'].values
tau = df['tau'].values
# 5.3 наивная разница средних + 5.4 ATE разными методами
res = {'Наивная разница средних': Y[D == 1].mean() - Y[D == 0].mean()}
res['OLS (линейная корректировка)'] = sm.OLS(Y, sm.add_constant(np.column_stack([D, X]))).fit().params[1]
g = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0).fit(np.column_stack([D, X]), Y)
mu1 = g.predict(np.column_stack([np.ones_like(D), X])); mu0 = g.predict(np.column_stack([np.zeros_like(D), X]))
res['Условные мат. ожидания (g-computation)'] = (mu1 - mu0).mean()
e = np.clip(LogisticRegression(max_iter=1000).fit(X, D).predict_proba(X)[:, 1], 0.02, 0.98)
res['IPW (взвешивание на склонность)'] = (D*Y/e - (1-D)*Y/(1-e)).mean()
res['AIPW (двойная устойчивость)'] = (mu1 - mu0 + D*(Y-mu1)/e - (1-D)*(Y-mu0)/(1-e)).mean()
from doubleml import DoubleMLData, DoubleMLIRM, DoubleMLIIVM
dat = DoubleMLData(df[['Y', 'D']+ctrl].copy(), y_col='Y', d_cols='D', x_cols=ctrl)
ml_g = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0)
ml_m = lambda: LogisticRegression(max_iter=1000)
irm = DoubleMLIRM(dat, ml_g=ml_g(), ml_m=ml_m(), n_folds=5); irm.fit()
res['Двойное машинное обучение (DML, без IV)'] = float(irm.coef[0])
ate_tab = pd.DataFrame([[k, round(v, 2), round(v-true_ATE, 2)] for k, v in res.items()],
                       columns=['Метод оценки ATE', 'Оценка', 'Смещение от истинного ATE'])
ate_tab.loc[len(ate_tab)] = ['Истинный ATE (супервыборка)', round(true_ATE, 2), 0.0]
savetab(ate_tab, 'ate.csv')

# 5.5 условные средние эффекты (CATE): OLS-взаимодействия, S/T/X-learner, трансформация классов
from scipy.stats import spearmanr
basg = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0)
cate = {}
mi = sm.OLS(Y, sm.add_constant(np.column_stack([D, X, D[:, None]*X]))).fit().params
cate['OLS-взаимодействия'] = mi[1] + X @ mi[2+len(ctrl):]
s = basg().fit(np.column_stack([D, X]), Y)
cate['S-learner'] = s.predict(np.column_stack([np.ones_like(D), X])) - s.predict(np.column_stack([np.zeros_like(D), X]))
m1 = basg().fit(X[D == 1], Y[D == 1]); m0 = basg().fit(X[D == 0], Y[D == 0])
cate['T-learner'] = m1.predict(X) - m0.predict(X)
cate['Трансформация классов'] = basg().fit(X, (D-e)/(e*(1-e))*Y).predict(X)
tx1 = basg().fit(X[D == 1], Y[D == 1] - m0.predict(X[D == 1])).predict(X)
tx0 = basg().fit(X[D == 0], m1.predict(X[D == 0]) - Y[D == 0]).predict(X)
cate['X-learner'] = e*tx0 + (1-e)*tx1
rows = [[k, round(float(np.mean(v)), 2), round(float(rmse(v, tau)), 2), round(float(spearmanr(v, tau).correlation), 3)] for k, v in cate.items()]
rows.append(['Истинный CATE (tau)', round(tau.mean(), 2), 0.0, 1.0])
savetab(pd.DataFrame(rows, columns=['Метод CATE', 'Среднее', 'RMSE к истинному CATE', 'Ранг. корр. (Spearman)']), 'cate.csv')
dd2 = df.copy(); dd2['cate_x'] = cate['X-learner']
dd2.groupby(['online', 'city']).agg(истинный=('tau', 'mean'), Xlearner=('cate_x', 'mean')).round(2).to_csv(TAB/'cate_groups_cmp.csv')
plt.figure(figsize=(6, 4)); plt.hist(tau, bins=50, density=True, alpha=.5, label='истинный CATE', color='green')
plt.hist(cate['X-learner'], bins=50, density=True, alpha=.5, label='X-learner', color='orange')
plt.xlabel('Эффект подключения, тыс. руб.'); plt.ylabel('Плотность'); plt.legend()
plt.title('Оценённый (X-learner) и истинный CATE'); plt.tight_layout(); plt.savefig(FIG/'cate_est.png', dpi=150); plt.close()

# 5.6 LATE: Wald, DML без IV (смещён) и DML с IV
wald = (Y[Z == 1].mean() - Y[Z == 0].mean()) / (D[Z == 1].mean() - D[Z == 0].mean())
dat_iv = DoubleMLData(df[['Y', 'D', 'Z']+ctrl].copy(), y_col='Y', d_cols='D', x_cols=ctrl, z_cols='Z')
iivm = DoubleMLIIVM(dat_iv, ml_g=ml_g(), ml_m=ml_m(), ml_r=ml_m(), n_folds=5); iivm.fit()
late_dml = float(iivm.coef[0])
savetab(pd.DataFrame([['Wald / 2SLS (без ковариат)', round(wald, 2), round(wald-true_LATE, 2)],
                      ['DML без IV (смещён, не LATE)', round(float(irm.coef[0]), 2), round(float(irm.coef[0])-true_LATE, 2)],
                      ['DML с IV', round(late_dml, 2), round(late_dml-true_LATE, 2)],
                      ['Истинный LATE (супервыборка)', round(true_LATE, 2), 0.0]],
                     columns=['Метод оценки LATE', 'Оценка', 'Смещение от истинного LATE']), 'late.csv')

# 5.7 робастность: лучшие против худших ML-моделей в качестве вспомогательных
def reg_k(kind): return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0) if kind == 'best' else KNeighborsRegressor(n_neighbors=50)
def clf_k(kind): return LogisticRegression(max_iter=1000) if kind == 'best' else RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=10, random_state=0, n_jobs=-1)
rows = []
for kind in ['best', 'worst']:
    a = float(DoubleMLIRM(dat, ml_g=reg_k(kind), ml_m=clf_k(kind), n_folds=5).fit().coef[0])
    l = float(DoubleMLIIVM(dat_iv, ml_g=reg_k(kind), ml_m=clf_k(kind), ml_r=clf_k(kind), n_folds=5).fit().coef[0])
    rows.append([kind, round(a, 2), round(a-true_ATE, 2), round(l, 2), round(l-true_LATE, 2)])
savetab(pd.DataFrame(rows, columns=['Качество ML-моделей', 'DML без IV (ATE)', 'Смещение ATE', 'DML с IV (LATE)', 'Смещение LATE']), 'late_robust.csv')
json.dump({'true_ATE': true_ATE, 'true_LATE': true_LATE, 'compliers': float(comp.mean())}, open(TAB/'true_effects.json', 'w'))
print(f'[Р5] истинный ATE={true_ATE:.2f}, LATE={true_LATE:.2f}; наивная={res["Наивная разница средних"]:.2f}, '
      f'DML без IV={float(irm.coef[0]):.2f}, DML с IV(LATE)={late_dml:.2f}, Wald={wald:.2f}', flush=True)
print('=== ВСЕ РАЗДЕЛЫ ВЫПОЛНЕНЫ; результаты в ./figures и ./tables ===', flush=True)
