import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import math
import seaborn as sn
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./train.csv')
valid = pd.read_csv('./valid.csv')
test = pd.read_csv('./test.csv')

train.info()

train.isna().sum()

train.columns

valid.columns

test.columns

def getModels():
    return {
        'svm' : svm.SVC(),
        'random_forest' : RandomForestClassifier(),
        'logistic_regression' : LogisticRegression(solver = 'liblinear', multi_class = 'auto'),
        'knn' : KNeighborsClassifier()
    }

## Label 1 ##

x_train = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
x_valid = valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
x_test = test.drop(['ID'], axis=1)

y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]
y_valid = valid[['label_1', 'label_2', 'label_3', 'label_4']]

label1_value_counts = y_train['label_1'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(label1_value_counts.index, label1_value_counts.values)


plt.xlabel('Label 1')
plt.ylabel('Count')
plt.title('Label 1 values distribution')

plt.tight_layout()
plt.show()

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

pca = PCA(n_components=0.95, svd_solver = 'full')
x_train_pca = pca.fit_transform(x_train_scaled)
x_valid_pca = pca.transform(x_valid_scaled)
x_test_pca = pca.transform(x_test_scaled)

scores = []
for model_name, model in getModels().items():
    fold_scores = cross_val_score(model, x_train_pca, y_train['label_1'], cv=3, scoring='accuracy', n_jobs = -1)
    scores.append({
        'model' : model_name,
        'score' : np.mean(fold_scores)
    })

df = pd.DataFrame(scores, columns = ['model', 'score'])
df

model_data = {
    'svm' : {
        'model':svm.SVC(gamma = 'auto'),
        'params' : {
            'C': [1, 10, 100],
            'kernel': ['rbf', 'linear']}
    },
    'logistic_regression' : {
        'model':LogisticRegression(solver = 'liblinear', multi_class = 'auto'),
        'params' : {
            'C': [1, 10, 100]
        }
    }
}

accuracy_scores = []

for model_name, model_info in model_data.items():
    clf = RandomizedSearchCV(model_info['model'], param_distributions = model_info['params'], n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    clf.fit(x_train_pca, y_train['label_1'])
    accuracy_scores.append({
      'model' : model_name,
      'best_score' : clf.best_score_,
      'best_params' : clf.best_params_
})

df_hyp_tuning = pd.DataFrame(accuracy_scores, columns = ['model', 'best_score', 'best_params'])
df_hyp_tuning

best_model_l1 = LogisticRegression(C=1)
best_model_l1.fit(x_train_pca, y_train['label_1'])

#validation
fold_scores_l1 = cross_val_score(best_model_l1, x_valid_pca, y_valid['label_1'], cv=3, scoring='accuracy', n_jobs = -1)
np.mean(fold_scores_l1)

y_test_label1 = best_model_l1.predict(x_test_pca)
y_test_label1.shape

## Label 2 ##

train['label_2'].isnull().sum()

valid['label_2'].isnull().sum()

train_l2 = train.dropna(subset = ['label_2'])
train_l2['label_2'].isnull().sum()

valid_l2 = valid.dropna(subset = ['label_2'])
valid_l2['label_2'].isnull().sum()

x_train = train_l2.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
x_valid = valid_l2.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
x_test = test.drop(['ID'], axis=1)

y_train = train_l2[['label_1', 'label_2', 'label_3', 'label_4']]
y_valid = valid_l2[['label_1', 'label_2', 'label_3', 'label_4']]

label2_value_counts = y_train['label_2'].value_counts()

plt.figure(figsize=(10, 6))
bars = plt.bar(label2_value_counts.index, label2_value_counts.values)


plt.xlabel('Label 2')
plt.ylabel('Count')
plt.title('Label 2 values distribution')

for bar, count in zip(bars, label2_value_counts.values):
    plt.annotate(str(count), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')


plt.tight_layout()
plt.show()

scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

pca = PCA(n_components=0.95, svd_solver = 'full')
x_train_pca = pca.fit_transform(x_train_scaled)
x_val_pca = pca.transform(x_val_scaled)
x_test_pca = pca.transform(x_test_scaled)

scores = []
for model_name, model in getModels().items():
    fold_scores = cross_val_score(model, x_train_pca, y_train['label_2'], cv=3, scoring='accuracy', n_jobs = -1)
    scores.append({
        'model' : model_name,
        'score' : np.mean(fold_scores)
    })

df_raw = pd.DataFrame(scores, columns = ['model', 'score'])
df_raw

model_data = {
    'svm' : {
        'model':svm.SVC(gamma = 'auto'),
        'params' : {
            'C': [1, 10, 100],
            'kernel': ['rbf', 'linear']}
    },
    'knn' : {
        'model' : KNeighborsClassifier(),
        'params' : {
            'n_neighbors' : list(range(1,5)),
            'p' : [1,2]
        }
    }
}

accuracy_scores = []

for model_name, model_info in model_data.items():
    clf = RandomizedSearchCV(model_info['model'], param_distributions = model_info['params'], n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    clf.fit(x_train_pca, y_train['label_2'])
    accuracy_scores.append({
      'model' : model_name,
      'best_score' : clf.best_score_,
      'best_params' : clf.best_params_
  })

df_hyp_tuning = pd.DataFrame(accuracy_scores, columns = ['model', 'best_score', 'best_params'])
df_hyp_tuning

best_model_l2 = KNeighborsClassifier(n_neighbors = 1, p = 2)
best_model_l2.fit(x_train_pca, y_train['label_2'])

#validation
fold_scores_l2 = cross_val_score(best_model_l2, x_val_pca, y_valid['label_2'], cv=3, scoring='accuracy', n_jobs = -1)
np.mean(fold_scores_l2)

y_test_label2 = best_model_l2.predict(x_test_pca)
y_test_label2.shape

## Label 3 ##

train['label_3'].isnull().sum()

valid['label_3'].isnull().sum()

x_train = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
x_valid = valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
x_test = test.drop(['ID'], axis=1)

y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]
y_valid = valid[['label_1', 'label_2', 'label_3', 'label_4']]

ax = sn.countplot(x=y_train['label_3'])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9, color='black')

plt.xlabel('Label 3')

plt.show()

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

pca = PCA(n_components=0.95, svd_solver = 'full')
x_train_pca = pca.fit_transform(x_train_scaled)
x_val_pca = pca.transform(x_valid_scaled)
x_test_pca = pca.transform(x_test_scaled)

scores = []
for model_name, model in getModels().items():
    fold_scores = cross_val_score(model, x_train_pca, y_train['label_3'], cv=3, scoring='accuracy', n_jobs = -1)
    scores.append({
        'model' : model_name,
        'score' : np.mean(fold_scores)
    })

df_raw = pd.DataFrame(scores, columns = ['model', 'score'])
df_raw

model_data = {
    'svm' : {
        'model':svm.SVC(gamma = 'auto'),
        'params' : {
            'C': [1, 10, 100],
            'kernel': ['rbf', 'linear']}
    },
    'logistic_regression' : {
        'model':LogisticRegression(solver = 'liblinear', multi_class = 'auto'),
        'params' : {
            'C': [1, 10, 100]
        }
    }
}

accuracy_scores = []

for model_name, model_info in model_data.items():
  clf = RandomizedSearchCV(model_info['model'], param_distributions = model_info['params'], n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
  clf.fit(x_train_pca, y_train['label_3'])
  accuracy_scores.append({
      'model' : model_name,
      'best_score' : clf.best_score_,
      'best_params' : clf.best_params_
  })

df_hyp_tuning = pd.DataFrame(accuracy_scores, columns = ['model', 'best_score', 'best_params'])
df_hyp_tuning

best_model_l3 = svm.SVC(C = 100, kernel = 'rbf')
best_model_l3.fit(x_train_pca, y_train['label_3'])

#validation
fold_scores_l3 = cross_val_score(best_model_l3, x_val_pca, y_valid['label_3'], cv=3, scoring='accuracy', n_jobs = -1)
np.mean(fold_scores_l3)

y_test_label3 = best_model_l3.predict(x_test_pca)
y_test_label3.shape

## Label 4 ##

train['label_4'].isnull().sum()

valid['label_4'].isnull().sum()

x_train = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
x_valid = valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
x_test = test.drop(['ID'], axis=1)

y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]
y_valid = valid[['label_1', 'label_2', 'label_3', 'label_4']]

label4_value_counts = y_train['label_4'].value_counts()

plt.figure(figsize=(10, 6))
bars = plt.bar(label4_value_counts.index, label4_value_counts.values)


plt.xlabel('Label 4')
plt.ylabel('Count')
plt.title('Label 4 values distribution')

for bar, count in zip(bars, label4_value_counts.values):
    plt.annotate(str(count), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')


plt.tight_layout()
plt.show()

sampler = RandomOverSampler(random_state=0)
x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train['label_4'])

label4_value_counts = y_train_resampled.value_counts()

plt.figure(figsize=(10, 6))
bars = plt.bar(label4_value_counts.index, label4_value_counts.values)


plt.xlabel('Label 4')
plt.ylabel('Count')
plt.title('Label 4 values distribution after OverSampling')

for bar, count in zip(bars, label4_value_counts.values):
    plt.annotate(str(count), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')


plt.tight_layout()
plt.show()

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_resampled)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

pca = PCA(n_components=0.95, svd_solver = 'full')
x_train_pca = pca.fit_transform(x_train_scaled)
x_val_pca = pca.transform(x_valid_scaled)
x_test_pca = pca.transform(x_test_scaled)

scores = []
for model_name, model in getModels().items():
    fold_scores = cross_val_score(model, x_train_pca, y_train_resampled['label_4'], cv=3, scoring='accuracy', n_jobs = -1)
    scores.append({
        'model' : model_name,
        'score' : np.mean(fold_scores)
    })

df_raw = pd.DataFrame(scores, columns = ['model', 'score'])
df_raw

model_data = {
    'svm' : {
        'model':svm.SVC(gamma = 'auto'),
        'params' : {
            'C': [1, 10, 100],
            'kernel': ['rbf', 'linear']}
    },
    'logistic_regression' : {
        'model':LogisticRegression(solver = 'liblinear', multi_class = 'auto'),
        'params' : {
            'C': [1, 10, 100]
        }
    },
    'knn' : {
        'model' : KNeighborsClassifier(),
        'params' : {
            'n_neighbors' : list(range(1,5)),
            'p' : [1,2]
        }
    }
}

accuracy_scores = []

for model_name, model_info in model_data.items():
  clf = RandomizedSearchCV(model_info['model'], param_distributions = model_info['params'], n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
  clf.fit(x_train_pca, y_train_resampled['label_4'])
  accuracy_scores.append({
      'model' : model_name,
      'best_score' : clf.best_score_,
      'best_params' : clf.best_params_
  })

df_hyp_tuning = pd.DataFrame(accuracy_scores, columns = ['model', 'best_score', 'best_params'])
df_hyp_tuning

best_model_l4 = svm.SVC(C = 100, kernel = 'rbf')
best_model_l4.fit(x_train_pca, y_train_resampled)

#validation
fold_scores_l4 = cross_val_score(best_model_l4, x_val_pca, y_valid['label_4'], cv=3, scoring='accuracy', n_jobs = -1)
np.mean(fold_scores_l4)

y_test_label4 = best_model_l4.predict(x_test_pca)
y_test_label4.shape

final_df = test[['ID']]
final_df['label_1'] = y_test_label1
final_df['label_2'] = y_test_label2
final_df['label_3'] = y_test_label3
final_df['label_4'] = y_test_label4

final_df.to_csv('Outputs/Layer-10/solutions_layer10.csv', index=False)









