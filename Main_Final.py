from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import *
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

# input path for X and y
input_path_X = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx'
input_path_y = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx'

# loading X and y
X = pd.read_excel(input_path_X, index_col=0)
y = pd.read_excel(input_path_y, index_col=0)

# casting y to a list
y = list(y[0])

# removing empty columns
X = X.loc[:, X.any()]

# droping non-numeric columns
numeric_features = X.select_dtypes(include=np.number)
X = X.loc[:, numeric_features.columns]

# filling na's with columns means
X = X.apply(lambda l: l.fillna(l.mean()), axis=0)

## EDA

print(X.info())

print(X.describe())

def get_scattered_chunks(
    data: pd.DataFrame, n_chunks: int = 5, chunk_size: int = 3
) -> pd.DataFrame:
    """
    Returns a subsample of equally scattered chunks of rows.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    n_chunks : int
        Number of chunks to collect for the subsample
    chunk_size : int
        Number of rows to include in each chunk

    Returns
    -------
    pd.DataFrame
        Subsample data
    """

    endpoint = len(data) - chunk_size
    sample_indices = np.linspace(0, endpoint, n_chunks, dtype=int)
    sample_indices = [
        index for i in sample_indices for index in range(i, i + chunk_size)
    ]
    return data.iloc[sample_indices, :]


data_chunks = get_scattered_chunks(X, n_chunks=5, chunk_size=3)
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(data_chunks)


# plotting the heatmap of the complete dataset
correlation_matrix = X.corr()
fig, ax = plt.subplots(figsize=(13, 13))
ax.set_title("Heatmap of the complete dataset")
_ = sns.heatmap(correlation_matrix)
plt.show()

# plotting class distribution before over-sampling
fig, ax = plt.subplots()
class_distribution = pd.Series(y).value_counts(normalize=True)
ax = class_distribution.plot.barh()
ax.set_title("Class distribution",  fontsize=14)
pos_label = class_distribution.idxmin()
plt.tight_layout()
plt.show()
print("\n The positive label considered as the minority class is: odor classified as sour\n")

# stratify splitting the data- test size of 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)


def find_optimal_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Running a grid search cv on the the data to find the optimal model.
    Models used: Random Forest, SVM, Logistic Regression and XGboost.
    Then testing the optimal model on the test data, plotting confusion matrix and permutation feature importance.
    """
    # scaling the training data (without the validation data)
    scaler = StandardScaler()

    # over-sampling using ADASYN
    sampler = ADASYN(random_state=1)

    # Initialze the estimators for random forest, SVM, logistic regression and XGboost
    clf1 = RandomForestClassifier(random_state=42)
    clf2 = SVC(probability=True, random_state=42, kernel="rbf")
    clf3 = LogisticRegression(random_state=42, max_iter=10000)
    clf4 = XGBClassifier(objective='binary:logistic', seed=42, use_label_encoder=False, eval_metric='logloss')
    
    # Initiaze the hyperparameters for each dictionary

    # parameters for random forest
    param1 = {}
    param1['classifier__n_estimators'] = [50, 100]
    param1['classifier__max_depth'] = [2, 5]
    param1['classifier__min_samples_split'] = [2, 70]
    param1['classifier__min_samples_leaf'] = [1, 50]
    param1['classifier'] = [clf1]

    # parameters for SVM
    param2 = {}
    param2['classifier__C'] = [2e-3, 2e7]
    param2['classifier__gamma'] = [2e-7, 2e3]
    param2['classifier'] = [clf2]

    # parameters for logistic regression
    param3 = {}
    param3['classifier__C'] = [100, 10, 1.0, 0.1, 0.01]
    param3['classifier__penalty'] = ['l1', 'l2']
    param3['classifier__solver'] = ['saga', 'liblinear']
    param3['classifier'] = [clf3]

    # parameters for XGboost
    param4 = {}
    param4['classifier__max_depth'] = [2, 5]
    param4['classifier__min_child_weight'] = [1, 6]
    param4['classifier__gamma'] = [0.1, 10]
    param4['classifier__reg_alpha'] = [0.1, 20]
    param4['classifier__reg_lambda'] = [0.001, 100]
    param4['classifier__learning_rate'] = [0.01, 1]
    param4['classifier__n_estimators'] = [10, 200]
    param4['classifier'] = [clf4]

    # making a pipeline of scaling, over-sampling and hyperparameter tuning.
    pipeline = Pipeline([
        ('scaler', scaler),
        ('sampler', sampler),
        ('classifier', clf1)
    ])
    params = [param1, param2, param3, param4]

    # Stratified K-Fold
    stratified_kfold = StratifiedKFold(n_splits=5,
                                       shuffle=True,
                                       random_state=1)

    # training the grid search model
    gs = GridSearchCV(pipeline, params, cv=stratified_kfold, n_jobs=-1, scoring='f1', verbose=1).fit(X_train, y_train)

    # printing best performing model and its corresponding hyperparameters
    print("The best model is:\n ", gs.best_params_)

    # printing the f1 score for the best model on training data
    print("\n The f1 score of the model on training data:\n ", "{:.4f}".format((gs.best_score_)))

    # predicting using the best model with X_test
    y_pred = gs.predict(X_test)

    # printing the f1 score for the best model on test data
    print("\n The f1 score of the model on test data:\n ", "{:.4f}".format(f1_score(y_test, y_pred)))

    # printing classification report
    report = classification_report(y_test, y_pred)
    print("\n classification report:\n ", report)

    # plotting confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap=plt.cm.Greens)
    plt.title(f"Confusion Matrix")
    plt.xlabel("Predicted label", fontsize=15)
    plt.ylabel("True label", fontsize=15)
    plt.show()
      
    # calculating and plotting feature permutation importance 
    r = permutation_importance(gs, X_test, y_test, n_repeats=30, random_state=1, scoring="f1")
    
    fig_importances = pd.Series(r.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    fig_importances.plot.bar(yerr=r.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    print("Sorted feature permutation importance: \n")
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]:<8}  "
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")           


# running the function on the dataset
feat_names = list(X.columns)
print("Searching for the optimal model \n")
find_optimal_model(X_train, X_test, y_train, y_test, feat_names)

# initialize Boruta feature selection
forest = RandomForestRegressor(
   n_jobs=-1,
   max_depth=5
)

print("\n Running Boruta feature selection...\n")

boruta = BorutaPy(
   estimator=forest,
   n_estimators='auto',
   max_iter=500,
   random_state=1
)

# fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X_train), np.array(y_train))

# print Boruta results
green_area = X.columns[boruta.support_].to_list()
blue_area = X.columns[boruta.support_weak_].to_list()
print('Features in the green area:\n', green_area)
print('Features in the blue area:\n', blue_area)

# call transform() on X_train to filter it down to selected features
X_train_boruta = boruta.transform(X_train.values)

# call transform() on X_test to filter it down to selected features
X_test_boruta = boruta.transform(X_test.values)

# plotting heatmap after Boruta feature selection
df_boru = pd.DataFrame(X_train_boruta)
df_boru.columns = green_area
cor_matrix = df_boru.corr()
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_title("Heatmap after Boruta", fontsize=16)
_ = sns.heatmap(cor_matrix, annot=True)
plt.show()

# running the function on the dataset after Boruta feature selection
print("\n Searching for the optimal model after Boruta feature selection \n")
find_optimal_model(X_train_boruta, X_test_boruta, y_train, y_test, green_area)