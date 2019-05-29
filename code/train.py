from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.externals import joblib
import data_util

def train_by_lr(x_train,y_train):
    clf = LogisticRegression(C=1.0,penalty='l1',tol=0.01)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 500],
        'penalty':['l1','l2']
    }
    clf = train_by_gridsearchcv(x_train, y_train, clf, param_grid)
    print('lr best_param:', clf.best_params_) #{'C': 10, 'penalty': 'l1'}
    joblib.dump(clf, data_util.model_path + '\\lr_model.m')

def train_by_rf(x_train,y_train):
    clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=4)
    param_grid = {
        "n_estimators": [10, 100, 200,500,1000],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [2, 4, 6,8],
    }
    clf = train_by_gridsearchcv(x_train, y_train, clf, param_grid)
    print('rf best_param:', clf.best_params_)#{'criterion': 'entropy', 'min_samples_leaf': 2, 'n_estimators': 200}
    joblib.dump(clf, data_util.model_path + '\\rf_model.m')

def train_by_xgboost(x_train,y_train):
    clf = XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        gamma=0.5,
        subsample=0.6,
        colsample_bytree=0.6,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27
    )

    param_grid = {
        'max_depth':range(3,10,2),  #range(3,15,2)
        'min_child_weight':range(1,7,2), #range(1,9,2)
        'learning_rate':[0.01,0.1,1],#[0.01,0.05,0.1,0.3,0.5,1],
        'n_estimators': [100, 500, 1000] #[100, 500, 1000, 1500, 2000]
    }
    clf = train_by_gridsearchcv(x_train,y_train,clf,param_grid)
    print('xgboost best_param:',clf.best_params_)
    joblib.dump(clf,data_util.model_path+'\\xgboost_model.m')

def train_by_gridsearchcv(x_train,y_train,clf,param_grid):
    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=7)
    score = 'roc_auc'
    clf = GridSearchCV(clf,param_grid=param_grid,cv=cv,scoring=score,n_jobs=-1)
    clf.fit(x_train,y_train)
    return clf

if __name__ == '__main__':
    x_train,y_train = data_util.get_train_data()
    train_by_lr(x_train,y_train)
    train_by_rf(x_train,y_train)
    train_by_xgboost(x_train,y_train)
