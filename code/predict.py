from sklearn.externals import joblib
import data_util
import pandas as pd

def predict_by_lr(x_predict):
    clf = joblib.load(data_util.model_path + '\\lr_model.m')
    return clf.predict_proba(x_predict)

def predict_by_rf(x_predict):
    clf = joblib.load(data_util.model_path + '\\rf_model.m')
    return clf.predict_proba(x_predict)

def predict_by_xgboost(x_predict):
    clf = joblib.load(data_util.model_path + '\\xgboost_model.m')
    return clf.predict_proba(x_predict)


if __name__ == '__main__':
    x_predict, id_predict = data_util.get_predict_data()
    prob_lr = predict_by_lr(x_predict)
    print('prob_lr:', prob_lr)
    print('prob_lr:',prob_lr[0])
    print('prob_lr:', prob_lr[:,1])
    prob_rf = predict_by_rf(x_predict)
    prob_xgboost = predict_by_xgboost(x_predict)
    result = pd.concat([id_predict,pd.DataFrame(prob_lr[:,0]),pd.DataFrame(prob_xgboost[:,0]),
                        pd.DataFrame(prob_rf[:,0])],axis=1)
    print(result.columns)
    result.columns = ['id','lr','xgboost','rf']
    print(result)
    result.to_csv(data_util.model_result,index=None)
