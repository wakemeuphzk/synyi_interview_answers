from sklearn import feature_selection
model_data_path = 'E:\synyi\synyi_interview-master\model_data.csv'
model_path = 'E:\synyi\synyi_interview-master\model'
model_result = 'E://synyi//synyi_interview-master//result//prediciton.csv'
import pandas as pd

#分析数据，查看数据分布，各个特征情况
def data_analysis(data):
    print('origin_data head:', data.head())
    print('origin_data:',data.info())
    print(data['dataset'].value_counts())  #train:3000   ready2predict:793
    print('null label: ',data['label'].isnull().value_counts())
    train_test_data = data[~(data['label'].isnull())]
    print(train_test_data['label'].value_counts())   #0:1 = 2684:316
    train_test_data_x = train_test_data.drop(['label', 'id', 'dataset'], axis=1)
    print('x:',train_test_data_x)
    print('handle null x:',handle_null(train_test_data_x))
    y = train_test_data['label']
    print('y',y)

# 剔除特征：缺失率达到90%以上的特征去除
# 处理空值:均值填充
# 再选择20个特征(1%左右的特征)
def select_features(data_df,y):
    total_rows = data_df.shape[0]
    # print('total_rows:',total_rows)
    # 去除缺失率90%以上的特征
    for column in list(data_df.columns[data_df.isnull().sum() > 0.90 * total_rows]):
        data_df = data_df.drop([column], axis=1)
    print(data_df)
    data_df = handle_null(data_df)
    #fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=1)
    #data_df = fs.fit_transform(data_df,y)
    print(data_df)
    return data_df

# 处理空值:均值填充
def handle_null(data):
    data_df = data.drop(['label', 'id', 'dataset'], axis=1)
    #print(data_df.isnull().sum() > 0)
    #print(data_df.columns[data_df.isnull().sum() > 0])
    for column in list(data_df.columns[data_df.isnull().sum() > 0]):
        mean_val = data_df[column].mean()
        #print('mean',mean_val)
        data_df[column].fillna(mean_val, inplace=True)
    print(data_df)
    data_df = data_df.fillna(0.0)
    return data_df

def get_handled_data(train_test_data):
    data_df_x = train_test_data.drop(['label', 'id', 'dataset'], axis=1)
    data_df_x = select_features(data_df_x)
    return data_df_x

def get_train_data():
    data_df = get_origin_data(model_data_path)
    train_test_data = data_df[~(data_df['label'].isnull())]
    data_df_x = get_handled_data(train_test_data)
    train_test_data_x = data_df_x
    print(train_test_data_x)
    print(train_test_data['label'].value_counts())
    return train_test_data_x,train_test_data['label']
    #x_train, x_test, y_train, y_test = train_test_split(train_test_data_x, train_test_data['label'], test_size=0.15, random_state=1234565)
    #return x_train,y_train

def get_predict_data():
    data_df = get_origin_data(model_data_path)
    predict_data = data_df[data_df['label'].isnull()]
    data_df_x = get_handled_data(predict_data)

    print(data_df_x)
    print(predict_data['id'])
    return data_df_x,predict_data['id']

def get_origin_data(data_path):
    return pd.read_csv(data_path)

if __name__ == '__main__':
    data_df = get_origin_data(model_data_path)
    #data_analysis(data_df)
    get_train_data()
    get_predict_data()


