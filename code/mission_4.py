
history_file = 'E:\synyi\synyi_interview-master\history_data.csv'
sample_file = 'E:\synyi\synyi_interview-master\sample.csv'
result_file = 'E://synyi//synyi_interview-master//result//mission_4_result.csv'
import pandas as pd

def generate_old(history,sample):
    # for index, row in sample.iterrows():
    #     merge_row = pd.merge(history, pd.DataFrame(row,['id','obs_time']), how='left', on=['id']) #row["c1"], row["c2"]
    #     print(merge_row)
    merge_all = pd.merge(history, sample, how='left', on=['id'])
    #print(merge_all)
    merge_all['days_interval'] = pd.to_datetime(merge_all['obs_time']) - pd.to_datetime(merge_all['time'])
    print(merge_all['days_interval'])
    merge_10 = merge_all[pd.Timedelta(days=0)<=merge_all['days_interval'] & merge_all['days_interval']<=pd.Timedelta(days=10)]
    merge_60 = merge_all[pd.Timedelta(days=0)<=merge_all['days_interval'] & merge_all['days_interval'] <= pd.Timedelta(days=60)]
    #print('merge_1_10:',merge_1_10)

    #merge_all['10_days_before_obs_time'] = merge_all['obs_time'] - timedelta(days=10)
    #merge_all['60_days_before_obs_time'] = merge_all['obs_time'] - timedelta(days=60)
    #print(merge_all)
    grouped_10_A = merge_10['A'].groupby(merge_all['id'])
    grouped_60_A = merge_60['A'].groupby(merge_all['id'])
    grouped_10_B = merge_10['B'].groupby(merge_all['id'])
    grouped_60_B = merge_60['B'].groupby(merge_all['id'])
    print(grouped_10_A.mean())
    print(grouped_10_B.mean())
    print(grouped_10_A.isnull().sum())
    print(grouped_10_B.mean())
    print(grouped_60_A.mean())
    print(grouped_60_B.mean())
    print(grouped_60_A.mean())
    print(grouped_60_B.mean())


def generate_result(history,sample):
    #左连接
    merge_all = pd.merge(history, sample, how='left', on=['id'])
    #print(merge_all)
    #日期间隔
    merge_all['days_interval'] = pd.to_datetime(merge_all['obs_time']) - pd.to_datetime(merge_all['time'])
    print(merge_all['days_interval'])
    interval_format = lambda x: int(str(x).split(' days')[0])
    merge_all['days_interval'] = merge_all['days_interval'].apply(interval_format)
    print(merge_all['days_interval'])
    #10日内
    merge_10 = merge_all[(merge_all['days_interval'] >= 0) & (merge_all['days_interval'] <= 10)]
    #60日内
    merge_60 = merge_all[(merge_all['days_interval'] >= 0) & (merge_all['days_interval'] <= 60)]

    #根据sample记录来分组
    merge_all_group_by_10 = merge_10.groupby(['id', 'obs_time'])
    merge_all_group_by_60 = merge_60.groupby(['id', 'obs_time'])

    #统计
    print(merge_all_group_by_10['A'].mean())
    print(merge_all_group_by_10['A'].count())
    print(merge_all_group_by_60['A'].mean())
    print(merge_all_group_by_60['A'].count())
    print(merge_all_group_by_10['B'].mean())
    print(merge_all_group_by_10['B'].count())
    print(merge_all_group_by_60['B'].mean())
    print(merge_all_group_by_60['B'].count())

    result = pd.concat([merge_all_group_by_10['A'].mean(),merge_all_group_by_10['A'].count(),
                     merge_all_group_by_60['A'].mean(),merge_all_group_by_60['A'].count(),
                     merge_all_group_by_10['B'].mean(), merge_all_group_by_10['B'].count(),
                     merge_all_group_by_60['B'].mean(), merge_all_group_by_60['B'].count()
                     ],axis=1)

    result.columns = ['A_mean_10_days','A_not_null_count_10_days','A_mean_60_days','A_not_null_count_60_days','B_mean_10_days','B_not_null_count_10_days','B_mean_60_days','B_not_null_count_60_days']

    # 写入文件
    result.to_csv(result_file)

if __name__ == '__main__':
    history_data = pd.read_csv(history_file, parse_dates=['time'])
    print(history_data['time'])
    sample_data = pd.read_csv(sample_file)
    print(sample_data['obs_time'])
    generate_result(history_data, sample_data)