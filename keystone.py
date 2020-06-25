import csv, json, sys, redis, os
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime
from slacker import Slacker

from sensitives import SENSITIVES

slack = Slacker(SENSITIVES['slack_token'])
r = redis.Redis(host=SENSITIVES['redis_ip'], port=6379, password=SENSITIVES['redis_pw'])

bm_symbols = [
    'KS11',  # 코스피
    'KQ11',  # 코스닥
    'SPX',   # SP500
    'DJI',   # Dow Jones
    'IXIC'   # Nasdaq
]

emerging_market_symbols = [
    'DBB',   # 실물경기 / 산업재 금속 가격
    'PICK',  # 실물경기 / 산업재 금속 가격
    'MXI',   # 실물경기 / 원자재 가격
    'GSG',   # 실물경기 / 원자재 가격
    'HEDJ',  # 실물경기 / 수출 지표
    'FLM',   # 기업경기 / 건설업
    'EXI',   # 기업경기 / 제조업
    'SOXX',  # 기업경기 / 기술업
    'IXN',   # 기업경기 / 기술업
    'CEW',   # 신흥국 통화
    'JNK',   # 위험
    'VXEEM'  # 위험
]

weights = {
    'DBB': (0.25 / 3) / 2,
    'PICK': (0.25 / 3) / 2,
    'MXI': (0.25 / 3) / 2,
    'GSG': (0.25 / 3) / 2,
    'HEDJ': (0.25 / 3),
    'FLM': (0.25 / 3),
    'EXI': (0.25 / 3),
    'SOXX': (0.25 / 3) / 2,
    'IXN': (0.25 / 3) / 2,
    'CEW': 0.25,
    'JNK': (0.25 / 2),
    'VXEEM': (0.25 / 2)
}

start = '1990-01-01'
end = datetime.now().strftime('%Y-%m-%d')

def keyst_update_data():
    merge_df = None
    cnt = 0
    for symbol in emerging_market_symbols:
        df = fdr.DataReader(symbol, start, end)
        df.rename(columns={'Close': symbol}, inplace=True)
        if cnt == 0:
            merge_df = df[symbol]
        else:
            merge_df = pd.concat([merge_df, df[symbol]], axis=1)
        cnt += 1
        print('{} DONE'.format(symbol))
    try:
        merge_df.to_csv('./task_data/keystone.csv')
    except:
        os.mkdir('./task_data')
        merge_df.to_csv('./task_data/keystone.csv')

def keyst_update_bm():
    for sym in bm_symbols:
        idx = fdr.DataReader(sym, '1990-01-01')
        idx.rename(columns={
            'Close': '{}_c'.format(sym),
            'Volume': '{}_v'.format(sym)
        }, inplace=True)
        idx = idx[['{}_c'.format(sym), '{}_v'.format(sym)]]
        idx.index = pd.Series(idx.index).apply(lambda x: x.strftime('%Y%m%d'))
        redis_res = r.set(sym, json.dumps(idx.to_dict()))
        print('{} DONE'.format(sym))

def keyst_make_index():
    df = pd.read_csv('./task_data/keystone.csv')
    df.index = pd.to_datetime(df.Date)
    df.drop('Date', axis=1, inplace=True)
    df.fillna(method='ffill', inplace=True)
    ret = df.pct_change()
    yc = (ret + 1).cumprod().dropna()

    w = 60
    avg_ret = ret.dot(pd.Series(weights))
    avg_ret_mean = avg_ret.rolling(window=w).mean()
    avg_ret_std = avg_ret.rolling(window=w).std()
    avg_ret_z = (avg_ret - avg_ret_mean) / avg_ret_std
    index = avg_ret_z.rolling(window=w).mean()
    index.dropna().to_csv('./task_data/index.csv')

def keyst_cache_index():
    df = pd.read_csv('./task_data/index.csv')
    df.index = df.Date.apply(lambda x: str(x).replace('-', ''))
    df.drop('Date', axis=1, inplace=True)
    df.columns = ['KeystIndex']
    json_data = json.dumps(df.to_dict())
    redis_res = r.set('KeystIndex', json_data)

def keyst_scale_cache_index():
    sym = 'KS11'
    key = pd.read_csv('./task_data/index.csv')
    df = pd.DataFrame(json.loads(r.get(sym)))

    key.columns = ['Date', 'Index']
    key.index = pd.to_datetime(key.Date)
    key.drop('Date', axis=1, inplace=True)

    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df = df[key.index[0]:key.index[-1]]
    df['Index'] = key['Index']

    ret = df.pct_change()
    yc = (ret + 1).fillna(1).cumprod()

    bm_max_val = yc['{}_c'.format(sym)].max()
    bm_min_val = yc['{}_c'.format(sym)].min()
    bm_new_min_val = bm_max_val - 1*bm_min_val
    multiple = bm_max_val - bm_new_min_val

    key_max_val = yc['Index'].max()
    key_min_val = yc['Index'].min()
    yc['Index_scaled'] = (yc['Index'] - key_min_val) / (key_max_val - key_min_val)
    yc['Invest_amt'] = 1 - yc['Index_scaled']
    yc['Index_scaled'] = (yc['Index_scaled'] * multiple) + bm_new_min_val
    yc['Index_low_band'] = yc['Index_scaled'].rolling(120).min()
    yc['Index_high_band'] = yc['Index_scaled'].rolling(120).max()

    to_save_df = yc[['KS11_c', 'Index_scaled']]
    to_save_df.index = to_save_df.index.strftime('%Y%m%d')
    json_data = json.dumps(to_save_df.to_dict())
    redis_res = r.set('KeystScaledIndexWithBM', json_data)

def keyst_cache_index_factors():
    df = pd.read_csv('./task_data/keystone.csv')
    df.index = pd.to_datetime(df.Date)
    df.drop('Date', axis=1, inplace=True)
    df.fillna(method='ffill', inplace=True)
    ret = df.pct_change()

    ret['real_econ'] = 0
    for sym in ['DBB', 'PICK', 'MXI', 'GSG', 'HEDJ']:
        ret['real_econ'] = ret['real_econ'] + (ret[sym] * weights[sym] * 4)
        
    ret['ind_econ'] = 0
    for sym in ['FLM', 'EXI', 'SOXX', 'IXN']:
        ret['ind_econ'] = ret['ind_econ'] + (ret[sym] * weights[sym] * 4)
        
    ret['currency'] = ret['CEW'] * weights['CEW'] * 4

    ret['risk'] = 0
    for sym in ['JNK', 'VXEEM']:
        ret['risk'] = ret['risk'] + (ret[sym] * weights[sym] * 4)
        
    index_factors = ret[['real_econ', 'ind_econ', 'currency', 'risk']].dropna()
    index_factors.index = index_factors.index.strftime('%Y%m%d')
    index_factors = (index_factors + 1).cumprod()
    json_data = json.dumps(index_factors.to_dict())
    redis_res = r.set('KeystIndexFactors', json_data)

# tasks
slack.chat.post_message('#blended-keystone-data', 'Keystone 신흥국 지표 task 시작')
keyst_update_data()
keyst_update_bm()
keyst_make_index()
keyst_cache_index()
keyst_scale_cache_index()
keyst_cache_index_factors()
slack.chat.post_message('#blended-keystone-data', 'Keystone 신흥국 지표 task 완료')