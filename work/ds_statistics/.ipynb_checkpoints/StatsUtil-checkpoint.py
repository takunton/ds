import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# チップ支払額dfを取得
def get_tip_df():
    # seabornライブラリのサンプルデータ（チップ支払額）を取得
    result = sns.load_dataset('tips')

    # 列：代金とチップの割合を追加
    result['tip_rate'] = result['tip'] / result['total_bill']
    
    return result

# クラメールの連関係数を計算
def cramers_v(x, y):
    # x2を計算
    cont_table = pd.crosstab(x, y)
    x2 = stats.chi2_contingency(cont_table, correction=False)[0]

    # min(a,b)を計算
    min_ab = min(cont_table.shape)
    
    # クラメールの連関係数計算
    v = np.sqrt(x2 / ((min_ab - 1) * len(x)))
    return v

# グラフ描画用のサンプル関数
def sample_func(x):
    return x**2 + 5

# 正規分布の区間ごとの割合を表示
def norm_percent(mean, std, num):
    samples = stats.norm(loc=mean, scale=std).rvs(num)
    
    for i in range(1, 4):
        s_cnt = np.count_nonzero((samples >= mean - std * i) & (samples <= mean + std * i))
        print('s{} : {}%'.format(i, s_cnt/num))

# 平均と分散の標本平均値を表示
def calc_sample_mean(df, loop, sample_num):
    mean_array = []

    for _ in range(loop):
        # サンプルデータ取得
        sample_data = df.sample(sample_num)
        mean_array.append(sample_data.mean())
        
    sample_mean = np.mean(mean_array)
    sample_var = np.var(mean_array)

    print('標本平均:{}'.format(sample_mean))
    print('標本分散:{}'.format(sample_var))
    print('\r')
    print('母平均　:{}'.format(np.mean(df)))
    print('母分散　:{}'.format(np.var(df)/sample_num))
    
# 分散と不偏分散の標本平均値を表示
def calc_sample_var(df, loop, sample_num):
    var_array = []
    uvar_array = []

    for _ in range(loop):
        # サンプルデータ取得
        sample_data = df.sample(sample_num)
        # 分散
        var_array.append(np.var(sample_data))
        # 不偏分散
        uvar_array.append(stats.tvar(sample_data))

    sample_var = np.mean(var_array)
    sample_uvar = np.mean(uvar_array)

    print('分散平均　　:{}'.format(sample_var))
    print('不偏分散平均:{}'.format(sample_uvar))
    print('\r')
    print('母分散　　　:{}'.format(np.var(df)))

# 母比率の区間推定の精度を試算
def sim_rate(df, column, item, loop, sample_num, ci):
    result_array = []
    
    # 母比率を計算
    pupulaction_ratio = len(df[df[column] == item]) / len(df)
    
    # 標本比率から区間推定
    sim_df = df[column]

    for _ in range(loop):
        # サンプルデータ取得
        sample_data = sim_df.sample(sample_num)
        # 標本比率を計算
        sample_ratio = len(sample_data[sample_data == item]) / sample_num
        # 信頼区間（人数）を取得
        min_num, max_num = stats.binom.interval(ci, sample_num, sample_ratio)
        # 信頼区間（割合）を取得
        min_ratio = min_num / sample_num
        max_ratio = max_num / sample_num

        if(min_ratio <= pupulaction_ratio and pupulaction_ratio <= max_ratio):
            result_array.append(1)
        else:
            result_array.append(0)
            
    # デバック用ロジック
    # print('標本比率:{},信頼区間:{}~{}'.format(sample_ratio, min_ratio, max_ratio))
    
    print('【推定結果】成功:{},失敗:{}'.format(result_array.count(True),result_array.count(False)))

# 母平均の区間推定の精度を試算
def sim_mean(df, column, loop, sample_num, ci):
    result_array = []
    
    # 母平均を計算
    pupulation_mean = df[column].mean()
    
    # 標本比率から区間推定
    sim_df = df[column]

    for _ in range(loop):
        # サンプルデータ取得
        sample_data = sim_df.sample(sample_num)
        # 標本平均を計算
        sample_mean = sample_data.mean()
        # 標本標準偏差を計算
        sample_std = np.sqrt(stats.tvar(df[column]) / sample_num)
        # 信頼区間を取得
        min_val, max_val = stats.norm.interval(ci, sample_mean, sample_std)

        if(min_val <= pupulation_mean and pupulation_mean <= max_val):
            result_array.append(1)
        else:
            result_array.append(0)
            
        # デバック用ロジック
        # print('標本平均:{},信頼区間:{}~{}'.format(cor_mean, min_num, max_num))
    
    print('【推定結果】成功:{},失敗:{}'.format(result_array.count(True),result_array.count(False)))

# 母平均の区間推定（t分布）の精度を試算
def sim_mean_t(df, column, loop, sample_num, ci):
    result_array = []
    
    # 母平均を計算
    pupulation_mean = df[column].mean()
    
    # 標本比率から区間推定
    sim_df = df[column]

    for _ in range(loop):
        # サンプルデータ取得
        sample_data = sim_df.sample(sample_num)
        # 標本平均を計算
        sample_mean = sample_data.mean()
        # 標本標準偏差を計算
        sample_std = np.sqrt(stats.tvar(df[column]) / sample_num)
        # 信頼区間を取得
        min_val, max_val = stats.t.interval(ci, sample_num - 1, sample_mean, sample_std)

        if(min_val <= pupulation_mean and pupulation_mean <= max_val):
            result_array.append(1)
        else:
            result_array.append(0)
            
        # デバック用ロジック
        # print('標本平均:{},信頼区間:{}~{}'.format(sample_mean, min_val, max_val))
    
    print('【推定結果】成功:{},失敗:{}'.format(result_array.count(True),result_array.count(False)))

# F分布値を取得
def get_f(x, dfn, dfd):
    return stats.f(dfn=dfn, dfd=dfd).pdf(x)

# Cohen's dを計算
def cohen_d(x, y):
    # 共通不偏分散
    tvar = (len(x) - 1) * stats.tvar(x) + (len(y) - 1) * stats.tvar(y)
    dof = len(x) + len(y) - 2
    std = np.sqrt(tvar / dof)
    
    result = (np.abs(x.mean() - y.mean())) / std
    return result