import os
import datetime
import numpy as np
import pandas as pd
import scipy
import scipy.signal
pd.options.display.max_columns = None    # disp all columns


def make_features(DIR, data_str, train, param):
    feature_set = []
    fs = param['fs']
    N = param['N']
    max_f = param['max_f']
    delta_f = param['delta_f']
    delta_t = param['delta_t']
    fft_win = param['fft_win']
    j = 0
    for segment_id in train['segment_id']:
        segment_df = pd.read_csv(os.path.join(DIR, f'{data_str}\\{segment_id}.csv'))
        segment = [segment_id]
        j = j+1
        if j % 500 == 1:
            print(j)
        for sensor in segment_df.columns:
            x = segment_df[sensor][:N]
            # if x.isna().sum() > 1000:     ##########
            #     segment += ([np.NaN] * 10)
            #     continue
            f, t, Z = scipy.signal.stft(x.fillna(0), fs=fs, window='hann', nperseg=fft_win)
            f = f[:round(max_f/delta_f)+1]
            Z = np.abs(Z[:round(max_f/delta_f)+1]).T    # ～max_f, row:time,col:freq

            th = Z.mean() * 1     ##########
            Z_pow = Z.copy()
            Z_pow[Z < th] = 0
            Z_num = Z_pow.copy()
            Z_num[Z >= th] = 1

            Z_pow_sum = Z_pow.sum(axis=0)
            Z_num_sum = Z_num.sum(axis=0)

            A_pow = Z_pow_sum[round(10/delta_f):].sum()
            A_num = Z_num_sum[round(10/delta_f):].sum()
            BH_pow = Z_pow_sum[round(5/delta_f):round(8/delta_f)].sum()
            BH_num = Z_num_sum[round(5/delta_f):round(8/delta_f)].sum()
            BL_pow = Z_pow_sum[round(1.5/delta_f):round(2.5/delta_f)].sum()
            BL_num = Z_num_sum[round(1.5/delta_f):round(2.5/delta_f)].sum()
            C_pow = Z_pow_sum[round(0.6/delta_f):round(1.2/delta_f)].sum()
            C_num = Z_num_sum[round(0.6/delta_f):round(1.2/delta_f)].sum()
            D_pow = Z_pow_sum[round(2/delta_f):round(4/delta_f)].sum()
            D_num = Z_num_sum[round(2/delta_f):round(4/delta_f)].sum()
            segment += [A_pow, A_num, BH_pow, BH_num, BL_pow, BL_num, C_pow, C_num, D_pow, D_num]
        feature_set.append(segment)

    cols = ['segment_id']
    for i in range(10):
        for j in ['A_pow', 'A_num','BH_pow', 'BH_num','BL_pow', 'BL_num','C_pow', 'C_num','D_pow', 'D_num']:
            cols += [f's{i+1}_{j}']
    feature_df = pd.DataFrame(feature_set, columns=cols)
    feature_df['segment_id'] = feature_df['segment_id'].astype('int')
    return feature_df


if __name__ == '__main__':
    DIR = "D:\\DataSetAll\\INGV"
    train = pd.read_csv(os.path.join(DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DIR, 'sample_submission.csv'))
    FE_path = ".\\FEdata\\model_trainFE1\\"
    if not os.path.exists(FE_path):
        os.mkdir(FE_path)

    fs = 100  # sampling frequency
    N = 60001  # data size
    fft_win = 256  # FFT segment size
    max_f = 20  # ～20Hz
    delta_f = fs / fft_win  # 0.39Hz
    delta_t = fft_win / fs / 2  # 1.28s
    param = {'fs': fs, 'N': N, 'fft_win': fft_win, 'max_f': max_f, 'delta_f': delta_f, 'delta_t': delta_t}

    # STFT
    Train_file_name = os.path.join(FE_path, f'train_set.csv')
    print('-----------训练集特征提取-----------')
    train_fe = make_features(DIR=DIR, data_str='train', train=train, param=param)
    train_set = pd.merge(train, train_fe, on='segment_id')
    train_set.to_csv(Train_file_name, index=False)

    Test_file_name = os.path.join(FE_path, f'train_set.csv')
    print('-----------测试集特征提取-----------')
    test_fe = make_features(DIR=DIR, data_str='test', train=test, param=param)
    test_set = pd.merge(test, test_fe, on='segment_id')
    test_set.to_csv(Test_file_name, index=False)

