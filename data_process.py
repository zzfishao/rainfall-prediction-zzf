import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print(device)


def load_data(file):
    rawdata = pd.read_csv(f'data/groupByCZ/{file}.csv')
    # rawdata.columns = ['', 'RCD', 'TM', 'INTV', 'DRP', 'MODITIME', 'SOURCE']
    rawdata = rawdata.loc[:, ~rawdata.columns.str.contains('^Unnamed')]
    # rawdata.fillna(rawdata.mean(), inplace=True)
    # rawdata = rawdata.drop(columns='_c0')
    # rawdata.fillna(0, inplace=True)
    # rawdata._c3 = rawdata._c3.astype("float32")
    return rawdata


def strlist_to_date(s):
    def is_run(yyyy):
        if yyyy % 400 == 0:
            return 1
        if yyyy % 4 == 0 and yyyy % 100 != 0:
            return 1
        return 0

    res = []
    for i in s:
        yyyy = int(i[:4])
        mm = int(i[5:7])
        dd = int(i[8:10])
        month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
        day = month[mm - 1] + dd + (is_run(yyyy) if mm >= 2 else 0)
        res.append(day)
    return res


def rainfall_to_level(pp):
    res = []
    for i in pp:
        res.append(0 if i <= 0.1 else (
            1 if i <= 10 else (2 if i <= 25 else (3 if i <= 50 else (4 if i <= 100 else (5 if i <= 250 else 6))))))
    return res


def ema_corr(data, beta):
    v_ema = []
    v_ema_corr = []
    v_pre = 0
    for i, t in enumerate(data):
        v_t = beta * v_pre + (1 - beta) * t
        v_ema.append(v_t)
        v_pre = v_t

    for i, t in enumerate(v_ema):
        v_ema_corr.append(t / (1 - np.power(beta, i + 1)))
    return v_ema_corr


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def process_data(file, seq_len, data_Normalize):
    dataset = load_data(file)
    # print(dataset)
    ema_beta = 0.5

    files = ['2015_2020', '2013_2022DRPDRY', '330112MYL', 'JiChang_2005_1']

    if data_Normalize:
        # for i in dataset:
        #     i = ema_corr(i, ema_beta)
        if file == files[0]:
            # dataset.DRP = rainfall_to_level(dataset.DRP.values)
            dataset.DRP = ema_corr(dataset.DRP, ema_beta)
            dataset.DYE = ema_corr(dataset.DYE, ema_beta)
            dataset.AVINQ = ema_corr(dataset.AVINQ, ema_beta)
            dataset.AVOTQ = ema_corr(dataset.AVOTQ, ema_beta)
            # dataset.AVZ = ema_corr(dataset.AVZ, ema_beta)
            dataset.AVQ = ema_corr(dataset.AVQ, ema_beta)
        if file == files[1]:
            dataset.DRP = ema_corr(dataset.DRP, 0.5)
            dataset.DRY = ema_corr(dataset.DRY, 0.5)

    # plt.figure(figsize=(16, 9))

    # if file == files[0]:
    #     plt.plot(dataset['DRP'], label='DRP')
    #     plt.plot(dataset['DYE'], label='DYE')
    #     plt.plot(dataset['AVINQ'], label='AVINQ')
    #     plt.plot(dataset['AVOTQ'], label='AVOTQ')
    #     # plt.plot(dataset['AVZ'], label = 'AVZ')
    #     plt.plot(dataset['AVQ'], label='AVQ')
    #
    # if file == files[1]:
    #     plt.plot(dataset['DRP'], label='DRP')
    #     plt.plot(dataset['DRY'], label='DRY')
    # if file == files[2]:
    #     plt.plot(dataset['MYL6'], label='DRP')
    # plt.xticks(range(0, dataset.shape[0], 365), dataset['TM'].loc[::365], rotation=90)
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f'data/img/{file}.jpg')
    dataset = dataset.drop(columns='TM')

    # print(dataset)
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    x, y = sliding_windows(dataset, seq_len)

    print(x.shape, y.shape)
    train_size = int(len(y) * 0.8)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x))).to(device)
    dataY = Variable(torch.Tensor(np.array(y))).to(device)

    trainX = Variable(torch.Tensor(np.array(x[:train_size]))).to(device)
    trainY = Variable(torch.Tensor(np.array(y[:train_size]))).to(device)

    testX = Variable(torch.Tensor(np.array(x[train_size:]))).to(device)
    testY = Variable(torch.Tensor(np.array(y[train_size:]))).to(device)
    # print(trainX.shape)
    # print(trainY.shape)
    return dataX, dataY, trainX, trainY, testX, testY, scaler


def get_score(data_predict_, dataY_plot_):
    def MAPE(labels, predicts, mask):
        """
            Mean absolute percentage. Assumes ``y >= 0``.
            Defined as ``(y - y_pred).abs() / y.abs()``
        """
        loss = np.abs(predicts - labels) / (np.abs(labels) + 1)
        loss *= mask
        non_zero_len = mask.sum()
        return np.sum(loss) / non_zero_len

    mae = mean_absolute_error(dataY_plot_, data_predict_)
    mse = mean_squared_error(dataY_plot_, data_predict_)
    rmse = np.sqrt(mean_squared_error(dataY_plot_, data_predict_))
    # mape = (abs(dataY_plot_ - data_predict_) / dataY_plot_).mean()
    r_2 = r2_score(dataY_plot_, data_predict_)
    dataY_plot_mask = (1 - (abs(dataY_plot_ - 0) < 1))
    mape = MAPE(dataY_plot_, data_predict_, dataY_plot_mask)
    # for i in range(len(data_predict_)):  #求Σ过程
    #     mae = mae + abs(dataY_plot_[i] - data_predict_[i])  #MAE差的绝对值
    #     if dataY_plot_[i].any() == 0:  #MAPE需要分情况处理
    #         # continue
    #         if data_predict_[i] != 0:
    #             mape = mape + 1
    #     else:
    #         mape = mape + abs((dataY_plot_[i] - data_predict_[i]) / dataY_plot_[i])
    #         print(abs((dataY_plot_[i] - data_predict_[i]) / dataY_plot_[i]))
    # mape /= data_predict_.shape[0]
    # mape = mape.mean()
    # score = list([mae, rmse, mape, r_2])
    return mse, mae, r_2, mape, rmse


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 初始化
# ema = EMA(model, 0.999)
# ema.register()
#
#
# # 训练过程中，更新完参数后，同步update shadow weights
# def train():
#     optimizer.step()
#     ema.update()


# eval前，apply shadow weights；eval之后，恢复原来模型的参数
# def evaluate():
#     ema.apply_shadow()
#     # evaluate
#     ema.restore()


if __name__ == '__main__':
    file = '330112MYL'
    # file = 'JiChang_2005_2'
    dataset = load_data(file)
    print(dataset)
    # plt.figure(figsize=(20, 10))
    # # plt.plot(dataset['avgRain'], label='avgRain')
    # # plt.plot(dataset['avgRain'], label='avgRain')
    # plt.plot(dataset['MYL1'], label='MYL1')
    # plt.xticks(range(0, dataset.shape[0], 365), dataset['date'].loc[::365], rotation=90)
    # # plt.tight_layout()
    # plt.legend()
    # plt.show()
    # dataset.DRP = ema_corr(dataset.DRP, 0.5)
    # dataset.DRY = ema_corr(dataset.DRY, 0.1)
    # print(dataset)
    # plt.plot(dataset.AVZ)
    # plt.plot(dataset.DRY)
    # plt.show()

    corrmatrix = dataset.corr(method='pearson')
    print(corrmatrix.loc['MYL6'])
