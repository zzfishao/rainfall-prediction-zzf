import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scienceplots
import matplotlib.pyplot as plt

from data_process import process_data, get_score, EMA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 读取模型

files = ['2015_2020', '2013_2022DRPDRY', '330112MYL']
models = ['LSTM', 'GRU', 'CNN_LSTM', 'CNN_LSTM_PRO', 'CNN_GRU', 'LSTM_Attention', 'LSTMAttention']


def load_model(file=0, model=0, seq=50, Normalize=True, bidirection=False):
    load_model = models[model]
    load_file = files[file]
    load_seq = seq
    load_Normalize = Normalize
    load_bidirection = bidirection
    dataX, dataY, trainX, trainY, testX, testY, scaler = process_data(load_file, load_seq, load_Normalize)
    load_Path = f'data/model/model{load_model}{"_bi" if load_bidirection else ""}_file{load_file}_seq{load_seq}_Normalize{load_Normalize}.pt'

    print(f'load model: {load_Path}')
    model = torch.load(load_Path, map_location=torch.device('cpu'))
    model.eval()
    train_predict = model(dataX)
    data_predict = train_predict.cpu().data.numpy()
    dataY_plot = dataY.cpu().data.numpy()
    data_predict = scaler.inverse_transform(data_predict)
    # data_predict[:,query_index] = np.array(normalization(data_predict[:,query_index]))
    dataY_plot = scaler.inverse_transform(dataY_plot)
    # data_predict[:,query_index] = np.array(rainfall_to_level(data_predict[:,query_index]))
    # dataY_plot[:,query_index] = np.array(rainfall_to_level(dataY_plot[:,query_index]))
    return dataY_plot, data_predict, seq


def rainfall_to_level(pp):
    res = []
    for i in pp:
        res.append(0 if i <= 2 else (
            1 if i <= 10 else (2 if i <= 25 else (3 if i <= 50 else (4 if i <= 100 else (5 if i <= 250 else 6))))))
    return res


def normalization(data_predict):
    res = []
    for i in data_predict:
        x = []
        # x.append(1 if i <= 1.5 else (2 if i <= 2.125 else (
        #     3 if i <= 3 else (4 if i <= 4 else (5 if i <= 5 else (6 if i <= 6 else 7))))))
        # x.append(i + 20)
        res.append(((i / 3.5) ** 3))
    return res



if __name__ == '__main__':
    la = [["Inflow", "Outflow", "Spatial Rainfall", "Spatial Evaporation", "outflow and inflow"],
          ["Spatial Rainfall", "Spatial Evaporation"]]
    file = 1
    query_index = 0
    ss1, ss2 = 20, 27
    with plt.style.context(['science', 'ieee']):
        plt.figure(figsize=(16, 10))

        Y1, P1, seq1 = load_model(file, 0, 60, True, False)
        print(get_score(P1[:, query_index], Y1[:, query_index]))
        plt.subplot(2, 2, 1)
        s1 = (Y1.shape[0] + seq1) // 100 * ss1
        s2 = (Y1.shape[0] + seq1) // 100 * ss2
        plt.plot(Y1[365 - seq1 + s1:365 - seq1 + s2, query_index], label=f'{la[file][query_index]} actual')
        # plt.plot(dataY_plot[trainX.shape[0] - dataX.shape[0]:, query_index], label='DYE actual')
        plt.plot(P1[365 - seq1 + s1:365 - seq1 + s2, query_index], label=f'{la[file][query_index]} predict')
        plt.xlabel('days')
        plt.ylabel(f'{la[file][query_index]}')
        plt.title('Our Algorithm')
        plt.legend()

        Y1, P1, seq1 = load_model(file, 2, 90, True, True)
        print(get_score(P1[:, query_index], Y1[:, query_index]))
        plt.subplot(2, 2, 2)
        s1 = (Y1.shape[0] + seq1) // 100 * ss1
        s2 = (Y1.shape[0] + seq1) // 100 * ss2
        plt.plot(Y1[365 - seq1 + s1:365 - seq1 + s2, query_index], label=f'{la[file][query_index]} actual')
        # plt.plot(dataY_plot[trainX.shape[0] - dataX.shape[0]:, query_index], label='DYE actual')
        plt.plot(P1[365 - seq1 + s1:365 - seq1 + s2, query_index], label=f'{la[file][query_index]} predict')
        plt.xlabel('days')
        plt.ylabel(f'{la[file][query_index]}')
        plt.title('LSTM')
        plt.legend()

        Y1, P1, seq1 = load_model(file, 1, 60, True, False)
        print(get_score(P1[:, query_index], Y1[:, query_index]))
        plt.subplot(2, 2, 3)
        s1 = (Y1.shape[0] + seq1) // 100 * ss1
        s2 = (Y1.shape[0] + seq1) // 100 * ss2
        plt.plot(Y1[365 - seq1 + s1:365 - seq1 + s2, query_index], label=f'{la[file][query_index]} actual')
        # plt.plot(dataY_plot[trainX.shape[0] - dataX.shape[0]:, query_index], label='DYE actual')
        plt.plot(P1[365 - seq1 + s1:365 - seq1 + s2, query_index], label=f'{la[file][query_index]} predict')
        plt.xlabel('days')
        plt.ylabel(f'{la[file][query_index]}')
        plt.title('GRU')
        plt.legend()

        Y1, P1, seq1 = load_model(file, 0, 365, True, False)
        print(get_score(P1[:, query_index], Y1[:, query_index]))
        plt.subplot(2, 2, 4)
        s1 = (Y1.shape[0] + seq1) // 100 * ss1
        s2 = (Y1.shape[0] + seq1) // 100 * ss2
        plt.plot(Y1[365 - seq1 + s1:365 - seq1 + s2, query_index], label=f'{la[file][query_index]} actual')
        # plt.plot(dataY_plot[trainX.shape[0] - dataX.shape[0]:, query_index], label='DYE actual')
        plt.plot(P1[365 - seq1 + s1:365 - seq1 + s2, query_index], label=f'{la[file][query_index]} predict')
        plt.xlabel('days')
        plt.ylabel(f'{la[file][query_index]}')
        plt.title('Transformer')
        plt.legend()
        # print(data_predict.shape[0] + load_seq)
        # plt.plot(data_predict[trainX.shape[0] - dataX.shape[0]:, query_index], label='DYE predict')
        # plt.axvline(x=trainX.shape[0], c='r', linestyle='--')
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.savefig(f'data/img/{files[file]}_{la[file][query_index]}_predict.svg')
        plt.show()
