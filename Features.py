import numpy as np
from Feature.CKSAAP import GetCKSAAP_4
from Feature.CTDC import GetCTDC_4
from Feature.PAAC import GetPAAC_4
import csv
from Feature.TPEMPPS import GetTPEMPPS


def GetCCP_4(train_negative, train_positive, test_negative, test_positive):
    X_train1, y_train, X_test1, y_test, r = GetCKSAAP_4(train_negative, train_positive, test_negative, test_positive)
    X_train2, y_train, X_test2, y_test = GetCTDC_4(train_negative, train_positive, test_negative, test_positive)
    X_train3, y_train, X_test3, y_test = GetPAAC_4(train_negative, train_positive, test_negative, test_positive)

    X_train = np.concatenate((X_train1, X_train2, X_train3), axis=1)
    X_test = np.concatenate((X_test1, X_test2, X_test3), axis=1)

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test, r


def GetProtT5_T_4(train_negative, train_positive, test_negative, test_positive):
    with open(train_negative, mode='r') as file1:
        csv_reader1 = csv.reader(file1)
        train_negative_ProtT5 = np.array([row[1:] for row in csv_reader1])

    with open(train_positive, mode='r') as file2:
        csv_reader2 = csv.reader(file2)
        train_positive_ProtT5 = np.array([row[1:] for row in csv_reader2])

    with open(test_negative, mode='r') as file3:
        csv_reader3 = csv.reader(file3)
        test_negative_ProtT5 = np.array([row[1:] for row in csv_reader3])

    with open(test_positive, mode='r') as file4:
        csv_reader4 = csv.reader(file4)
        test_positive_ProtT5 = np.array([row[1:] for row in csv_reader4])

    X_train = np.concatenate((train_positive_ProtT5, train_negative_ProtT5), axis=0)
    X_test = np.concatenate((test_positive_ProtT5, test_negative_ProtT5), axis=0)

    N_train_Pos = train_positive_ProtT5.shape[0]
    N_train_Neg = train_negative_ProtT5.shape[0]
    N_test_Pos = test_positive_ProtT5.shape[0]
    N_test_Neg = test_negative_ProtT5.shape[0]
    num_P = N_train_Pos + N_test_Pos
    num_N = N_train_Neg + N_test_Neg

    ratio = num_P / num_N if num_P > num_N else num_N / num_P
    # 提取特征和标签数据的值
    y_train_Negative = np.zeros(train_negative_ProtT5.shape[0])
    y_train_Positive = np.ones(train_positive_ProtT5.shape[0])
    y_train = np.concatenate((y_train_Positive, y_train_Negative), axis=0)

    y_test_Negative = np.zeros(test_negative_ProtT5.shape[0])
    y_test_Positive = np.ones(test_positive_ProtT5.shape[0])
    y_test = np.concatenate((y_test_Positive, y_test_Negative), axis=0)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test, round(ratio, 2)


def GetTPEMPPS_CCP(train_negative, train_positive, test_negative, test_positive):
    X_train1, y_train, X_test1, y_test, ratio = GetCCP_4(train_negative, train_positive, test_negative,
                                                         test_positive)
    X_train2, y_train, X_test2, y_test, ratio = GetTPEMPPS(train_negative, train_positive, test_negative,
                                                           test_positive)

    X_train = np.concatenate((X_train2, X_train1), axis=1)
    X_test = np.concatenate((X_test2, X_test1), axis=1)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test, ratio

def GetMerged_Feature(train_all, test_all):
    N_train_Pos = 0
    N_train_Neg = 0
    N_test_Pos = 0
    N_test_Neg = 0
    with open(train_all, mode='r') as file1:
        csv_reader1 = csv.reader(file1)
        next(csv_reader1)  # 跳过第一行（列名）
        train_Merged = np.array([row[3:] for row in csv_reader1])

    with open(test_all, mode='r') as file2:
        csv_reader2 = csv.reader(file2)
        next(csv_reader2)  # 跳过第一行（列名）
        test_Merged = np.array([row[3:] for row in csv_reader2])

    with open(train_all, mode='r') as file1:
        csv_reader1 = csv.reader(file1)
        next(csv_reader1)  # 跳过第一行（列名）
        for row in csv_reader1:
            label = row[1]  # 假设第二列是标签列
            if label.startswith('Pos'):
                N_train_Pos += 1
            elif label.startswith('Neg'):
                N_train_Neg += 1
    
    with open(test_all, mode='r') as file2:
        csv_reader2 = csv.reader(file2)
        next(csv_reader2)  # 跳过第一行（列名）
        for row in csv_reader2:
            label = row[1]  # 假设第二列是标签列
            if label.startswith('Pos'):
                N_test_Pos += 1
            elif label.startswith('Neg'):
                N_test_Neg += 1


    X_train = train_Merged
    X_test = test_Merged

    num_P = N_train_Pos + N_test_Pos
    num_N = N_train_Neg + N_test_Neg
    # print(num_P)
    # print(num_N)
    ratio = num_P / num_N if num_P > num_N else num_N / num_P
    y_train_Negative = np.zeros(N_train_Neg)
    y_train_Positive = np.ones(N_train_Pos)
    y_train = np.concatenate((y_train_Positive, y_train_Negative), axis=0)

    y_test_Negative = np.zeros(N_test_Neg)
    y_test_Positive = np.ones(N_test_Pos)
    y_test = np.concatenate((y_test_Positive, y_test_Negative), axis=0)

    return X_train, y_train, X_test, y_test, round(ratio, 2)

def GetESM2(train_all, test_all):
    N_train_Pos = 0
    N_train_Neg = 0
    N_test_Pos = 0
    N_test_Neg = 0
    start_col = 557 - 1
    end_col = 1836
    with open(train_all, mode='r') as file1:
        csv_reader1 = csv.reader(file1)
        header = next(csv_reader1)  # 第一行：列名
        # 提取要的列名
        first_col_name = header[start_col]
        last_col_name = header[end_col - 1]
        print(f"提取到的第一列列名: {first_col_name}")
        print(f"提取到的最后一列列名: {last_col_name}")
        # 提取数据部分：第 557 列到 1836 列
        train_Merged = np.array([row[start_col:end_col] for row in csv_reader1])
        
    with open(test_all, mode='r') as file2:
        csv_reader2 = csv.reader(file2)
        next(csv_reader2)  # 跳过第一行（列名）
        test_Merged = np.array([row[start_col:end_col] for row in csv_reader2])

    with open(train_all, mode='r') as file1:
        csv_reader1 = csv.reader(file1)
        next(csv_reader1)  # 跳过第一行（列名）
        for row in csv_reader1:
            label = row[1]  # 假设第二列是标签列
            if label.startswith('Pos'):
                N_train_Pos += 1
            elif label.startswith('Neg'):
                N_train_Neg += 1
    
    with open(test_all, mode='r') as file2:
        csv_reader2 = csv.reader(file2)
        next(csv_reader2)  # 跳过第一行（列名）
        for row in csv_reader2:
            label = row[1]  # 假设第二列是标签列
            if label.startswith('Pos'):
                N_test_Pos += 1
            elif label.startswith('Neg'):
                N_test_Neg += 1


    X_train = train_Merged
    X_test = test_Merged

    num_P = N_train_Pos + N_test_Pos
    num_N = N_train_Neg + N_test_Neg
    # print(num_P)
    # print(num_N)
    ratio = num_P / num_N if num_P > num_N else num_N / num_P
    y_train_Negative = np.zeros(N_train_Neg)
    y_train_Positive = np.ones(N_train_Pos)
    y_train = np.concatenate((y_train_Positive, y_train_Negative), axis=0)

    y_test_Negative = np.zeros(N_test_Neg)
    y_test_Positive = np.ones(N_test_Pos)
    y_test = np.concatenate((y_test_Positive, y_test_Negative), axis=0)

    return X_train, y_train, X_test, y_test, round(ratio, 2)


def GetOthers(train_all, test_all):
    N_train_Pos = 0
    N_train_Neg = 0
    N_test_Pos = 0
    N_test_Neg = 0
    start_col = 3
    end_col = 557 - 1
    with open(train_all, mode='r') as file1:
        csv_reader1 = csv.reader(file1)
        header = next(csv_reader1)  # 第一行：列名
        # 提取要的列名
        first_col_name = header[start_col]
        last_col_name = header[end_col - 1]
        print(f"提取到的第一列列名: {first_col_name}")
        print(f"提取到的最后一列列名: {last_col_name}")
        # 提取数据部分：第 3 列到 556 列
        train_Merged = np.array([row[start_col:end_col] for row in csv_reader1])
        
    with open(test_all, mode='r') as file2:
        csv_reader2 = csv.reader(file2)
        next(csv_reader2)  # 跳过第一行（列名）
        test_Merged = np.array([row[start_col:end_col] for row in csv_reader2])

    with open(train_all, mode='r') as file1:
        csv_reader1 = csv.reader(file1)
        next(csv_reader1)  # 跳过第一行（列名）
        for row in csv_reader1:
            label = row[1]  # 假设第二列是标签列
            if label.startswith('Pos'):
                N_train_Pos += 1
            elif label.startswith('Neg'):
                N_train_Neg += 1
    
    with open(test_all, mode='r') as file2:
        csv_reader2 = csv.reader(file2)
        next(csv_reader2)  # 跳过第一行（列名）
        for row in csv_reader2:
            label = row[1]  # 假设第二列是标签列
            if label.startswith('Pos'):
                N_test_Pos += 1
            elif label.startswith('Neg'):
                N_test_Neg += 1


    X_train = train_Merged
    X_test = test_Merged

    num_P = N_train_Pos + N_test_Pos
    num_N = N_train_Neg + N_test_Neg
    # print(num_P)
    # print(num_N)
    ratio = num_P / num_N if num_P > num_N else num_N / num_P
    y_train_Negative = np.zeros(N_train_Neg)
    y_train_Positive = np.ones(N_train_Pos)
    y_train = np.concatenate((y_train_Positive, y_train_Negative), axis=0)

    y_test_Negative = np.zeros(N_test_Neg)
    y_test_Positive = np.ones(N_test_Pos)
    y_test = np.concatenate((y_test_Positive, y_test_Negative), axis=0)

    return X_train, y_train, X_test, y_test, round(ratio, 2)