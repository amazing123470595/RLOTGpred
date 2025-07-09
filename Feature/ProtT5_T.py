import csv
import numpy as np

file = 'Training_CD_ProtT5_features'
in_N = 'MyData/neg_' + file + '.csv'
out_N = 'MyData/neg_' + file + '_T.csv'

in_P = 'MyData/pos_' + file + '.csv'
out_P = 'MyData/pos_' + file + '_T.csv'


def wedi(in_Path, out_Path):
    # 打开一个新文件以写入数据
    with open(out_Path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)

        # 读取 Positive_ProtT5 数据集
        with open(in_Path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row2 in csv_reader:
                title = row2[0]
                features = np.array(row2[1:], dtype=float)
                features = features.reshape(-1, 1024)
                features = features[20]
                # 写入标题和池化特征到新文件
                csv_writer.writerow([title] + features.tolist())


wedi(in_P, out_P)
wedi(in_N, out_N)
