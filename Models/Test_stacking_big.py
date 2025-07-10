import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, recall_score, average_precision_score
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import OneHotCategorical
from torchrl.data import DiscreteTensorSpec
from torchrl.modules import ProbabilisticActor
import torch.nn.functional as F
from Features import GetProtT5_T_4, GetTPEMPPS_CCP, GetESM2, GetOthers
import pandas as pd

device = "cpu"

test_all = "../Dataset/test/features/merged_feature_independent.csv"
train_all = "../Dataset/train/features/merged_feature_training.csv"
test_negative1 = "../Dataset/test/features/neg_Independent_CD_ProtT5_features_T.csv"
test_positive1 = "../Dataset/test/features/pos_Independent_CD_ProtT5_features_T.csv"
train_negative1 = "../Dataset/train/features/neg_Training_CD_ProtT5_features_T.csv"
train_positive1 = "../Dataset/train/features/pos_Training_CD_ProtT5_features_T.csv"
test_negative2 = "../Dataset/test/fasta/neg_Independent.fasta"
test_positive2 = "../Dataset/test/fasta/pos_Independent.fasta"
train_negative2 = "../Dataset/train/fasta/neg_Training.fasta"
train_positive2 = "../Dataset/train/fasta/pos_Training.fasta"


# -----------------------------------------------------加载特征和模型-----------------------------------------------------

X1_train, y_train, X1_test, y_test, _ = GetProtT5_T_4(train_negative1, train_positive1, test_negative1, test_positive1)
X2_train, y_train, X2_test, y_test, _ = GetTPEMPPS_CCP(train_negative2, train_positive2, test_negative2, test_positive2)
X3_train, y_train, X3_test, y_test, _ = GetESM2(train_all, test_all)
X4_train, y_train, X4_test, y_test, _ = GetOthers(train_all, test_all)

X1_train = X1_train.astype(np.float32)
X2_train = X2_train.astype(np.float32)
X3_train = X3_train.astype(np.float32)
X4_train = X4_train.astype(np.float32)

X1_test = X1_test.astype(np.float32)
X2_test = X2_test.astype(np.float32)
X3_test = X3_test.astype(np.float32)
X4_test = X4_test.astype(np.float32)

# -----------------------------------------------------加载预训练模型-----------------------------------------------------
# ------------------模型1------------------
actor_net1 = nn.Sequential(
    nn.LazyLinear(X1_test.shape[1], device=device),
    nn.ReLU(),
    nn.LazyLinear(2, device=device),
)
policy_module1 = ProbabilisticActor(
    module=TensorDictModule(actor_net1, in_keys=["observation"], out_keys=["logits"]),
    spec=DiscreteTensorSpec(2),
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)
policy_module1.load_state_dict(torch.load('平衡_ProtT5_ACC0.8775_MCC0.7551_SN0.8850_SP0.8700.pth', map_location=device))
policy_module1.eval()

# ------------------模型2------------------
actor_net2 = nn.Sequential(
    nn.LazyLinear(X2_test.shape[1], device=device),
    nn.ReLU(),
    nn.LazyLinear(2, device=device),
)
policy_module2 = ProbabilisticActor(
    module=TensorDictModule(actor_net2, in_keys=["observation"], out_keys=["logits"]),
    spec=DiscreteTensorSpec(2),
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)
policy_module2.load_state_dict(torch.load('平衡_TPEMPPS_CCP_ACC0.8575_MCC0.7151_SN0.8650_SP0.8500.pth', map_location=device))
policy_module2.eval()

# ------------------模型3------------------
actor_net3 = nn.Sequential(
    nn.LazyLinear(X3_test.shape[1], device=device),
    nn.ReLU(),
    nn.LazyLinear(2, device=device),
)
policy_module3 = ProbabilisticActor(
    module=TensorDictModule(actor_net3, in_keys=["observation"], out_keys=["logits"]),
    spec=DiscreteTensorSpec(2),
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)
policy_module3.load_state_dict(torch.load('平衡_ESM2_ACC0.8800_MCC0.7600_SN0.8800_SP0.8800.pth', map_location=device))
policy_module3.eval()

# ------------------模型4------------------
actor_net4 = nn.Sequential(
    nn.LazyLinear(X4_test.shape[1], device=device),
    nn.ReLU(),
    nn.LazyLinear(2, device=device),
)
policy_module4 = ProbabilisticActor(
    module=TensorDictModule(actor_net4, in_keys=["observation"], out_keys=["logits"]),
    spec=DiscreteTensorSpec(2),
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)
policy_module4.load_state_dict(torch.load('平衡_Others_ACC0.8800_MCC0.7600_SN0.8850_SP0.8750.pth', map_location=device))
policy_module4.eval()

# ------------------融合模型------------------
meta_model = nn.Sequential(
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
)
meta_model.load_state_dict(torch.load("4_big_stacking_model_weights_Acc:0.8950 MCC:0.7906 Sn:0.9150 Sp:0.8750.pth", map_location=device))
meta_model.eval()

# -----------------------------------------------------生成 stacking 特征并进行预测-----------------------------------------------------
probs_list = []  # shape=(N,4)

with torch.no_grad():
    for i in range(X1_test.shape[0]):
        td1 = TensorDict({"observation": X1_test[i]}, [])
        logits1 = policy_module1(td1)["logits"]
        prob1 = F.softmax(logits1, dim=-1)[1].item()

        td2 = TensorDict({"observation": X2_test[i]}, [])
        logits2 = policy_module2(td2)["logits"]
        prob2 = F.softmax(logits2, dim=-1)[1].item()

        td3 = TensorDict({"observation": X3_test[i]}, [])
        logits3 = policy_module3(td3)["logits"]
        prob3 = F.softmax(logits3, dim=-1)[1].item()

        td4 = TensorDict({"observation": X4_test[i]}, [])
        logits4 = policy_module4(td4)["logits"]
        prob4 = F.softmax(logits4, dim=-1)[1].item()

        probs_list.append([prob1, prob2, prob3, prob4])

X_meta_test = np.array(probs_list).astype(np.float32)

# 使用融合模型进行预测
with torch.no_grad():
    logits_test = meta_model(torch.tensor(X_meta_test))
    probs_test = F.softmax(logits_test, dim=-1)
    predictions = probs_test.argmax(dim=-1).numpy()
    prob_positive = probs_test[:,1].numpy()

# -----------------------------------------------------计算指标-----------------------------------------------------
TN, FP, FN, TP = confusion_matrix(y_test, predictions, labels=[0,1]).ravel()

# 找到不一致的索引
# wrong_indices = np.where(y_test != predictions)[0]
# print("预测错误的样本序号:", wrong_indices)

MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+1e-8)
Acc = (TP+TN)/(TP+TN+FP+FN)
Sn = TP / (TP+FN) if TP+FN else 0
Sp = TN / (TN+FP) if TN+FP else 0
F1 = f1_score(y_test, predictions, zero_division=0)
AUC = roc_auc_score(y_test, prob_positive)

print(f"Acc:{Acc:.4f} MCC:{MCC:.4f} Sn:{Sn:.4f} Sp:{Sp:.4f} F1:{F1:.4f}")

# -----------------------------------------------------保存结果-----------------------------------------------------
df_metrics = pd.DataFrame([{
    'Acc': Acc, 'MCC': MCC, 'Sn': Sn, 'Sp': Sp, 'F1': F1
}])
df_metrics.to_csv('Data/stacking_metrics.csv', index=False)

df_predictions = pd.DataFrame({
    'Probabilities': prob_positive,
    'Labels': y_test
})
df_predictions.to_csv('Data/stacking_probabilities.csv', index=False)