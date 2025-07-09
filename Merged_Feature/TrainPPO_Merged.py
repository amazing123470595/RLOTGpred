import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score, average_precision_score
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import OneHotCategorical
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    StepCounter,
    TransformedEnv, )

from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
# from My_advantages import GAE # 自己的GAE
from torchrl.objectives.value import GAE   # 注释掉：不再用官方GAE
from tqdm import tqdm
from Env import PPOEnvZcc
from Features import GetMerged_Feature
import torch.nn.functional as F

device = "cpu"
lr = 3e-4
max_grad_norm = 1.0
frames_per_batch = 10000
total_frames = 72511 * 5
sub_batch_size = 1000
num_epochs = 10
clip_epsilon = (
    0.2
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# 数据信息
model_name = "Merged_Feature"

# 全部数据集
test_all = "../Dataset/test/features/merged_feature_independent.csv"
train_all = "../Dataset/train/features/merged_feature_training.csv"

X_train, y_train, X_test, y_test, ratio = GetMerged_Feature(train_all, test_all)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(int)

env = PPOEnvZcc(X_train, y_train, ratio, total_frames, model_name)
env = TransformedEnv(
    env,
    Compose(
        DoubleToFloat(
            in_keys=["observation"],
        ),
        StepCounter(),
    ),
)

actor_net = nn.Sequential(
    nn.LazyLinear(X_train.shape[1], device=device),
    nn.ReLU(),
    nn.LazyLinear(2, device=device),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["logits"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)

value_net = nn.Sequential(
    nn.LazyLinear(X_train.shape[1], device=device),
    nn.ReLU(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

policy_module(env.reset())
value_module(env.reset())

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

for i, tensordict_data in enumerate(collector):
    for _ in range(num_epochs):
        with torch.no_grad():
            advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        f" MCC={env.GetTest()[0]} Acc={env.GetTest()[1]} Sn={env.GetTest()[2]} Sp={env.GetTest()[3]}"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

    pbar.set_description(", ".join([cum_reward_str, lr_str]))
    scheduler.step()

env.save_all_steps_inf()
# 设置模型为评估模式
policy_module.eval()
# 存储预测结果
predicted_actions = []
# 概率值
Probabilities = []

with torch.no_grad():
    for X in X_test:
        td = TensorDict(
            {},
            [],
        )
        td["observation"] = X
        # 使用加载的模型进行预测
        prediction = policy_module(td)["action"]
        prediction = np.argmax(prediction)
        predicted_actions.append(prediction)

        # 获取logits
        logits = policy_module(td)["logits"]
        # 概率分布
        prob = F.softmax(logits, dim=-1)
        Probabilities.append(prob[1])

TN, FP, FN, TP = confusion_matrix(y_test, predicted_actions, labels=[0, 1]).ravel()
numerator = (TP * TN) - (FP * FN)
denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
MCC = numerator / denominator if denominator != 0 else 0
Acc = (TP + TN) / (TN + FP + FN + TP)
Sn = TP / (TP + FN) if TP + FN else 0
Sp = TN / (TN + FP) if TN + FP else 0
G_mean = np.sqrt(Sn * Sp)
F1 = f1_score(y_test, predicted_actions, zero_division=0)
Recall = recall_score(y_test, predicted_actions)
Precision = TP / (TP + FP) if TP + FP else 0
AUC = roc_auc_score(y_test, Probabilities)
AP = average_precision_score(y_test, Probabilities)
BalancedAccuracy = (recall_score(y_test, predicted_actions) +
                    (confusion_matrix(y_test, predicted_actions).ravel()[0] /
                     (confusion_matrix(y_test, predicted_actions).ravel()[0] +
                      confusion_matrix(y_test, predicted_actions).ravel()[1]))) / 2

# 保存模型参数
torch.save(policy_module.state_dict(), f'{model_name}_ACC{Acc:.4f}_MCC{MCC:.4f}_SN{Sn:.4f}_SP{Sp:.4f}.pth')

# 计算性能指标
metrics_dict = {
    'Balanced Accuracy': BalancedAccuracy,
    'MCC': MCC,
    'Sensitivity': Sn,
    'Specificity': Sp,
    'F1 Score': F1,
}

# 输出和保存性能指标
df_metrics = pd.DataFrame([metrics_dict])
df_metrics.to_csv('Data/performance_metrics.csv', index=False)

# 保存预测概率和标签
df_predictions = pd.DataFrame({
    'Probabilities': Probabilities,
    'Labels': y_test
})
df_predictions.to_csv('Data/probabilities.csv', index=False)