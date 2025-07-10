import csv
from typing import Optional
import numpy as np
import torch
from numpy import inf
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from torchrl.envs import (
    EnvBase, )


class PPOEnvZcc(EnvBase):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, ratio, total_frames, model_name: str, seed=None,
                 device="cpu"):
        super().__init__(device=device, batch_size=[])

        self.X_train = X_train
        self.y_train = y_train
        self.ratio = ratio
        self.TN = 0
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.step = 0
        self.total_frames = total_frames
        self.id = np.arange(self.X_train.shape[0])

        # 模型名字用于保存每一步信息
        self.model_name = model_name
        self.step_inf = []

        self._make_spec()
        if seed is None:
            # seed = torch.empty((), dtype=torch.int64).random_().item()
            seed = 42 # 先固定种子
            print("seed: ", seed)
        self.set_seed(seed)

    def _step(self, tensordict):
        self.step += 1

        done = tensordict["done"]
        if done:
            return self.reset()

        env_action = int(self.y_train[self.id[0]])
        self.id = np.delete(self.id, 0)
        action = tensordict["action"].squeeze(-1)
        action = np.argmax(action).item()
        logits = tensordict["logits"].squeeze(-1).detach().numpy()

        # 平衡会奖励机制
        # 参数自调
        RewardTP = 1.0 * self.ratio
        RewardTN = 1.0
        RewardFN = -1.0 * self.ratio
        RewardFP = -1.0 

        if action == env_action:
            if env_action:
                self.TP += 1
                reward = RewardTP
            else:
                self.TN += 1
                reward = RewardTN
        else:
            done = True
            if env_action:
                self.FN += 1
                reward = RewardFN
            else:
                self.FP += 1
                reward = RewardFP

        if len(self.id) == 0:
            self.id = np.arange(self.X_train.shape[0])

        new_observation = self.X_train[self.id[0]]
        out = TensorDict(
            {
                "observation": new_observation,
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        # 记录每一步情况
        self.step_inf.append((self.step, logits, action, env_action, reward))
        return out

    def _reset(self, tensordict):
        done = False
        np.random.shuffle(self.id)

        observation = self.X_train[self.id[0]]
        out = TensorDict(
            {
                "observation": observation,
                "done": done,
            },
            batch_size=[],
        )
        return out

    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                low=-inf,
                high=inf,
                shape=self.X_train.shape[1:],
                dtype=torch.float64,
            ),
            shape=()
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = DiscreteTensorSpec(2)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def GetTest(self):
        numerator = (self.TP * self.TN) - (self.FP * self.FN)
        denominator = ((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN)) ** 0.5
        MCC = numerator / denominator if denominator != 0 else 0
        Acc = (self.TP + self.TN) / (self.TN + self.FP + self.FN + self.TP)
        Sn = self.TP / (self.TP + self.FN) if self.TP + self.FN else 0
        Sp = self.TN / (self.TN + self.FP) if self.TN + self.FP else 0
        return "{:.4f}".format(MCC), "{:.4f}".format(Acc), "{:.4f}".format(Sn), "{:.4f}".format(Sp)

    # 将步数信息保存到文件
    def save_all_steps_inf(self):
        with open(f"Data/{self.model_name}_Steps.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Logits_1", "Logits_2", "Action", "EnvAction", "Reward"])
            for step_info in self.step_inf:
                writer.writerow(
                    [step_info[0], step_info[1][0], step_info[1][1], step_info[2], step_info[3], step_info[4]])
