import numpy as np
import torch
from torch import nn

## クラス内で使う関数
def idx2mask(idx, max_size):
    mask = np.zeros(max_size)
    mask[idx] = 1.0
    return mask

## 次の行動を出力するニューラルネットワークの定義
class ActionPredictNetwork(nn.Module):
    def __init__(self, dim_state, actions):
        super().__init__()
        action_len = len(actions)
        hidden_size1 = dim_state * 10
        hidden_size3 = action_len * 10
        hidden_size2 = int(np.sqrt(hidden_size1 * hidden_size3))

        self.fc1 = nn.Linear(dim_state, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, action_len)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

## ニューラルネットワークを用いてQ関数を表現したクラス
class Qnetwork:
    def __init__(self, dim_state, actions, device='cpu', gamma=0.99, lr=1e-3, double_mode=True):
        self.dim_state = dim_state
        self.actions = actions
        self.action_len = len(actions)
        self.gamma = gamma
        self.double_mode = double_mode
        self.device = device

        self.main_network = ActionPredictNetwork(self.dim_state, self.actions).to(device)
        self.target_network = ActionPredictNetwork(self.dim_state, self.actions).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    ## 行動を出力する学習モデルに、Q関数の推定値を出力する学習モデルのパラメータを反映する
    def sync_target_network(self, soft):
        weights = self.main_network.state_dict()
        self.target_network.load_state_dict(weights)

    ## NNのパラメータを更新する
    def update_on_batch(self, exps):
        state, action, reward, next_state, terminated, truncated = zip(*exps)
        action_index = [
            self.actions.index(a) for a in action
        ]
        action_mask = np.array([
            idx2mask(a, self.action_len)
            for a in action_index
        ])
        state = np.array(state)
        reward = np.array(reward)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        done = np.array(terminated or truncated)

        with torch.no_grad():
            self.target_network.eval()
            next_target_q_values_batch = self.target_network(next_state)
            self.main_network.eval()
            next_q_values_batch = self.main_network(next_state)

        if self.double_mode:
            future_return = [
                next_target_q_values[np.argmax(next_q_values.cpu().numpy())].cpu().numpy()
                for next_target_q_values, next_q_values in zip(next_target_q_values_batch, next_q_values_batch)
            ]
        else:
            future_return = [
                torch.max(next_q_values.cpu().numpy()) for next_q_values
                in next_target_q_values_batch
            ]

        # Q値の推定
        x = torch.tensor(state, dtype=torch.float32).to(self.device)
        output = self.main_network(x)
        pred = (output * torch.tensor(action_mask, dtype=torch.float32).to(self.device)).sum(dim=1)

        # 目標Q値を計算
        y = reward + self.gamma * (1 - done) * future_return
        tensor_y = torch.tensor(y, dtype=torch.float32).to(self.device)

        # 学習
        self.main_network.train()
        loss = self.loss_fn(pred, tensor_y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        td_error = np.mean(np.abs(y - pred.detach().cpu().numpy()))

        return loss.item(), td_error

	## モデルを保存する
    def save_model(self, path):
        torch.save(self.main_network.state_dict(), path)

## 方策を表すクラス
class EpsilonGreedyPolicy:
    def __init__(self, q_network, epsilon):
        self.q_network = q_network
        self.epsilon = epsilon

    ## 発生させた乱数がε未満ならばランダムに行動を返却する
    ## ε以上ならばNNの出力（Q値）を最大にする行動を返却する
    def get_action(self, state, actions):
        if np.random.uniform() < self.epsilon:
            q_values = None
            action = np.random.choice(actions)
        else:
            state = np.reshape(state, (1, len(state)))
            with torch.no_grad():
                self.q_network.main_network.eval()
                q_values = self.q_network.main_network(torch.tensor(state).to(self.q_network.device))
            action = actions[np.argmax(q_values.detach().cpu().numpy())]
        return action, self.epsilon, q_values
