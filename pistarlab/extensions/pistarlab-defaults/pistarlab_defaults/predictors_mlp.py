
from typing import Tuple, Any


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from pistarlab import ctx
# because stocastic


def policy_loss_fn(action_selected_prob, action_weights):
    return torch.mean(-torch.log(action_selected_prob) * action_weights)


def multi_choice(v):
    def rchoice(x):
        return np.random.choice(len(x), p=x)
    return np.apply_along_axis(rchoice, 1, v)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class AC2ModelMLP(nn.Module):

    def __init__(self, input_size, action_num, hidden_nodes=256):
        super(AC2ModelMLP, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.action_output_layer = nn.Linear(hidden_nodes, action_num)
        self.value_output_layer = nn.Linear(hidden_nodes, 1)

    def forward(self, x) -> Tuple[Any, Any]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_output_layer(x))
        val_est = self.value_output_layer(x)
        return action_probs, val_est


class PolicyModel:
    """
    Policy Function approximator.
    """

    def __init__(
            self,
            input_size,
            action_num,
            learning_rate=0.0001,
            hidden_nodes=256
    ):
        self.net = AC2ModelMLP(
            input_size=input_size,
            action_num=action_num,
            hidden_nodes=hidden_nodes)

        self.policy_optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, eps=1e-3)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.update_counter = 0


class ValueModel:
    """
    Policy Function approximator.
    """

    def __init__(
            self,
            input_size,
            action_num,
            learning_rate=0.0001,
            hidden_nodes=256):
        self.net = AC2ModelMLP(
            input_size=input_size,
            action_num=action_num,
            hidden_nodes=hidden_nodes)

        self.value_loss_fn = torch.nn.MSELoss(reduction='sum')
        self.value_optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, eps=1e-3)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.update_counter = 0


class PolicyEstimatorMLP:
    """
    Policy Function approximator.
    """

    def __init__(self, model: AC2ModelMLP):
        self.model = model
        self.logger = ctx.get_logger()

    def predict_choice(self, ob):
        ob = ob.astype("float32")
        action_probs, _ = self.model.net(torch.from_numpy(ob).to(self.model.device))
        action_probs = action_probs.detach().cpu().numpy()

        # sample, then 2d to 1d
        choice = np.squeeze(multi_choice(action_probs))
        #Batching disabled (returns one choice)
        return choice, action_probs

    def update(self, ob, advantage, action):
        ob = ob.astype("float32")
        ob = torch.from_numpy(ob).to(self.model.device)

        advantage = np.expand_dims(advantage, axis=0)
        advantage = torch.from_numpy(advantage).to(self.model.device)

        # Perform prediction again? why not used prior predicted probs?
        action_probs, _ = self.model.net(ob)

        # from ..utils.remote_pdb import set_trace
        # set_trace() # you'll see the port number in the logs

        selected_action_probs = np.take_along_axis(
            action_probs,
            action.reshape(action.shape[1], 1, 1), 
            axis=2).squeeze(axis=2)
        # print(selected_action_probs)
        loss = -torch.mean(torch.log(selected_action_probs) * advantage)

        # loss = torch.mean(torch.log(selected_action_probs) * advantage)
        self.model.policy_optimizer.zero_grad()
        loss.backward()

        self.model.policy_optimizer.step()

        self.model.update_counter += 1
        return loss.item()


class ValueEstimatorMLP:
    """
    Value Function approximator.
    """

    def __init__(self, model: AC2ModelMLP):
        self.logger = ctx.get_logger()
        self.model = model

    def predict(self, ob):
        ob = ob.astype("float32")
        _, value_estimate = self.model.net(torch.from_numpy(ob).to(self.model.device))
        val = value_estimate.detach().cpu().numpy()
        return np.squeeze(val)

    def update(self, ob, target):
        ob = ob.astype("float32")
        target = np.float32(target)

        #convert scalar to shape = (1,1)
        # target = np.expand_dims(target, axis=0)
        # assert(target.ndim == 2)
        target = torch.from_numpy(target).to(self.model.device)

        ob = torch.from_numpy(ob).to(self.model.device)

        # forward pass , TODO: use value from prior forward pass
        # self.logger.info(f"ob.shape: {ob.shape}")
        _, value_estimate = self.model.net(ob)

        # assert(value_estimate.ndim == 2)

        loss = self.model.value_loss_fn(value_estimate, target)
        self.model.value_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.net.parameters(),0.01)
        self.model.value_optimizer.step()

        self.model.update_counter += 1

        return loss.item()
