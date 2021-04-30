import logging
import random
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .predictors import PolicyEstimator, ValueEstimator
from .transformers import (
    IntToOneHotTransformer, MirrorTransformer, NpArrayFlattenTransformer,
    Transformer, build_action_transformer, build_observation_transformer)

from modelinspector.inspector import Inspector
import os
from ..storage import Storage
from ..core import create_sys_context
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

class ReinforceMLP(nn.Module):

    def __init__(self, input_size, hidden_nodes, action_num):
        super(ReinforceMLP, self).__init__()
        if hidden_nodes is None:
            hidden_nodes= 128
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_nodes)
        self.action_output_layer = nn.Linear(hidden_nodes, action_num)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.action_output_layer(x))
        return action_probs
import copy
def writer(store,data,uid,session_id,filepath):
    data_root = os.path.join("model_inspector","session_data", uid)
    store.delayed_write_by_path(
        copy.deepcopy(data),
        os.path.join(data_root, session_id , filepath))

class ReinforcePolicyEstimator(PolicyEstimator):


    def __init__(
            self,
            input_size,
            action_num,
            learning_rate=0.0001,
            hidden_nodes=None,
            uid = "model",
            log_state_freq = 300
            ):
            ctx = create_sys_context()
            self.net = ReinforceMLP(
                input_size=input_size,
                action_num=action_num,
                hidden_nodes = hidden_nodes)

            self.policy_optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(self.device)

            self.inspector = Inspector(session_id = 'policy', 
            writer=lambda data, session_id, filepath: writer(ctx.get_store(),data,uid,session_id, filepath))
          
            self.update_counter = 0
            self.log_state_freq = log_state_freq

    def predict_choice(self, ob):
        # add dims for channel + batch
        ob = torch.from_numpy(ob.astype('float32')).to(self.device)
        action_probs = self.net(ob)
        
        # From torch to numpy
        action_probs = action_probs.detach().cpu().numpy() 
        logging.debug("action_probs shape:{}".format(action_probs.shape))

        # sample, then 2d to 1d
        choice = np.squeeze(multi_choice(action_probs))
        logging.debug("choice type:{}".format(type(choice)))
        return choice, action_probs

    def update(self, ob, action, rewards):
        # ob = ob.astype("float32")
        ob = torch.from_numpy(ob.astype('float32')).to(self.device)
        action = np.expand_dims(action, axis=0)
        log_action_probs = torch.log(self.net(ob))
        selected_action_probs = log_action_probs[np.arange(len(action)),action] * rewards
        loss = -selected_action_probs

        loss.backward()

        self.policy_optimizer.step()

        if self.update_counter % self.log_state_freq == 0:
            self.inspector.log_state(epoch=0,
                        itr=self.update_counter, 
                        model=self.net,
                        input_dict={"input.1":ob},
                        output_dict={"output":selected_action_probs},
                        loss_dict={'loss':loss},
                        label_dict={})
            self.inspector.log_metrics(
                epoch=0,
                itr=self.update_counter, 
                metrics={'loss':loss.item()})
         
        self.update_counter +=1
        return loss.item()
