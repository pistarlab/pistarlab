import logging
import random
from typing import Tuple, Any

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

    def __init__(self, input_size, hidden_nodes, action_num, hidden_nodes2 = 20):
        super(AC2ModelMLP, self).__init__()
        if hidden_nodes is None:
            hidden_nodes= 128
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes2)
        self.action_output_layer = nn.Linear(hidden_nodes2, action_num)
        self.value_output_layer = nn.Linear(hidden_nodes2, 1)
        
    def forward(self, x)->Tuple[Any,Any]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_output_layer(x))
        val_est = self.value_output_layer(x)
        return action_probs, val_est
import copy
def writer(store,data,uid,session_id,filepath):
    data_root = os.path.join("model_inspector","session_data", uid)
    store.delayed_write_by_path(
        copy.deepcopy(data),
        os.path.join(data_root, session_id , filepath))

class PolicyModel:
    """
    Policy Function approximator.
    """

    def __init__(
            self,
            input_size,
            action_num,
            learning_rate=0.05,
            hidden_nodes=None,
            uid = "model",
            store:Storage=None,
            log_state_freq = 300
            ):
            self.net = AC2ModelMLP(
                input_size=input_size,
                action_num=action_num,
                hidden_nodes = hidden_nodes)

            self.policy_optimizer = optim.Adam(self.net.parameters(),lr=learning_rate, eps=1e-3)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(self.device)

            self.inspector = Inspector(session_id = 'policy', writer=lambda data, session_id, filepath: writer(store,data,uid,session_id, filepath))
          
            self.update_counter = 0
            self.log_state_freq = log_state_freq


class ValueModel:
    """
    Policy Function approximator.
    """

    def __init__(
            self,
            input_size,
            action_num,
            learning_rate=0.05,
            hidden_nodes=None,
            uid = "model",
            store:Storage = None,
            log_state_freq = 300
            ):
            self.net = AC2ModelMLP(
                input_size=input_size,
                action_num=action_num,
                hidden_nodes = hidden_nodes)

            self.value_loss_fn = torch.nn.MSELoss(reduction='sum')
            self.value_optimizer = optim.Adam(self.net.parameters(),lr=learning_rate, eps=1e-3)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(self.device)
            self.inspector = Inspector(session_id = 'value', writer=lambda data, session_id, filepath: writer(store,data,uid,session_id, filepath))
            self.update_counter = 0
            self.log_state_freq = log_state_freq

class PolicyEstimatorMLP(PolicyEstimator):
    """
    Policy Function approximator.
    """
    _save_name = 'PolicyEstimatorTORCH'

    def __init__(self, model: AC2ModelMLP, log_state_freq=300):
        self.model = model
        self.log_state_freq = log_state_freq
       

    def predict_choice(self, ob):
        # add dims for channel + batch
        ob = ob.astype("float32")
        action_probs, _ = self.model.net(torch.from_numpy(ob).to(self.model.device))
        
        # From torch to numpy
        action_probs = action_probs.detach().cpu().numpy() 

        # sample, then 2d to 1d
        choice = np.squeeze(multi_choice(action_probs))
        return choice, action_probs

    def update(self, ob, advantage, action):
        ob = ob.astype("float32")
        ob = torch.from_numpy(ob).to(self.model.device)

        advantage = np.expand_dims(advantage, axis=0)
        advantage = torch.from_numpy(advantage).to(self.model.device)

        action = np.expand_dims(action, axis=0)
        
        # Perform prediction again? why not used prior predicted probs?
        action_probs, _ = self.model.net(ob)
        selected_action_probs = action_probs[np.arange(len(action)),action]
        # print(selected_action_probs)
        loss = -torch.mean(torch.log(selected_action_probs) * advantage)

        # loss = torch.mean(torch.log(selected_action_probs) * advantage)
        loss.backward()

        self.model.policy_optimizer.step()

        if self.model.update_counter % self.model.log_state_freq == 0:
            self.model.inspector.log_state(epoch=0,
                        itr=self.model.update_counter, 
                        model=self.model.net,
                        input_dict={"input.1":ob},
                        output_dict={"output":action_probs},
                        loss_dict={'loss':loss},
                        name_dict={})
            self.model.inspector.log_metrics(
                epoch=0,
                itr=self.model.update_counter, 
                metrics={'loss':loss.item()})
         
        self.model.update_counter +=1
        return loss.item()


class ValueEstimatorMLP(ValueEstimator):
    """
    Value Function approximator.
    """

    def __init__(self, model: AC2ModelMLP):
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
        target = np.expand_dims(np.expand_dims(target,axis=0),axis=0)
        assert(target.ndim == 2)
        target = torch.from_numpy(target).to(self.model.device)

        ob = torch.from_numpy(ob).to(self.model.device)

        # forward pass , TODO: use value from prior forward pass
        _, value_estimate = self.model.net(ob)

        assert(value_estimate.ndim == 2)

        loss = self.model.value_loss_fn(value_estimate,target)
        loss.backward()
        self.model.value_optimizer.step()


        if self.model.update_counter % self.model.log_state_freq == 0:
            self.model.inspector.log_state(epoch=0,
                        itr=self.model.update_counter, 
                        model=self.model.net,
                        input_dict={"input.1":ob},
                        output_dict={"output":value_estimate},
                        loss_dict={'loss':loss},
                        name_dict={})
            self.model.inspector.log_metrics(
                epoch=0,
                itr=self.model.update_counter, 
                metrics={'loss':loss.item()})
                
        self.model.update_counter +=1

        return loss.item()
