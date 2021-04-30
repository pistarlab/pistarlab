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

# because stocastic
def policy_loss_fn(action_selected_prob, action_weights):
    return torch.mean(-torch.log(action_selected_prob) * action_weights)

def multi_choice(v):
    def rchoice(x):
        return np.random.choice(len(x), p=x)
    return np.apply_along_axis(rchoice, 1, v)

def make_ob_4_dims(v):
    if (v.ndim == 1):
        return v.reshape(1,1,1,v.shape[0])
    elif(v.ndim ==2):
        return np.expand_dims(np.expand_dims(v, axis=0),axis=0)
    elif (v.ndim == 3):
        if v.shape[0]>3:
            v = np.transpose(v,(2,1,0))
        return np.expand_dims(v, axis=0)
    elif(v.ndim !=4):
        raise Exception("ob_ndim: {}, should be 4".format(v.ndim))
    else:
        return v

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

import copy
def writer(store,data,uid,session_id,filepath):
    data_root = os.path.join("model_inspector","session_data", uid)
    store.delayed_write_by_path(
        copy.deepcopy(data),
        os.path.join(data_root, session_id , filepath))

class AC2Model(nn.Module):

    def __init__(self, w, h, input_channels, hidden_nodes, action_num, view_size = 400):
        super(AC2Model, self).__init__()
        if hidden_nodes is None:
            hidden_nodes= 128
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.view_size = view_size # w * h * channels 
        self.fc1 = nn.Linear(self.view_size, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, 84)
        self.action_output_layer = nn.Linear(84, action_num)
        self.value_output_layer = nn.Linear(84, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, self.view_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_output_layer(x))
        val_est = self.value_output_layer(x)
        return action_probs, val_est

class SharedPolicyModel(object):
    """
    Policy Function approximator.
    """

    def __init__(
            self,
            ob_input_shape,
            action_output_shape:int,
            hidden_nodes=None,
            uid = "model",
            store = None,
            log_state_freq = 300
            ):
            self.net = AC2Model(
                w = ob_input_shape[1], 
                h = ob_input_shape[2], 
                input_channels = ob_input_shape[0], 
                hidden_nodes = hidden_nodes,
                action_num=action_output_shape,
                view_size = 256)

            # self.policy_optimizer = optim.SGD(self.net.parameters(), lr=p_learning_rate, momentum=0.5)
            self.policy_optimizer = optim.Adam(self.net.parameters())
            self.value_loss_fn = torch.nn.MSELoss(reduction='sum')
            # self.value_optimizer = optim.SGD(self.net.parameters(), lr=v_learning_rate, momentum=0.5)
            self.value_optimizer = optim.Adam(self.net.parameters())           
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(self.device)

            self.inspector = Inspector(session_id = 'shared', writer=lambda data, session_id, filepath: writer(store,data,uid,session_id, filepath))
          
            self.update_counter = 0
            self.log_state_freq = log_state_freq
   

    def print_internals(self):
        pass


    def save(self, prefix, store):
        pass

    def load(self, prefix, store):
        pass

class PolicyEstimatorShared(PolicyEstimator):
    """
    Policy Function approximator.
    """
    _save_name = 'PolicyEstimatorTORCH'

    def __init__(self, model: SharedPolicyModel, log_state_freq=300):
        self.model = model
        self.log_state_freq = log_state_freq
       

    def predict_choice(self, ob):
        # add dims for channel + batch
        ob = make_ob_4_dims(ob.astype(dtype='float32'))
        
        action_probs, _ = self.model.net(torch.from_numpy(ob).to(self.model.device))
        
        # From torch to numpy
        action_probs = action_probs.detach().cpu().numpy() 

        # sample, then 2d to 1d
        choice = np.squeeze(multi_choice(action_probs))
        return choice, action_probs

    def update(self, ob, advantage, action,retain_graph=False):
        ob = make_ob_4_dims(ob.astype(dtype='float32'))
        ob = torch.from_numpy(ob).to(self.model.device)

        action = np.expand_dims(action, axis=0)
        advantage = np.expand_dims(advantage, axis=0)
        advantage = torch.from_numpy(advantage).to(self.model.device)
        
        # Perform prediction again? why not used prior predicted probs?
        action_probs,_ = self.model.net(ob)
        selected_action_probs = action_probs[np.arange(len(action)),action]

        loss = -torch.log(selected_action_probs) * advantage

        loss.backward(retain_graph=retain_graph)
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


class ValueEstimatorShared(ValueEstimator):
    """
    Value Function approximator.
    """

    def __init__(self, model: SharedPolicyModel):
        self.model = model

    def predict(self, ob):
        # add dims for channel + batch
        ob = make_ob_4_dims(ob.astype(dtype='float32'))
        _, value_estimate = self.model.net(torch.from_numpy(ob).to(self.model.device))
        val = value_estimate.detach().cpu().numpy()      
        return np.squeeze(val)

    def update(self, ob, target, retain_graph=False):

        #convert scalar to shape = (1,1)
        target = np.expand_dims(np.expand_dims(target,axis=0),axis=0)
        assert(target.ndim == 2)
        target = torch.from_numpy(target.astype(dtype='float32')).to(self.model.device)
        
        # add dims for channel + batch
        ob = make_ob_4_dims(ob.astype(dtype='float32'))

        ob = torch.from_numpy(ob).to(self.model.device)

        # forward pass , TODO: use value from prior forward pass
        _, value_estimate = self.model.net(ob)

        assert(value_estimate.ndim == 2)

        loss = self.model.value_loss_fn(value_estimate,target)
        loss.backward(retain_graph=retain_graph)
        self.model.value_optimizer.step()
        return loss.item()
