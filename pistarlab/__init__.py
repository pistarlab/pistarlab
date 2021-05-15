import os
import warnings
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging

__version__ = '0.0.1-dev2'

from .core import ctx
from .agent import Agent
from .task import Task, AgentTask
