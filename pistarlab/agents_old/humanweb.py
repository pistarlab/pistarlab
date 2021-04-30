from ..config import AgentConfig
from ..common import Action, Observation
from ..session import Environment
from ..utils.misc import gen_uid
from ..agent import StepperAgent, Agent
import pygame
import json
import PIL.Image
import numpy as np
import os
import cv2
class HumanWebConfig(AgentConfig):
    pass

class HumanWeb(StepperAgent):
    
    @staticmethod
    def instance_from_config(config: AgentConfig,**kwargs):
        return HumanWeb(config=config,**kwargs)

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.fps_limit = 30
        self.clock = pygame.time.Clock()


    def start_episode(self, ob: Observation,session:Environment=None) -> Action:       
        return self.step(ob,session)
    
    def step(self, ob: Observation,session:Environment=None) -> Action:
        rc = ctx.get_redis_client()    

        self.clock.tick(self.fps_limit)
        rc.set("AGENT_SESSION_SUMMARY__{}".format(self.get_uid()),json.dumps(session.get_summary()))
        
        #TODO: Should only do this if connection is established - other option is use a streaming sever
        # Note file will not task after moving to multiple nodes
        preview = session.env.get_preview()
        filename = os.path.join(ctx.get_store().root_path,self.entity_type,self.get_uid(),'snap.jpg')
        cv2.imwrite(filename, preview)

        ##im = PIL.Image.fromarray(preview)
        ##im.save(os.path.join(ctx.get_store().root_path,self.entity_type,self.get_uid(),'snap.png'))
        # rc.set("AGENT_PREVIEW__{}".format(self.get_uid()),img_bytes)
        # rc.set("AGENT_OB__{}".format(self.get_uid()),ob.value)
        # rc.set("AGENT_REWARD__{}".format(self.get_uid()),ob.reward)
        # rc.set("AGENT_DONE__{}".format(self.get_uid()),ob.done)
        # rc.set("AGENT_INFO__{}".format(self.get_uid()),ob.info)
        # action_value = rc.get("AGENT_ACTION__{}".format(self.get_uid()))


        return Action(self.config.action_space.sample())

    def end_episode(self,ob: Observation,session:Environment=None) -> Action:
        return Action(None)
        

