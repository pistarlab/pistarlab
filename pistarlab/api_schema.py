import json
import math
from datetime import timezone

import graphene
from graphene import relay
from graphene.types.generic import GenericScalar
from graphene_sqlalchemy import SQLAlchemyConnectionField, SQLAlchemyObjectType
from graphene_sqlalchemy_filter import FilterableConnectionField, FilterSet
from graphql_relay.node.node import from_global_id

import pistarlab.dbmodels as db

from .meta import *


def updateRLSummaryWithDetails(summary):
    if summary is None:
        return {}
    if math.isnan(float(summary['reward_mean_windowed'])):
        summary['reward_mean_windowed'] = 0

    if math.isnan(float(summary['reward_mean_windowed'])):
        summary['reward_mean_windowed'] = 0

    if summary['episode_count'] == 0:
        summary['mean_steps_per_episode'] = 0
    else:
        summary['mean_steps_per_episode'] = summary['step_count'] / summary['episode_count']

    if summary['step_count'] == 0:
        summary['mean_reward_per_step'] = 0
    else:
        summary['mean_reward_per_step'] = summary['reward_total'] / summary['step_count']

    if summary['episode_count'] == 0:
        summary['mean_reward_per_episode'] = 0
    else:
        summary['mean_reward_per_episode'] = summary['reward_total'] / summary['episode_count']

    delta = summary.get('runtime', 0)
    if delta == 0:
        summary['steps_per_second'] = "0.0"
    else:
        summary['steps_per_second'] = summary['step_count'] / delta
    return summary

    # @staticmethod
    # def resolve_steps_per_second(self, info):
    #     if self.summary is None:
    #         return 0.0
    #     delta = self.summary.get('runtime', 0)
    #     total_steps = self.summary['step_count']
    #     if delta == 0:
    #         return 0.0
    #     else:
    #         return total_steps / delta


class ComponentSpec(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    class Meta:
        model = db.ComponentSpecModel
        interfaces = (relay.Node, )


class Component(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    class Meta:
        model = db.ComponentModel
        interfaces = (relay.Node, )


class AgentSpec(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    class Meta:
        model = db.AgentSpecModel
        interfaces = (relay.Node, )


class AgentTag(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    class Meta:
        model = db.AgentTagModel
        interfaces = (relay.Node, )


class Agent(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    recent_sessions = graphene.List(lambda: Session, limit=graphene.Int())

    def resolve_recent_sessions(self, info, limit=3):
        query = Session.get_query(info).filter(db.SessionModel.agent_id == self.id)
        return query.order_by(db.SessionModel.created.desc()).limit(limit).all()

    status = graphene.String()

    class Meta:
        model = db.AgentModel
        interfaces = (relay.Node, )


class AgentFilter(FilterSet):
    archived = graphene.Boolean()

    class Meta:
        model = db.AgentModel
        fields = {
            'archived': [...],  # shortcut!
        }


class AgentSetArchive(graphene.Mutation):

    class Arguments:
        id = graphene.String()
        archive = graphene.Boolean()

    success = graphene.Boolean()

    @staticmethod
    def mutate(root, info, id, archive):
        ident = from_global_id(id)
        ctx = info.context.get('ctx')
        agent: db.AgentModel = ctx.get_agent_dbmodel(ident[1])
        agent.archived = archive
        ctx.get_dbsession().commit()
        return AgentSetArchive(success=True)


class AgentSetConfig(graphene.Mutation):

    class Arguments:
        id = graphene.String()
        config = graphene.String()

    success = graphene.Boolean()

    @staticmethod
    def mutate(root, info, id, config):
        ident = from_global_id(id)
        ctx = info.context.get('ctx')
        agent: db.AgentModel = ctx.get_agent_dbmodel(ident[1])
        agent.config = json.loads(config)
        ctx.get_dbsession().commit()
        return AgentSetConfig(success=True)


class AgentSetNotes(graphene.Mutation):

    class Arguments:
        id = graphene.String()
        notes = graphene.String()

    success = graphene.Boolean()

    @staticmethod
    def mutate(root, info, id, notes):
        ident = from_global_id(id)
        ctx = info.context.get('ctx')
        agent: db.AgentModel = ctx.get_agent_dbmodel(ident[1])
        agent.notes = notes
        ctx.get_dbsession().commit()
        return AgentSetNotes(success=True)


class AgentSetName(graphene.Mutation):

    class Arguments:
        id = graphene.String()
        name = graphene.String()

    success = graphene.Boolean()

    @staticmethod
    def mutate(root, info, id, name):
        ident = from_global_id(id)
        ctx = info.context.get('ctx')
        agent: db.AgentModel = ctx.get_agent_dbmodel(ident[1])
        agent.name = name
        ctx.get_dbsession().commit()
        return AgentSetName(success=True)

class EnvWrapper(graphene.ObjectType):
    entry_point = graphene.String(name="entry_point")
    kwargs = graphene.JSONString()


class Environment(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    class Meta:
        model = db.EnvironmentModel
        interfaces = (relay.Node, )

    specs = graphene.List(lambda: EnvSpec)

    def resolve_specs(self, info):
        query = EnvSpec.get_query(info)
        query = query.filter(db.EnvSpecModel.environment_id == self.id)
        return query.all()


class EnvSpec(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    class Meta:
        model = db.EnvSpecModel
        interfaces = (relay.Node, )


class TaskSpec(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    class Meta:
        model = db.TaskSpecModel
        interfaces = (relay.Node, )


class Task(SQLAlchemyObjectType):
    ident = graphene.String(source='id')

    class Meta:
        model = db.TaskModel
        interfaces = (relay.Node, )

    parent_task = graphene.Field(lambda: Task)

    def resolve_parent_task(self, info):
        query = Task.get_query(info)
        return query.get(self.parent_task_id)


class Session(SQLAlchemyObjectType):
    ident = graphene.String(source='id')
    summary_raw = graphene.JSONString(source='summary')
    # summary = graphene.Field(SessionSummary)
    # task = Task(source="task")

    class Meta:
        model = db.SessionModel
        interfaces = (relay.Node, )

    summary = GenericScalar()

    def resolve_summary(self, info, **args):
        if self.session_type == "DATA_SESSION":
            return self.summary
        else:
            return updateRLSummaryWithDetails(self.summary)

    #TODO: verify task (parent) status as well
    # status = graphene.String()

#     @staticmethod
    # def resolve_status(self,info,**args):
    #     self.status
    #     self.task.status

    #     pass

    created_timestamp = graphene.Float()

    def resolve_created_timestamp(self, info):

        return self.created.replace(tzinfo=timezone.utc).timestamp()

    # steps_per_second = graphene.Float()

    # @staticmethod
    # def resolve_steps_per_second(self, info):
    #     if self.summary is None:
    #         return 0.0
    #     delta = self.summary.get('runtime', 0)
    #     total_steps = self.summary['step_count']
    #     if delta == 0:
    #         return 0.0
    #     else:
    #         return total_steps / delta


class SessionSetArchive(graphene.Mutation):

    class Arguments:
        id = graphene.String()
        archive = graphene.Boolean()

    success = graphene.Boolean()

    @staticmethod
    def mutate(root, info, id, archive):
        ident = from_global_id(id)
        ctx = info.context.get('ctx')
        item: db.SessionModel = ctx.get_session(ident[1])
        item.archived = archive
        ctx.get_dbsession().commit()
        return SessionSetArchive(success=True)


class SessionFilter(FilterSet):
    archived = graphene.Boolean()

    class Meta:
        model = db.SessionModel
        fields = {
            'archived': [...],  # shortcut!
        }


class Mutations(graphene.ObjectType):
    agent_set_archive = AgentSetArchive.Field()
    agent_set_config = AgentSetConfig.Field()
    agent_set_notes = AgentSetNotes.Field()
    agent_set_name = AgentSetName.Field()
    session_set_archive = SessionSetArchive.Field()


class Query(graphene.ObjectType):
    node = relay.Node.Field()

    # Agents
    agent = graphene.Field(Agent, ident=graphene.String())

    @staticmethod
    def resolve_agent(parent, info, ident):
        query = Agent.get_query(info)
        return query.get(ident)

    # Agents
    env_spec = graphene.Field(EnvSpec, ident=graphene.String())

    @staticmethod
    def resolve_env_spec(parent, info, ident):
        query = EnvSpec.get_query(info)
        return query.get(ident)

    agents = graphene.List(lambda: Agent)

    @staticmethod
    def resolve_agents(parent, info):
        query = Agent.get_query(info)
        return query.all()

    agent_tags = graphene.List(lambda: Agent)

    @staticmethod
    def resolve_agent_tags(parent, info):
        query = AgentTag.get_query(info)
        return query.all()

    recent_agents = graphene.List(lambda: Agent, limit=graphene.Int())

    @staticmethod
    def resolve_recent_agents(parent, info, limit=12):
        query = Agent.get_query(info).filter(db.AgentModel.archived == False)
        return query.order_by(db.AgentModel.created.desc()).limit(limit).all()

    # Agent Spec
    agent_spec = graphene.Field(AgentSpec, ident=graphene.String())

    @staticmethod
    def resolve_agent_spec(parent, info, ident):
        query = AgentSpec.get_query(info)
        return query.get(ident)

    agent_specs = graphene.List(lambda: AgentSpec, disabled=graphene.Boolean())

    @staticmethod
    def resolve_agent_specs(parent, info, disabled=None):
        query = AgentSpec.get_query(info)
        if disabled:
            query = query.filter(db.AgentSpecModel.disabled == disabled)
        return query.all()

    # component Spec
    component_spec = graphene.Field(ComponentSpec, ident=graphene.String())

    @staticmethod
    def resolve_component_spec(parent, info, ident):
        query = ComponentSpec.get_query(info)
        return query.get(ident)

    # component_specs = graphene.List(lambda: ComponentSpec, disabled=graphene.Boolean())

#     @staticmethod
    # def resolve_component_specs(parent, info, disabled=None):
    #     query = ComponentSpec.get_query(info)
    #     if disabled:
    #         query = query.filter(db.ComponentSpecModel.disabled == disabled)
    #     return query.all()

    # TaskSpec Specs
    task_spec = graphene.Field(TaskSpec, ident=graphene.String())

    @staticmethod
    def resolve_task_spec(parent, info, ident):
        query = TaskSpec.get_query(info)
        return query.get(ident)

    task_specs = graphene.List(lambda: TaskSpec, disabled=graphene.Boolean())

    @staticmethod
    def resolve_task_specs(parent, info, disabled=None):
        query = TaskSpec.get_query(info)
        if disabled:
            query = query.filter(db.TaskSpecModel.disabled == disabled)
        return query.all()

    # Env Specs
    env_specs = graphene.List(lambda: EnvSpec, disabled=graphene.Boolean())

    @staticmethod
    def resolve_env_specs(parent, info, disabled=None):
        query = EnvSpec.get_query(info)
        if disabled:
            query = query.filter(db.EnvSpecModel.disabled == disabled)
        return query.order_by(db.EnvSpecModel.id.asc()).all()

    # Best Session
    best_sessions_for_env_spec = graphene.List(lambda: Session, env_spec_id=graphene.String(), stat_name=graphene.String())

    @staticmethod
    def resolve_best_sessions_for_env_spec(parent, info, env_spec_id, stat_name):
        query = Session.get_query(info)
        from sqlalchemy import cast, text
        from sqlalchemy.types import FLOAT, JSON
        # TODO: not using backend for sorting due to json sort issues
        # from sqlalchemy.dialects.postgres import JSON
        return query \
            .filter(db.SessionModel.env_spec_id == env_spec_id).all()
            # .order_by(text(f"CAST(session.summary->>'{stat_name}' AS FLOAT) DESC")) \
            

    all_environments = SQLAlchemyConnectionField(Environment.connection)

    # Sessions
    session = graphene.Field(Session, ident=graphene.String())

    @staticmethod
    def resolve_session(parent, info, ident):
        query = Session.get_query(info)
        return query.get(ident)

    sessions = graphene.List(lambda: Session)

    @staticmethod
    def resolve_sessions(parent, info):
        query = Session.get_query(info)
        return query.all()

    all_agents = FilterableConnectionField(Agent.connection, filters=AgentFilter())

    all_agent_tags = SQLAlchemyConnectionField(AgentTag.connection)

    all_sessions = FilterableConnectionField(Session.connection, filters=SessionFilter())  # SQLAlchemyConnectionField(Session.connection)

    all_component_specs = SQLAlchemyConnectionField(ComponentSpec.connection)

    all_task = SQLAlchemyConnectionField(Task.connection)

    env_wrappers = graphene.List(lambda: EnvWrapper)

    @staticmethod
    def resolve_env_wrappers(parent, info, **kwargs):
        return sorted([
            {'entry_point': "gym_minigrid.wrappers:ImgObsWrapper", 'kwargs': {}},
            {'entry_point': "gym_minigrid.wrappers:RGBImgPartialObsWrapper", 'kwargs': {}},
            {'entry_point': "pistarlab.wrappers.default:ResizeImageObWrapper", 'kwargs': {}},
            {"entry_point": "pistarlab_petting_zoo.wrappers:PettingZooAECWrapper", "kwargs": {}},
            {"entry_point": "supersuit:color_reduction_v0", "kwargs": {}},
            {"entry_point": "supersuit:resize_v0", "kwargs": {}},
            {"entry_point": "supersuit:dtype_v0", "kwargs": {}},
            {"entry_point": "supersuit:flatten_v0", "kwargs": {}},
            {"entry_point": "supersuit:reshape_v0", "kwargs": {}},
            {"entry_point": "supersuit:normalize_obs_v0", "kwargs": {}},
            {"entry_point": "supersuit:frame_stack_v1", "kwargs": {}},
            {"entry_point": "supersuit:pad_observations_v0", "kwargs": {}},
            {"entry_point": "supersuit:pad_action_space_v0", "kwargs": {}},
            {"entry_point": "supersuit:agent_indicator_v0", "kwargs": {}},
            {"entry_point": "supersuit:reward_lambda_v0", "kwargs": {}},
            {"entry_point": "supersuit:clip_reward_v0", "kwargs": {}},
            {"entry_point": "supersuit:clip_actions_v0", "kwargs": {}},
            {"entry_point": "supersuit:frame_skip_v0", "kwargs": {}},
            {"entry_point": "supersuit:pad_observations_v0", "kwargs": {}},
            {"entry_point": "supersuit:sticky_actions_v0", "kwargs": {}},
            {"entry_point": "supersuit:delay_observations_v0", "kwargs": {}},
        ], key=lambda x: x['entry_point'])

    # Task
    task = graphene.Field(Task, ident=graphene.String())

    @staticmethod
    def resolve_task(parent, info, ident):
        query = Task.get_query(info)
        return query.get(ident)

    session_list = graphene.List(lambda: Session, idents=graphene.String())

    @staticmethod
    def resolve_session_list(parent, info, idents):
        query = Session.get_query(info)
        return [query.get(ident) for ident in idents.split(",")]

    tasks = graphene.List(lambda: Task)

    @staticmethod
    def resolve_tasks(parent, info):
        query = Task.get_query(info)
        return query.all()


schema = graphene.Schema(query=Query, mutation=Mutations)

# https://docs.graphene-python.org/en/latest/types/mutations/
# Filter->https://docs.graphene-python.org/projects/sqlalchemy/en/latest/examples/
# https://github.com/art1415926535/graphene-sqlalchemy-filter
# https://docs.graphene-python.org/en/latest/types/mutations/
# https://graphql.org/learn/queries/
# https://www.howtographql.com/graphql-python/3-mutations/
# {
#   allAgents(filters: {archived: true}) {
#     edges {
#       node {
#         seed
#       }
#     }
#   }
# }
# mutation myFirstMutation {
#   agentSetArchive(id:"QWdlbnQ6QS0xNQ==", archive:true){
#    success
#   }
# }
