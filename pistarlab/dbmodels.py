import json

import sqlalchemy as db
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.base import MANYTOONE
from sqlalchemy.schema import UniqueConstraint

from .meta import *


class AlchemyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj.__class__, DeclarativeMeta):
            # an SQLAlchemy class
            fields = {}
            for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    json.dumps(data)  # this will fail on non-encodable values, like other classes
                    fields[field] = data
                except TypeError:
                    fields[field] = None
            # a json-encodable dict
            return fields
        try:
            return json.JSONEncoder.default(self, obj)
        except Exception as e:
            return obj


# JSON stuff
# Note: https://stackoverflow.com/questions/53264047/sqlalchemy-filter-by-json-field
# https://stackoverflow.com/questions/5022066/how-to-serialize-sqlalchemy-result-to-json
# https://amercader.net/blog/beware-of-json-fields-in-sqlalchemy/

DbBase = declarative_base()


class SystemCounter(DbBase):
    __tablename__ = "system_counter"
    id = db.Column(db.String, primary_key=True)
    value = db.Column(db.Integer)


class ComponentSpecModel(DbBase):
    __tablename__ = COMPONENT_SPEC_ENTITY
    id = db.Column(db.String, primary_key=True)
    category = db.Column(db.String)  # Model/Wrapper
    created = db.Column(db.DateTime, server_default=db.func.now())
    displayed_name = db.Column(db.String)
    entry_point = db.Column(db.String)
    description = db.Column(db.String)
    parent_class_entry_point = db.Column(db.String)

    extension_id = db.Column(db.String, nullable=False)
    extension_version = db.Column(db.String, nullable=False)

    disabled = db.Column(db.Boolean, nullable=False, default=False)

    version = db.Column(db.String, default=False)

    config = db.Column(db.JSON)
    params = db.Column(db.JSON)
    meta = db.Column(db.JSON)

    components = relationship("ComponentModel", back_populates="spec")

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)


class ComponentModel(DbBase):
    __tablename__ = COMPONENT_ENTITY
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String, nullable=False)
    version = db.Column(db.String)
    config = db.Column(db.JSON)
    meta = db.Column(db.JSON)
    created = db.Column(db.DateTime, server_default=db.func.now())
    archived = db.Column(db.Boolean, default=False)

    spec_id = db.Column(db.String, db.ForeignKey(f'{COMPONENT_SPEC_ENTITY}.id'))
    spec = relationship("ComponentSpecModel", back_populates="components")

    # Agent
    agent_id = db.Column(db.String, db.ForeignKey(f'{AGENT_ENTITY}.id'))
    agent = relationship("AgentModel", back_populates="components")

    # Sub Components
    parent_component_id = db.Column(db.String, db.ForeignKey(f'{COMPONENT_ENTITY}.id'))
    parent_component = relationship("ComponentModel", back_populates=('child_components'))
    child_components = relationship("ComponentModel")

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)


# ----------------------------------------------------
# ---Agent--------------------------------------------
# ----------------------------------------------------
class AgentSpecModel(DbBase):
    __tablename__ = AGENT_SPEC_ENTITY
    id = db.Column(db.String, primary_key=True)
    created = db.Column(db.DateTime, server_default=db.func.now())
    displayed_name = db.Column(db.String)

    entry_point = db.Column(db.String)
    runner_entry_point = db.Column(db.String)

    description = db.Column(db.String)

    extension_id = db.Column(db.String, nullable=False)
    extension_version = db.Column(db.String, nullable=False)

    disabled = db.Column(db.Boolean, nullable=False, default=False)

    version = db.Column(db.String, nullable=False)

    config = db.Column(db.JSON)
    params = db.Column(db.JSON)
    meta = db.Column(db.JSON)

    # Agents
    agents = relationship("AgentModel", back_populates="spec")

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)


class AgentTagModel(DbBase):
    __tablename__ = "agent_tag"
    tag_id = db.Column(db.String, primary_key=True)
    agent_id = db.Column(db.String, db.ForeignKey(f'{AGENT_ENTITY}.id'), primary_key=True)

    agent = relationship("AgentModel", back_populates="tags")


class AgentModel(DbBase):
    __tablename__ = AGENT_ENTITY
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String)
    seed = db.Column(db.String, nullable=False)

    displayed_name = db.Column(db.String)
    notes = db.Column(db.String)
    created = db.Column(db.DateTime, server_default=db.func.now())
    archived = db.Column(db.Boolean, default=False)

    config = db.Column(db.JSON)
    last_checkpoint = db.Column(db.JSON)

    spec_id = db.Column(db.String, db.ForeignKey(f'{AGENT_SPEC_ENTITY}.id'))
    spec: AgentSpecModel = relationship("AgentSpecModel", back_populates="agents")

    meta = db.Column(db.JSON)
    stats = db.Column(db.JSON)

    sessions = relationship("SessionModel", back_populates="agent")

    components = relationship("ComponentModel", back_populates="agent")

    tags = relationship("AgentTagModel", back_populates="agent")

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)


# ----------------------------------------------------
# ---Env--------------------------------------------
# ----------------------------------------------------
class EnvironmentModel(DbBase):
    __tablename__ = "environment"
    id = db.Column(db.String, primary_key=True)

    displayed_name = db.Column(db.String)
    description = db.Column(db.String())
    categories = db.Column(db.String)
    created = db.Column(db.DateTime, server_default=db.func.now())

    extension_id = db.Column(db.String, nullable=False)
    extension_version = db.Column(db.String, nullable=False)

    version = db.Column(db.String, nullable=False)
    disabled = db.Column(db.Boolean, nullable=False, default=False)

    default_entry_point = db.Column(db.String)
    default_config = db.Column(db.JSON)
    default_meta = db.Column(db.JSON)

    env_specs = relationship('EnvSpecModel', back_populates=("environment"))

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)


class EnvSpecModel(DbBase):
    __tablename__ = ENV_SPEC_ENTITY
    id = db.Column(db.String, primary_key=True)  # spec_id
    displayed_name = db.Column(db.String)
    entry_point = db.Column(db.String)
    description = db.Column(db.String)
    created = db.Column(db.DateTime, server_default=db.func.now())
    user_defined = db.Column(db.Boolean, nullable=False, default=False)
    env_type = db.Column(db.String, nullable=False)

    config = db.Column(db.JSON)
    params = db.Column(db.JSON)
    meta = db.Column(db.JSON)
    tags = db.Column(db.String)

    environment_id = db.Column(db.String, db.ForeignKey(f'environment.id'))
    environment = relationship("EnvironmentModel", back_populates=('env_specs'))

    # Sessions
    sessions = relationship("SessionModel", back_populates="env_spec")

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)


# ----------------------------------------------------
# ---Task--------------------------------------------
# ----------------------------------------------------
class TaskSpecModel(DbBase):
    __tablename__ = TASK_SPEC_ENTITY
    id = db.Column(db.String, primary_key=True)
    created = db.Column(db.DateTime, server_default=db.func.now())
    displayed_name = db.Column(db.String)
    entry_point = db.Column(db.String)
    runner_entry_point = db.Column(db.String)
    description = db.Column(db.String)
    type_name = db.Column(db.String)

    extension_id = db.Column(db.String, nullable=False)
    extension_version = db.Column(db.String, nullable=False)

    disabled = db.Column(db.Boolean, nullable=False, default=False)

    version = db.Column(db.String, nullable=True)

    config = db.Column(db.JSON)
    params = db.Column(db.JSON)
    meta = db.Column(db.JSON)

    tasks = relationship("TaskModel", back_populates="spec")

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)


class TaskModel(DbBase):
    __tablename__ = TASK_ENTITY
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String)
    actor_uid = db.Column(db.String)
    displayed_name = db.Column(db.String)
    notes = db.Column(db.String)
    created = db.Column(db.DateTime, server_default=db.func.now())
    archived = db.Column(db.Boolean, default=False)
    type_code = db.Column(db.Integer, nullable=False, default=1)

    config = db.Column(db.JSON)

    status = db.Column(db.String, default=STATE_CREATED)
    status_msg = db.Column(db.String)
    status_timestamp = db.Column(db.DateTime, server_default=db.func.now())

    spec_id = db.Column(db.String, db.ForeignKey(f'{TASK_SPEC_ENTITY}.id'))
    spec = relationship("TaskSpecModel", back_populates=('tasks'))
    meta = db.Column(db.JSON)

    parent_task_id = db.Column(db.String, db.ForeignKey(f'{TASK_ENTITY}.id'))
    parent_task = relationship("TaskModel", back_populates=('child_tasks'), remote_side="TaskModel.id")
    child_tasks = relationship("TaskModel", remote_side="TaskModel.parent_task_id")

    primary_session_id = db.Column(db.String, db.ForeignKey(f'{SESSION_ENTITY}.id'))
    primary_session = relationship("SessionModel", primaryjoin="TaskModel.primary_session_id==SessionModel.id")

    sessions = relationship("SessionModel", primaryjoin="TaskModel.id==SessionModel.task_id")

    summary = db.Column(db.JSON)

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)

# ----------------------------------------------------
# ---Session------------------------------------------
# ----------------------------------------------------


class SessionModel(DbBase):

    __tablename__ = SESSION_ENTITY
    id = db.Column(db.String, primary_key=True)
    label = db.Column(db.String)
    comments = db.Column(db.String)
    created = db.Column(db.DateTime, server_default=db.func.now())
    archived = db.Column(db.Boolean, default=False)

    session_type = db.Column(db.String)  # "RLSession, RLParentSession, "

    # Agent
    agent_id = db.Column(db.String, db.ForeignKey(f'{AGENT_ENTITY}.id'))
    agent = relationship("AgentModel", back_populates="sessions")
    agent_run_config = db.Column(db.JSON)

    # Env Spec
    env_spec_id = db.Column(db.String, db.ForeignKey(f'{ENV_SPEC_ENTITY}.id'))
    env_spec = relationship("EnvSpecModel", back_populates=('sessions'))
    env_spec_version = db.Column(db.String)
    env_spec_config = db.Column(db.JSON)

    # Session Config
    config = db.Column(db.JSON)

    status = db.Column(db.String, default=STATE_CREATED)
    status_msg = db.Column(db.String)
    status_timestamp = db.Column(db.DateTime, server_default=db.func.now())

    parent_session_id = db.Column(db.String, db.ForeignKey(f'{SESSION_ENTITY}.id'))
    parent_session = relationship("SessionModel", back_populates=('child_sessions'), remote_side="SessionModel.id", primaryjoin="SessionModel.parent_session_id==SessionModel.id")
    child_sessions = relationship("SessionModel", primaryjoin="SessionModel.id==SessionModel.parent_session_id")

    run_info = db.Column(db.JSON)
    summary = db.Column(db.JSON)

    task_id = db.Column(db.String, db.ForeignKey(f'task.id'))
    task = relationship("TaskModel", back_populates="sessions", primaryjoin="TaskModel.id==SessionModel.task_id")

    linked_tasks = relationship("TaskModel", back_populates="primary_session", primaryjoin="TaskModel.primary_session_id==SessionModel.id")

    def __repr__(self):
        return json.dumps(self, cls=AlchemyEncoder)


# Some aliases
# TODO: Remove
EnvSpec = EnvSpecModel
AgentSpec = AgentSpecModel
TaskSpec = TaskSpecModel
