import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
import traceback

from flask import (Flask, Response, g, jsonify, make_response, redirect,
                   request, send_file, send_from_directory)
from flask_cors import CORS
from flask_graphql import GraphQLView
from matplotlib import cm
from pkg_resources import working_set
from pyinstrument import Profiler

from pistarlab import Agent, ctx
from pistarlab.agent import Agent
from pistarlab.api_schema import schema
from pistarlab.dbmodels import *
from pistarlab.meta import *
from pistarlab.meta import AGENT_ENTITY, F_LABEL
from pistarlab.task import AgentTask, Task
from functools import wraps

app = Flask(__name__)

CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# log.disabled = False
# app.logger.disabled = False


class CustomGraphQLView(GraphQLView):
    def get_context(self):
        return {
            'session': ctx.get_dbsession(),
            'ctx': ctx,
            'request': super().get_context()}


# GOod tutorial here: https://jeffersonheard.github.io/python/graphql/2018/12/08/graphene-python.html
app.add_url_rule(
    '/graphql',
    view_func=CustomGraphQLView.as_view(
        'graphql',
        schema=schema,
        graphiql=True  # for having the GraphiQL interface
    )
)


def not_in_readonly_mode(fn):
    @wraps(fn)
    def decorator(*args, **kwargs):
        if ctx.config.read_only_mode:
            msg = "Not available in read-only mode"
            ctx.get_logger().error(msg)
            response = make_response({'item': {'error': msg}})
            response.headers['Content-Type'] = 'application/json'
            return response
        return fn(*args, **kwargs)
    return decorator


@app.teardown_appcontext
def request_teardown(exception=None):
    ctx.get_data_context().close()

# ---------------------------------------
# -----------Admin Tools-----------------
# ---------------------------------------
@app.route("/api/admin/task/<command>/<uid>")
def api_task_admin(command, uid):
    success = False
    message = ""
    try:
        if command == "stop":
            task: Task = Task.load(uid)
            task.shutdown()
            success = True
            message = "Task stop Request Sucessful for {}".format(uid)
        elif command == "run":
            task = Task.load(uid)
            task.run()
            success = True
            message = "Task Run Request Sucessful for {}".format(uid)
        else:
            success = False
            message = "Error"
    except Exception as e:
        success = False
        message = "Error during {} request of TASK with id {}.\nException: {},\nTraceback: {}".format(
            command, uid, e, traceback.format_exc())
    response = make_response({'message': message, 'success': success})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/admin_data/")
@not_in_readonly_mode
def api_admin_data():
    ray = ctx.get_execution_context().ray
    data = {}
    data['available_resources'] = ray.available_resources()
    data['cluster_resources'] = ray.cluster_resources()
    data['available_resources'] = ray.available_resources()
    data['objects'] = ray.objects()
    data['nodes'] = ray.nodes()
    data['get_resource_ids'] = ray.get_resource_ids()
    data['get_gpu_ids'] = ray.get_gpu_ids()
    data['gpu_info'] = ctx.get_gpu_info()
    data['pistar_config'] = ctx.config.__dict__

    data['pistar_config']['redis_password'] = "HIDDEN"
    data['pistar_config']['db_config']['db_password'] = "HIDDEN"

    try:
        data['tensorflow_status'] = ctx.check_tensorflow_status()
    except Exception as e:
        logging.error(e)

    try:
        data['torch_status'] = ctx.check_torch_status()
    except Exception as e:
        logging.error(e)

    response = make_response({'data': data})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/workspace/")
def api_workspace_data():
    workspace = ctx.get_workspace_info()
    response = make_response({'data': workspace})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/overview/")
def api_overview_data():
    # TODO: Cleanup/speedup by using COUNT query
    data = {}
    data['total_agents'] = len(ctx.list_agents())
    data['total_agent_specs'] = len(ctx.list_agent_specs())
    data['total_env_specs'] = len(ctx.list_env_specs())
    data['total_installed_extensions'] = len(
        ctx.list_extensions(status_filter="INSTALLED"))
    data['total_sessions'] = len(ctx.list_sessions())
    data['active_sessions'] = len(
        ctx.list_sessions(status_filter={STATE_RUNNING}))
    response = make_response(data)
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/status/")
def api_status_data():
    data = {}
    response = make_response(data)
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/reload_default_data/")
@not_in_readonly_mode
def api_reload_default_data():
    message = ""
    from .dbinit import load_default_data
    try:
        load_default_data()
        message = "DONE"
    except Exception as e:
        message = f"failed {e}"
        logging.error(e)
    response = make_response({'data': message})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/open_extension_with_ide/<extension_id>")
@not_in_readonly_mode
def open_extension_with_ide(extension_id):
    workspace = ctx.get_workspace_info()
    selected_extension = None
    for extension in workspace['extensions']:
        if extension['id'] == extension_id:
            selected_extension = extension
            break

    success = False
    if selected_extension is not None:
        extension_path = selected_extension['full_path']
        try:
            subprocess.check_output(f"code {extension_path}", shell=True)
            success = True
        except Exception as e:
            message = f"Failed to launch  VSCode {e}"

    response = make_response({'success': success, "message": message})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/check_for_ide/")
@not_in_readonly_mode
def api_check_for_ide():
    success = False
    message = ""
    try:
        output = subprocess.check_output(f"code --version", shell=True)
        message = f"VS Code Found: {output}"
        success = True
    except Exception as e:
        message = f"VSCode not found? Error: {e}"
    response = make_response({'success': success, "message": message})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/set_log_level/<level>")
@not_in_readonly_mode
def set_log_level(level):
    if level == "INFO":
        log.setLevel(logging.INFO)
        success = True
    elif level == "ERROR":
        log.setLevel(logging.ERROR)
        success = True
    else:
        success = False

    response = make_response({'success': success})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/workspace/info")
def api_workspace_info():
    workspace_info = {}
    workspace_info['path'] = ctx.config.workspace_path
    response = make_response({'data': workspace_info})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/extension/create", methods=['POST'])
@not_in_readonly_mode
def api_extension_create():
    try:
        logging.info("Create Extension")
        data = request.get_json()
        ctx.create_new_extension(
            data['extension_id'], data['extension_name'], data['description'])
        response = make_response({'successful': True})
    except Exception as e:
        logging.error(e)
        response = make_response({'successful': False, 'error': "{}".format(
            e), "traceback": traceback.format_exc()})
    response.headers['Content-Type'] = 'application/json'
    return response

# -----------------------------------------
#              extensions
# -----------------------------------------
@app.route("/api/extensions/list")
def api_extensions_list():
    results = ctx.extension_manager.get_all_extensions()
    response = make_response({'items': results})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/extension/view/<extension_id>")
def api_extension_view(extension_id):
    results = ctx.extension_manager.get_extensions_by_id(extension_id)[0]
    response = make_response({'item': results})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/extensions/action/<action_name>/<extension_id>/<extension_version>")
@not_in_readonly_mode
def api_extension_action(action_name, extension_id, extension_version):
    if action_name == 'install':
        result = ctx.extension_manager.install_extension(
            extension_id, extension_version)
    elif action_name == 'uninstall':
        result = ctx.extension_manager.uninstall_extension(extension_id)
    elif action_name == 'reload':
        result = ctx.extension_manager.reload_extension_by_id(extension_id)
    else:
        result = False
    response = make_response({"success": result})
    response.headers['Content-Type'] = 'application/json'
    logging.info("Extension action complete: {}".format(result))

    return response

# -----------------------------------------
#              Snapshots
# -----------------------------------------
@app.route("/api/snapshots/list/<spec_id>")
def api_snapshots_list(spec_id=None):
    if spec_id == "undefined":
        spec_id = None
    snapshots = [entry for entry in ctx.get_snapshot_index(
    )['entries'].values() if spec_id is None or entry['spec_id'] == spec_id]
    response = make_response({'items': snapshots})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/snapshots/agent/list/<seed>")
def api_snapshots_list_for_agent_id(seed):
    snapshots = [entry for entry in ctx.get_snapshot_index(
    )['entries'].values() if entry['seed'] == seed]
    response = make_response({'items': snapshots})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/snapshot/publish", methods=['POST'])
@not_in_readonly_mode
def api_snapshot_publish():
    try:
        request_data = request.get_json()
        snapshot_version = request_data['snapshot_version']
        snapshot_description = request_data['snapshot_description']
        agent_id = request_data['agent_id']

        snapshot_data = ctx.create_agent_snapshot(
            entity_id=agent_id,
            snapshot_description=snapshot_description,
            snapshot_version=snapshot_version)
        ctx.update_snapshot_index()
        response = make_response({'item': {'snapshot_data': snapshot_data}})
    except Exception as e:
        response = make_response(
            {'item': {'error': "{}".format(e), "traceback": traceback.format_exc()}})
    response.headers['Content-Type'] = 'application/json'
    return response


def get_session_update(sid):
    return json.loads(ctx.get_redis_client().get(f"SESSION_STATUS_{sid}").decode())
    

@app.route("/api/session_command/<sid>/<command>", methods=['POST'])
@not_in_readonly_mode
def api_session_command(sid, command):
    try:
        params = request.get_json()
        logging.info(f"Sending command {command} with params: {params}")
        before_update = get_session_update(sid)['last_update']
        ctx.get_redis_client().publish(
            f"SESSION_COMMAND_REQUEST_{sid}",
            json.dumps({
                'command': command,
                'params': params
            }))
        data = {}
        for i in range(0,10):
            time.sleep(0.5)
            data = get_session_update(sid)
            if data['last_update'] > before_update:
                break
            
        response = make_response(data)

    except Exception as e:
        response = make_response(
            {'item': {'error': "{}".format(e), "traceback": traceback.format_exc()}})

    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/session_runtime_status/<sid>")
@not_in_readonly_mode
def api_session_runtime_status(sid):
    try:
        data = ctx.get_redis_client().get(f"SESSION_STATUS_{sid}")

        response = make_response(json.loads(data.decode()))

    except Exception as e:
        response = make_response(
            {'item': {'error': "{}".format(e), "traceback": traceback.format_exc()}})

    response.headers['Content-Type'] = 'application/json'
    return response

# -----------------------------
#         Data Streams
# -----------------------------

# TODO: Copy Pasta -- Fix me
@app.route('/api/stream/events')
def stream_events():
    logging.info("Connecting to server event stream")
    pub = ctx.get_redis_client().pubsub()
    pub.subscribe("PISTARLAB_EVENTS")
    pub.subscribe("PISTARLAB_LOGS_root:0")
    send_freq = 0.5
    last_sent = time.time()
    max_batch = 400

    def gen():
        data_batch = []
        while (True):
            msg = pub.get_message()
            if msg is None and len(data_batch) == 0:
                msg = next(pub.listen())
            if msg is not None:
                channel = msg['channel']
                data = msg['data']
                if data == 1 or data == 2:
                    continue
                data = json.loads(data.decode('utf-8'))
                if channel == b"PISTARLAB_LOGS":
                    data['type'] = "log"
                    data_batch.append(data)
            if len(data_batch) > 0 and (time.time() - last_sent) > send_freq:
                if len(data_batch) > max_batch:
                    data_batch = data_batch[-max_batch:]
                response_data = json.dumps(data_batch)
                data_batch = []

                yield f"data:{response_data}\n\n"

    return Response(gen(), mimetype='text/event-stream')


@app.route('/api/stream/scoped/<scope_name>')
def stream_scoped_logs(scope_name):
    logging.debug(f"Connecting to server event stream of {scope_name}")
    pub = ctx.get_redis_client().pubsub()
    channel_name = f"PISTARLAB_LOGS_{scope_name}"
    pub.subscribe(channel_name)
    send_freq = 0.5
    last_sent = time.time()
    max_batch = 400

    def gen():
        data_batch = []
        try:
            file_path = os.path.join(ctx.config.log_root, f"{scope_name}.txt")
            with open(file_path, 'r') as f:
                data_batch.extend(f.readlines())
        except Exception as e:
            logging.info(f"Fload to load {file_path} {e}")
            pass

        clear_old = True
        while (True):
            try:
                msg = pub.get_message()

                if len(data_batch) > 0 and (time.time() - last_sent) > send_freq:
                    if len(data_batch) > max_batch:
                        data_batch = data_batch[-max_batch:]
                    info = {
                        'entries': data_batch,
                        'clear_old': clear_old
                    }
                    response_data = json.dumps(info)
                    data_batch = []
                    clear_old = False
                    yield f"data:{response_data}\n\n"

                if msg is None and len(data_batch) == 0:
                    msg = next(pub.listen())
            except Exception as e:
                break
            if msg is not None:
                channel = msg['channel']
                data = msg['data']
                if data == 1 or data == 2:
                    continue
                if channel == channel_name.encode():
                    data = data.decode('utf-8')  # json.loads()
                    data_batch.append(data)
        return ""

    return Response(gen(), mimetype='text/event-stream')


@app.route('/api/stream/entity_logs/<entity>/<id>')
def stream_entity_logs(entity, id):
    logging.debug(f"Connecting to server event stream of {entity}__{id}")
    pub = ctx.get_redis_client().pubsub()
    channel_name = f"PISTARLAB_LOGS_{entity}_{id}"
    pub.subscribe(channel_name)
    send_freq = 0.5
    last_sent = time.time()
    max_batch = 400

    def gen():
        data_batch = []
        try:
            path = ctx.get_store().get_path_from_key((entity, id))
            for file in os.listdir(path):
                if file.startswith("log_"):
                    data_batch.extend(ctx.get_store().get(
                        key=(entity, id), name=file.replace(".txt", ""), stype="txt"))
        except:
            pass
        clear_old = True
        while (True):
            try:
                msg = pub.get_message()

                if len(data_batch) > 0 and (time.time() - last_sent) > send_freq:
                    if len(data_batch) > max_batch:
                        data_batch = data_batch[-max_batch:]
                    info = {
                        'entries': data_batch,
                        'clear_old': clear_old
                    }
                    response_data = json.dumps(info)
                    data_batch = []
                    clear_old = False
                    yield f"data:{response_data}\n\n"

                if msg is None and len(data_batch) == 0:
                    msg = next(pub.listen())
            except Exception as e:
                break
            if msg is not None:
                channel = msg['channel']
                data = msg['data']
                if data == 1 or data == 2:
                    continue
                if channel == channel_name.encode():
                    data = data.decode('utf-8')
                    data_batch.append(data)
        return ""

    return Response(gen(), mimetype='text/event-stream')

# -----------------------------------------
#               Agents
# -----------------------------------------
@app.route("/api/new_agent_submit", methods=['POST'])
def api_new_agent_submit():
    try:
        logging.info("Creating New Agent Instance")
        request_data = request.get_json()
        spec_id = request_data['specId']
        config = request_data['config']
        snapshot_id = request_data.get('snapshotId')
        if snapshot_id is None or snapshot_id == "":
            agent = Agent.create(spec_id=spec_id, config=config)
        else:
            logging.info(
                f"Creating New Agent Instance from snapshot id {snapshot_id}")
            agent = ctx.create_agent_from_snapshot(snapshot_id)

        response = make_response({'item': {'uid': agent.get_id()}})
    except Exception as e:
        response = make_response(
            {'item': {'error': "{}".format(e), "traceback": traceback.format_exc()}})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/agent/<action>/tag/<agent_id>/<tag>")
def api_agent_modify_tag(action, agent_id, tag):
    try:
        logging.info("Update Tags")
        logging.info("{}:{} for {}".format(action, tag, agent_id))
        if action == "add":
            ctx.add_agent_tag(agent_id, tag)
        elif action == "remove":
            ctx.remove_agent_tag(agent_id, tag)
        else:
            logging.info("NOT DOING ANYTHING:{}".format(action))
        response = make_response({'status': "ok"})

    except Exception as e:
        response = make_response(
            {'item': {'error': "{}".format(e), "traceback": traceback.format_exc()}})
    response.headers['Content-Type'] = 'application/json'
    return response

# -----------------------------------------
#                 Tasks
# -----------------------------------------
@app.route("/api/new_session_task_submit", methods=['POST'])
@app.route("/api/submit_agent_task", methods=['POST'])
@not_in_readonly_mode
def api_submit_agent_task():
    try:
        logging.info("Agent Task Submission Received")
        task_config = request.get_json()
        num_agents = len(task_config['agents'])
        if num_agents == 1:
            agent_data = task_config['agents'][0]
            agent = Agent.load(agent_data['ident'])
            agent_run_config = agent_data['run_config']
            session_config = agent_data['session_config']
            env_spec_id = task_config['env_spec_id']
            batch_size = task_config['batch_size']
            env_kwargs = task_config['env_kwargs']
            task = AgentTask.create(
                env_spec_id=env_spec_id,
                env_kwargs=env_kwargs,
                agent=agent,
                batch_size=batch_size,
                agent_run_config=agent_run_config,
                session_config=session_config)

        else:
            task = Task.create(spec_id="multiagent", config=task_config)

        task_id = task.get_id()
        logging.info("TaskID={}".format(task_id))
        if task_id is None:
            raise Exception("TODO IMPLEMENT ME")

        task.run()

        response = make_response({'item': {'uid': task_id}})
    except Exception as e:
        response = make_response(
            {'item': {'error': "{}".format(e), "traceback": traceback.format_exc()}})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/new_task_submit", methods=['POST'])
@not_in_readonly_mode
def api_new_task_submit():
    try:
        if ctx.config.read_only_mode:
            raise Exception("TASK SUBMISSION NOT ALLOWED IN readonly MODE")
        logging.info("Task Submission Received")
        spec_id = request.get_json()['specId']
        task_config = request.get_json()['config']
        if spec_id == 'agent_task':
            agent = Agent.load(task_config['agent_id'])
            agent_run_config = task_config['agent_run_config']
            session_config = task_config['session_config']
            env_spec_id = task_config['env_spec_id']
            env_kwargs = task_config['env_kwargs']
            task = AgentTask.create(
                env_spec_id=env_spec_id,
                env_kwargs=env_kwargs,
                agent=agent,
                agent_run_config=agent_run_config,
                session_config=session_config)

        else:
            task = Task.create(spec_id=spec_id, config=task_config)

        task_id = task.get_id()
        logging.info("TaskID={}".format(task_id))
        if task_id is None:
            raise Exception("TODO IMPLEMENT ME")

        task.run()

        response = make_response({'item': {'uid': task_id}})
    except Exception as e:
        response = make_response(
            {'item': {'error': "{}".format(e), "traceback": traceback.format_exc()}})
    response.headers['Content-Type'] = 'application/json'
    return response

# ---------------------------------------
#              Plot Data
# ---------------------------------------


def chunk_list_simple(data, chunk_max=10):
    """
    if more than one value in range for a given chunk, returns both max and min
    TODO: consider making other options available, eg: mean, mean + range, mean + variance
    """
    data_size = len(data)
    chunk_size = int(max(data_size / chunk_max, 1))
    chunk_num = min(chunk_max, len(data))
    result_data = []
    result_data.append(data[0])
    for chunk in range(chunk_num):
        start_idx = chunk * chunk_size
        end_idx = min(chunk * chunk_size + chunk_size, data_size)
        vals = [data[i][1] for i in range(start_idx, end_idx)]
        idx = data[end_idx - 1][0]
        last_item_min = (idx, min(vals))
        last_item_max = (idx, max(vals))
        if last_item_max != last_item_min:
            result_data.append(last_item_min)
            result_data.append(last_item_max)
        else:
            result_data.append(last_item_max)

    return result_data


def select_bin_size(count, bins=[(50, 10), (200, 20), (500, 50), (1000, 100), (10000, 1000), (50000, 2000), (100000, 10000), (1000000, 50000), (2000000, 100000)]):
    bin_size = 1
    for thresh, bin_size_candidate in bins:
        if count > thresh:
            bin_size = bin_size_candidate
        else:
            return bin_size

    return bin_size


def bin_data(data, bin_size=None, start_idx_offset=0):
    data_size = len(data)
    if bin_size is None:
        bin_size = select_bin_size(data_size)

    parts = int(data_size / bin_size)
    result_data = []
    include_stats = False
    if bin_size == 1:
        for i in range(data_size):
            val = data[i]
            result_data.append((i, val[1], val[0]))
    else:
        include_stats = True
        for chunk in range(parts):
            start_idx = chunk * bin_size
            end_idx = min(chunk * bin_size + bin_size, data_size)
            vals = [data[i][1] for i in range(start_idx, end_idx)]
            xlabels = [data[i][0] for i in range(start_idx, end_idx)]
            xlabel = max(xlabels)
            num_vals = len(vals)
            vmin = min(vals)
            vmax = max(vals)
            vmean = sum(vals) / num_vals
            std_dev = math.sqrt(sum((x - vmean)**2 for x in vals) / num_vals)
            vlower = vmean - std_dev
            vupper = vmean + std_dev
            result_data.append((start_idx+start_idx_offset,
                                vmean, vmin, vmax, vlower, vupper, xlabel))

    return result_data, include_stats


def slice_dict_head(data, idx):
    ndata = {}
    for k, items in data.items():
        ndata[k] = items[idx:]
    return ndata


@app.route("/api/session_plots_json/<sid>/<data_group>/<data_name>/<step_field>")
def api_session_plots_json(sid, data_group, data_name, step_field):
    bin_size = int(request.args.get('bin_size', "0"))
    max_steps = int(request.args.get('max_steps', "0"))
    total_count = 0
    actual_count = 0
    try:

        orig_data = ctx.get_store().get_session_data(session_id=sid, name=data_group)

        if orig_data is None:
            raise Exception(f"Session {sid}, {data_group} data not found.")

        step_counts = orig_data[step_field]
        total_count = len(step_counts)
        actual_count = total_count

        if max_steps > 0:
            max_steps = int(max_steps)
            if total_count > max_steps:
                orig_data = slice_dict_head(orig_data, total_count-max_steps)
                actual_count = len(step_counts)
                step_counts = orig_data[step_field]

        data = orig_data[data_name]
        if bin_size > 0:
            bin_size = int(bin_size)
        else:
            bin_size = select_bin_size(actual_count)

        values, include_stats = bin_data(
            [(step, value) for step, value in zip(step_counts, data)],
            bin_size=bin_size)

        logging.debug("total_data {}, final_values= {}".format(
            len(step_counts), len(values)))
        graph = {}
        graph['include_stats'] = include_stats
        if include_stats:
            idxs, means, mins, maxs, lowers, uppers, xlabels = map(
                list, zip(*values))
            graph['data'] = dict(
                idxs=idxs, means=means, mins=mins, maxs=maxs, lowers=lowers, uppers=uppers, xlabels=xlabels)
        else:
            idxs, vals, xlabels = map(list, zip(*values))
            graph['data'] = dict(idxs=idxs, vals=vals, xlabels=xlabels)

        graphJSON = json.dumps(graph)
        response = make_response(graphJSON)

    except Exception as e:
        response = make_response(
            {'error': '{}'.format(e), 'traceback': traceback.format_exc()})
        logging.error(f"Error: {e} 'traceback': {traceback.format_exc()}" )

    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/session_plotly_json/<sid>/<data_group>/<data_name>/<step_field>")
def api_session_plotly_json(sid, data_group, data_name, step_field):
    try:
        orig_data = ctx.get_store().get_session_data(session_id=sid, name=data_group)
        step_counts = orig_data[step_field]
        data = orig_data[data_name]
        values, include_stats = bin_data(
            [(step, value) for step, value in zip(step_counts, data)])
        logging.debug("total_data {}, final_values= {}".format(
            len(orig_data), len(values)))
        graph = {}
        if include_stats:
            idxs, means, mins, maxs, lowers, uppers, xlabels = map(
                list, zip(*values))
            graph['data'] = dict(x=idxs, y=means, name=sid)
        else:
            idxs, vals, xlabels = map(list, zip(*values))
            graph['data'] = dict(x=idxs, y=vals, name=sid)
        graph['layout'] = dict(title="{} - {}".format(data_group, data_name))
        graphJSON = json.dumps(graph)
        response = make_response(graphJSON)

    except Exception as e:
        response = make_response(
            {'error': '{}'.format(e), 'traceback': traceback.format_exc()})
        logging.error(e)

    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/agent_plots_json/<uid>")
def api_agent_plots_json(uid):
    bin_size = int(request.args.get('bin_size', "0"))
    max_steps = int(request.args.get('max_steps', "0"))
    total_count = 0
    actual_count = 0
    allplotdata = {}

    try:

        orig_data = ctx.get_store().get_multipart_dict(key=('agent', uid), name='stats')
        if orig_data is None:
            raise Exception(f"No stats data found for agent {uid}")
        orig_data.pop("model", "NA")  # TODO: fix at source

        total_count = len(orig_data['timestamp'])
        actual_count = total_count
        step_counts = range(total_count)
        start_idx_offset = 0

        if max_steps > 0:
            max_steps = int(max_steps)
            if total_count > max_steps:
                start_idx_offset = total_count-max_steps
                orig_data = slice_dict_head(orig_data, total_count-max_steps)
                actual_count = len(orig_data['timestamp'])
                step_counts = range(actual_count)

        cols = [col for col in orig_data.keys() if col not in [
            'timestamp', 'learn_step', 'task_id']]

        color_map = cm.get_cmap('tab20', 20)
        if bin_size > 0:
            bin_size = int(bin_size)
        else:
            bin_size = select_bin_size(actual_count)

        for i, col in enumerate(cols):
            data = orig_data[col]
            try:
                values, include_stats = bin_data(
                    [(step, value) for step, value in zip(step_counts, data)], bin_size=bin_size, start_idx_offset=start_idx_offset)
                logging.debug("total_data {}, final_values= {}".format(
                    len(orig_data), len(values)))
                graph = {}
                color_code = [int(v * 255) for v in color_map(i % 20)[0:3]]
                color = "rgb({},{},{})".format(
                    color_code[0], color_code[1], color_code[2])

                if include_stats:
                    idxs, means, mins, maxs, lowers, uppers, xlabels = map(
                        list, zip(*values))
                    graph['data'] = dict(
                        x=idxs, y=means, name=uid, line={"color": color})
                else:
                    idxs, vals, xlabels = map(list, zip(*values))
                    graph['data'] = dict(
                        x=idxs, y=vals, name=uid, line={"color": color})

                graph['layout'] = dict(title=col)
                allplotdata[col] = graph
            except Exception as e:
                logging.error(e)

                pass
        response_data = {
            'plot_data': allplotdata,
            'total_count': total_count,
            'actual_count': actual_count,
        }  # , cls=plotly.utils.PlotlyJSONEncoder)

        response = make_response(response_data)
    except Exception as e:
        logging.error('msg: {}, traceback: {}'.format(
            e, traceback.format_exc()))
        response = make_response(
            {'error': '{}'.format(e), 'traceback': traceback.format_exc()})

    response.headers['Content-Type'] = 'application/json'
    return response


# ---------------------------------------
# --------- Episode Info/Video Data------
# ---------------------------------------
def get_chunk(full_path, byte1=None, byte2=None):

    file_size = os.stat(full_path).st_size
    start = 0
    length = 102400

    if byte1 < file_size:
        start = byte1
    if byte2:
        length = byte2 + 1 - byte1
    else:
        length = file_size - start

    with open(full_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(length)
    return chunk, start, length, file_size


@app.route("/api/session_episode_mp4/<sid>/<eid>")
def session_episode_mp4(sid, eid):

    try:
        default_fps = ctx.get_session(sid).env_spec.meta.get('render_fps', 30)
    except:
        default_fps = 30
    video_filename = os.path.join(
        ctx.get_store().root_path, 'session', sid, 'episode', eid, "{}.mp4".format(eid))
    # TODO: Fix this type nonsense
    fps = int(request.args.get('fps', str(default_fps)))

    # refresh = bool(request.args.get('refresh', "True"))

    # if not refresh and os.path.exists(video_filename):
    #     return send_file(video_filename, mimetype="video/mp4")
    # os.remove(video_filename)

    image_path = os.path.join(
        ctx.get_store().root_path, 'session', sid, 'episode', eid, 'images')
    try:
        import ffmpeg
        (
            ffmpeg
            .input('{}/*.jpg'.format(image_path), pattern_type='glob', framerate=fps)
            .output(video_filename)
            .overwrite_output()
            .run()
        )

        return send_file(video_filename, mimetype="video/mp4")
    except Exception as e:
        response = make_response("{}".format(e))
        logging.error(e)
        response.headers['Content-Type'] = 'application/json'
        return response


def tryparse(x, type_, default):
    try:
        return type_(x)
    except Exception as e:
        return default


@app.route("/api/session_episodes/<sid>/")
def api_session_episodes(sid):

    _, episodes = ctx.get_store().list(('session', sid, 'episode'))
    episodes = sorted(episodes, key=lambda x: tryparse(
        x, int, 0), reverse=True)
    max_recorded_ep = None

    if len(episodes) > 0:
        max_recorded_ep = episodes[0]

    data = {'items': episodes, 'max_recorded_ep': max_recorded_ep,
            'total_recorded': len(episodes)}
    response = make_response(data)
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/session_episode_by_id/<sid>/<episode_id>")
def api_session_episode_by_id(sid, episode_id):
    episode = ctx.get_store().get_multipart_dict(
        ('session', sid, 'episode', episode_id), name='recording')
    data = {'item': episode}
    response = make_response(data)
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/session_max_episode_recorded/<sid>/")
def api_session_max_episode_recorded(sid):

    _, episodes = ctx.get_store().list(('session', sid, 'episode'))
    if episodes is None or len(episodes) == 0:
        response = make_response({'message': "Error"})
        response.headers['Content-Type'] = 'application/json'
        return response
    episodes = sorted(episodes, key=lambda x: tryparse(
        x, int, 0), reverse=True)
    max_recorded_ep = None

    if len(episodes) > 0:
        max_recorded_ep = episodes[0]

    data = {'max_recorded_ep': max_recorded_ep,
            'total_recorded': len(episodes)}
    response = make_response(data)
    response.headers['Content-Type'] = 'application/json'
    return response


# ---------------------------------------
# --------- DATA BROWSER-----------------
# ---------------------------------------
# ORIG SOURCE: https://codereview.stackexchange.com/questions/214418/simple-web-based-file-browser-with-flasks
def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


@app.route('/api/browser')
@app.route('/api/browser/')
@app.route('/api/browser/<path:urlFilePath>')
def api_browser(urlFilePath=""):
    # TODO: this method is definitely a security risk
    try:
        if urlFilePath is None or urlFilePath == "/":
            urlFilePath = ""

        fullFilePath = os.path.join(ctx.get_store().root_path, urlFilePath)
        logging.info("FullPath:{}".format(fullFilePath))
        if os.path.isdir(fullFilePath):
            itemList = os.listdir(fullFilePath)
            itemList = sorted(itemList, key=lambda x: x)

            itemListResults = []
            for item in itemList:
                item_data = {
                    'name': str(item),
                    'is_dir': os.path.isdir(os.path.join(fullFilePath, item)) or item == "/",
                    'date': time.ctime(os.path.getmtime(os.path.join(fullFilePath, item))),
                    'size': sizeof_fmt(os.stat(os.path.join(fullFilePath, item)).st_size)}
                itemListResults.append(item_data)

            if len(urlFilePath) > 0:
                urlFilePath = urlFilePath + "/"
            response = make_response(
                {'urlFilePath': urlFilePath, 'itemList': itemListResults})
            response.headers['Content-Type'] = 'application/json'
            return response
        elif os.path.isfile(fullFilePath):
            downloadURL = "/api/download/{}".format(urlFilePath)
            response = make_response({'downloadURL': downloadURL})
            response.headers['Content-Type'] = 'application/json'
            return response
    except Exception as e:
        response = make_response(
            {'error': "something bad happened", "exception": e})
        response.headers['Content-Type'] = 'application/json'
        return response

    response = make_response({'error': "invalid input {}".format(urlFilePath)})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route('/api/download/<path:urlFilePath>')
def api_download(urlFilePath=""):
    try:

        fullFilePath = os.path.join(ctx.get_store().root_path, urlFilePath)
        logging.info("FullPath:{}".format(fullFilePath))
        if os.path.isfile(fullFilePath):
            if fullFilePath.endswith(".json"):
                mimetype = "application/json"
            elif fullFilePath.endswith(".txt"):
                mimetype = "text/plain"
            else:
                mimetype = "text/{}".format(fullFilePath.split('.')[1])

            return send_file(fullFilePath, mimetype=mimetype)
    except Exception as e:
        response = make_response(
            {'error': "something bad happened", "exception": e})
        response.headers['Content-Type'] = 'application/json'
        return response
    response = make_response({'error': "invalid input {}".format(urlFilePath)})
    response.headers['Content-Type'] = 'application/json'
    return response


# -------------------------------
# Serve Static Files
# -------------------------------
@app.route("/api/env_preview_image/<image_name>")
def api_env_preview_image(image_name):
    path_root = ctx.get_store().get_path_from_key((SYS_CONFIG_DIR, 'envs'))
    filename = os.path.join(path_root, "images", "{}.jpg".format(image_name))
    try:
        return send_file(filename, mimetype="image/{}".format('jpg'))
    except Exception as e:
        response = make_response({'msg': "FILE NOT FOUND"})
        response.headers['Content-Type'] = 'application/json'
        return response


@app.route("/api/config")
def api_config():
    response = make_response(
        {  # "env": dict(os.environ),
            "sys_config": ctx.config.__dict__})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route('/')
def index():
    return send_file(os.path.join('uidist', 'index.html'))


@app.route('/<path:path>')
def servce_static(path):
    return send_from_directory('uidist', path)


@app.route("/api/prep_notebook/")
def api_prep_notebook():
    """Create a notebook for a given task"""
    # TODO:
    # - get notebook template
    # - replace tokens
    # - move to location
    # pkg_resources.
    # PISTARLAB_TASK_ID__
    pass


@app.after_request
def after_request(response):
    response.headers.add('Accept-Ranges', 'bytes')
    return response


def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--autoreload", action="store_true",
                        help="auto reload changes")
    parser.add_argument("--port", help="port", default=7777)
    parser.add_argument("--host", help="host", default="0.0.0.0")
    parser.add_argument("--enable_profiler",
                        action="store_true", help="Enable profiler")

    args = parser.parse_args()
    if args.enable_profiler:
        profiler = Profiler()
        profiler.start()

    profiler = None

    def graceful_exit(signum=None, frame=None):
        print("Shutting Down Services: signum:{}, frame: {}".format(signum, frame))
        ctx.close()
        if args.enable_profiler:
            profiler.stop()
            print(profiler.output_text(unicode=True,
                                       color=True, show_all=True, timeline=True))
        print("Backend Shutdown Complete")
        sys.exit()

    ctx.initialize()
    signal.signal(signal.SIGINT, graceful_exit)
    app.run(host=args.host, port=args.port,
            debug=args.debug, use_reloader=args.autoreload)


if __name__ == "__main__":
    main()
