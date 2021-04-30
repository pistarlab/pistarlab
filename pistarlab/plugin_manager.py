import glob
import importlib
import json
import logging
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from string import Template

import pkg_resources
import setuptools
from filelock import FileLock

from .utils.pkghelpers import run_bash_command


def open_json_file(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None


def save_to_json_file(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_plugin_key(plugin_id, plugin_version):
    return f'{plugin_id}__v{plugin_version}'


def get_plugin_key_from_plugin(plugin):
    return get_plugin_key(plugin['id'], plugin['version'])


def get_home():
    return str(Path.home())


def get_default_config_dir(proj_module_name):
    home = str(Path.home())
    return os.path.join(home, ".{}".format(proj_module_name))


def add_workspace_pkgs_to_path(path):
    os.makedirs(path, exist_ok=True)
    directories = glob.glob(path)
    paths = set(sys.path)
    for dir in directories:
        if dir not in paths:
            sys.path.append(dir)


def load_plugins_from_path(path):
    syspaths = set(sys.path)
    plugins = []
    for proj_dir in os.listdir(path):
        full_proj_dir = os.path.join(path, proj_dir)
        if not full_proj_dir in syspaths:
            sys.path.append(full_proj_dir)
        for pkg_name in setuptools.find_packages(where=full_proj_dir):
            try:
                pluginmod = importlib.import_module(name="{}.plugin".format(pkg_name))
                runnable = all([hasattr(pluginmod, funcname) for funcname in ['install', 'load', 'uninstall']])
                if runnable:
                    # workspace_pkgs.append({'path': full_proj_dir, 'name': pkg_name})
                    plugin_info_path = pkg_resources.resource_filename(pkg_name, "pistarlab_plugin.json")
                    with open(plugin_info_path, 'r') as f:
                        plugin_info = json.load(f)
                    plugins.append(plugin_info)
                else:
                    logging.info("Module missing function for {}".format(pkg_name))
            except:
                pass
    return plugins


def create_file_from_template(template_path, target_path, var_dict):
    with open(template_path, 'r') as f:
        template = Template(f.read())
    result_text = template.substitute(var_dict)
    with open(target_path, "w") as f:
        f.write(result_text)


def create_new_plugin(workspace_path, author, plugin_id, plugin_name, description="", author_email="", version="0.0.1-dev"):
    plugin_id = plugin_id.replace(" ", "-").replace("_", "-").lower()
    module_name = plugin_id.replace(" ", "_").replace("-", "_").lower()
    plugin_path = os.path.join(workspace_path, plugin_id)
    module_path = os.path.join(workspace_path, plugin_id, module_name)
    os.makedirs(module_path, exist_ok=True)
    plugin_info = {
        "id": plugin_id,
        "version": version,
        "name": plugin_name,
        "module_name": module_name,
        "categories": [],
        "description": description,
        "author": author,
        "author_email": author_email
    }

    with open(os.path.join(module_path, "pistarlab_plugin.json"), "w") as f:
        json.dump(plugin_info, f, indent=2)

    Path(os.path.join(module_path, "__init__.py")).touch()

    setup_template_path = pkg_resources.resource_filename('pistarlab', "templates/setup_py.txt")
    setup_target_path = os.path.join(plugin_path, "setup.py")
    create_file_from_template(setup_template_path, setup_target_path, plugin_info)

    plugin_template_path = pkg_resources.resource_filename('pistarlab', "templates/plugin_py.txt")
    plugin_target_path = os.path.join(module_path, "plugin.py")
    create_file_from_template(plugin_template_path, plugin_target_path, plugin_info)
    add_workspace_pkgs_to_path(workspace_path)


SOURCE_FILE_NAME = "plugin_sources.json"


class PluginManager:
    """
    Naming Requirements:
     - plugin package name = plugin_id
     - plugin plugin name = plugin_id with hyphans replaced with underscores
     - builtin plugin repo is under pistarlab/plugins
    """

    def __init__(self, proj_module_name, workspace_path, data_path, logger=None):

        self.proj_module_name = proj_module_name
        self.workspace_path = workspace_path
        self.data_path = data_path
        self.plugin_path = os.path.join(data_path, "plugins")
        self.logger = logger or logging

        os.makedirs(self.plugin_path, exist_ok=True)

    def get_sources(self):

        sources = {}
        sources["builtin"] = {
            "id": "builtin",
            "type": "file",
            "name": "Built-in",
            "description": "",
            "path": self.get_builtin_plugin_src_path(),
        }
        sources["workspace"] = {
            "id": "workspace",
            "type": "workspace",
            "name": "Workspace",
            "description": "",
            "path": self.workspace_path
        }

        extended_sources = open_json_file(os.path.join(self.data_path, SOURCE_FILE_NAME))
        if extended_sources is not None:
            sources.update(extended_sources)
        else:
            Path(os.path.join(self.data_path, SOURCE_FILE_NAME)).touch()
        return sources

    def plugin_id_to_module_name(self, plugin_id):
        return plugin_id.replace("-", "_")

    def get_plugins_from_sources(self):
        all_p = {}
        repo_filename = "repo.json"
        for source in self.get_sources().values():
            plugins = None
            if source["type"] == "url":
                repo_path = "{}/{}".format(source["path"], repo_filename)
                with urllib.request.urlopen(repo_path) as url:
                    plugins = json.loads(url.read().decode())
            elif source["type"] == "file":
                repo_path = os.path.join(source["path"], repo_filename)
                plugins = open_json_file(repo_path)
            elif source["type"] == "workspace":
                plugins = load_plugins_from_path(source["path"])
            if plugins is not None:
                for plugin in plugins:
                    plugin["source"] = source
                    plugin_key = get_plugin_key_from_plugin(plugin)
                    all_p[plugin_key] = plugin
                    # pull in metadata from files
                    if plugin.get('metafile', False):
                        meta_filename = "{}.json".format(plugin_key)
                        if source["type"] == "url":
                            meta_path = "{}/{}".format(source["path"], meta_filename)
                            with urllib.request.urlopen(meta_path) as url:
                                plugin["metadata"] = json.loads(url.read().decode())
                        elif source["type"] == "file":
                            plugin["metadata"] = open_json_file(os.path.join(source["path"], meta_filename))

        for plugin in all_p.values():
            plugin['status'] = "AVAILABLE"
        return all_p

    def get_installed_plugins(self):
        installed_p = open_json_file(
            os.path.join(self.plugin_path, "installed.json")
        )
        if installed_p is None:
            installed_p = {}

        return installed_p

    def get_all_plugins(self):
        plugins = self.get_plugins_from_sources()
        installed_p = self.get_installed_plugins()
        for plugin in installed_p.values():
            plugins[get_plugin_key_from_plugin(plugin)] = plugin
        return plugins

    def get_builtin_plugin_src_path(self):
        return pkg_resources.resource_filename(__name__, "plugins")

    def _run_module_function(self, module_name, function_name=None, kwargs={}):
        pluginmod = importlib.import_module(name="{}.plugin".format(module_name))
        return getattr(pluginmod, function_name)()

    def _run_plugin_package_install(self, plugin):
        cmd = None
        if plugin["source"]["type"] == "file":
            full_path = os.path.join(self.get_builtin_plugin_src_path(), plugin["id"])
            cmd = "pip install --user -e {}".format(full_path)
        elif plugin["source"]["type"] == "workspace":
            full_path = os.path.join(self.workspace_path, plugin["id"])
            cmd = "pip install --user -e {}".format(full_path)
        elif plugin["source"]["type"] == "remote":
            remote_path = plugin.get("remote_path")
            if remote_path is None:
                raise Exception("No remote path defined in plugin. Cannot install with out path.")
            cmd = "pip install --user {}".format(remote_path)
        else:
            cmd = "pip install --user {}=={}".format(plugin["id"], plugin["version"])

        self.logger.info("Install Command")

        # TODO: Should remove below: so far unsuccessful in loading modules installed for first time using pip during runtime
        # https://stackoverflow.com/questions/32478724/cannot-import-dynamically-installed-python-module
        import site
        cmd_result = run_bash_command(cmd)
        if not os.path.exists(site.USER_SITE):
            os.makedirs(site.USER_SITE)
        sys.path.insert(0, site.USER_SITE)
        importlib.invalidate_caches()
        return cmd_result

    def _run_plugin_package_remove(self, plugin):
        cmd = "pip uninstall -y {}".format(plugin["id"])
        self.logger.info("-------------------------------------------")
        self.logger.info(cmd)
        return run_bash_command(cmd)

    def finish_installing_new_plugins(self):
        self.logger.debug("Loading New Plugins")
        lock = FileLock(os.path.join(self.plugin_path, ".loading.lock"))
        with lock:
            for plugin in self.get_installed_plugins().values():
                if plugin['status'] == 'PREPPED_RELOAD':
                    self.logger.info("Finishing Install of {}".format(plugin['id']))
                    self.install_plugin(plugin_id=plugin['id'], plugin_version=plugin['version'])

    def load_plugins(self):
        self.logger.debug("Loading Plugins")
        lock = FileLock(os.path.join(self.plugin_path, ".loading.lock"))
        with lock:
            for plugin in self.get_installed_plugins().values():
                if plugin['status'] == "INSTALLED":
                    self.logger.debug("Loading {}".format(plugin['id']))
                    module_name = plugin.get('module_name', plugin['id'].replace('-', "_"))
                    self._run_module_function(module_name, "load", kwargs=plugin.get("load_kwargs", {}))

    def update_plugin_status(self, plugin, state, msg=""):
        lock = FileLock(os.path.join(self.plugin_path, ".updating.lock"))
        old_status = "NA"
        with lock:
            installed_p = self.get_installed_plugins()
            installed_p[plugin['id']] = plugin
            old_status = plugin.get('status')

            plugin['status'] = state
            plugin['status_msg'] = msg
            plugin['status_timestamp'] = time.time()

            save_to_json_file(
                installed_p, os.path.join(self.plugin_path, "installed.json")
            )
            self.logger.info("Updated {} status: from {} to {}".format(plugin['id'], old_status, plugin['status']))

    def cleanup(self):
        for plugin in self.get_installed_plugins().values():
            if plugin['status'] in ["INSTALLING", "UNINSTALLING"]:
                self.update_plugin_status(plugin, "INSTALL_FAILED")

    def remove_plugin_by_id(self, plugin_id):
        lock = FileLock(os.path.join(self.plugin_path, ".updating.lock"))
        with lock:
            installed_p = self.get_installed_plugins()
            installed_p.pop(plugin_id)
            save_to_json_file(
                installed_p, os.path.join(self.plugin_path, "installed.json")
            )

    def get_plugins_by_id(self, plugin_id):
        results = []
        for plugin in self.get_all_plugins().values():
            if plugin['id'] == plugin_id:
                results.append(plugin)
        return results

    def get_plugin(self, plugin_id, plugin_version):
        plugin_key = get_plugin_key(plugin_id, plugin_version)
        return self.get_all_plugins().get(plugin_key)

    def install_plugin(self, plugin_id, plugin_version):

        plugin_key = get_plugin_key(plugin_id, plugin_version)

        plugin = self.get_all_plugins().get(plugin_key)
        plugin_id = plugin['id']

        if plugin is None:
            raise Exception("Plugin {} not found.".format(plugin_key))

        elif plugin['status'] in ["INSTALLING", "UNINSTALLING"]:
            raise Exception("Cannot perform action. Plugin {} is currently {}.".format(plugin_key, plugin['status']))

        self.update_plugin_status(plugin, "INSTALLING")

        # install package
        try:
            self.logger.info("Installing {}".format(plugin_key))
            result = self._run_plugin_package_install(plugin)
            self.logger.info("Package Install Output\n {}".format(result))
        except Exception as e:
            msg = "{}\n{}".format(e, traceback.format_exc())
            self.logger.error(msg)
            self.update_plugin_status(plugin, "INSTALL_FAILED", msg=msg)
            return False

        # try to import, if failed, set state to RELOAD_REQUIRED
        try:
            importlib.invalidate_caches()
            importlib.import_module(self.plugin_id_to_module_name(plugin_id))
        except Exception as e:
            msg = "{}\n{}".format(e, traceback.format_exc())
            self.logger.error(msg)
            self.update_plugin_status(plugin, "PREPPED_RELOAD", msg=msg)
            return True

        # Run plugin install
        try:
            module_name = plugin.get('module_name', plugin_id.replace('-', "_"))
            result = self._run_module_function(module_name, "install")
            self.logger.info("Plugin Install Output\n {}".format(result))
            self.update_plugin_status(plugin, "INSTALLED")
            # self.update_plugin_packages_file()
            return True
        except Exception as e:
            msg = "{}\n{}".format(e, traceback.format_exc())
            self.logger.error(msg)
            self.update_plugin_status(plugin, "INSTALL_FAILED", msg=msg)
            return False

    # def update_plugin_packages_file(self):
    #     with open(os.path.join(self.plugin_path, "plugin_packages.txt"), "w") as f:
    #         for plugin_id in self.get_installed_plugins().keys():
    #             f.write(f"${plugin_id}\n")

    def reload_plugin_by_id(self, plugin_id):
        module_name = self.plugin_id_to_module_name(plugin_id)
        module = importlib.import_module(module_name)
        importlib.reload(module)
        return True

    def uninstall_plugin(self, plugin_id):
        self.logger.debug("Uninstalling {}".format(plugin_id))
        installed_p = self.get_installed_plugins()
        plugin = installed_p.get(plugin_id)
        if plugin['status'] in ["INSTALLING", "UNINSTALLING"]:
            raise Exception("Cannot perform action. Plugin {} is currently {}.".format(plugin_id, plugin['status']))

        # Mark as INSTALLING
        self.update_plugin_status(plugin, "UNINSTALLING")

        try:
            # Run Uninstallion
            module_name = plugin.get('module_name', plugin_id.replace('-', "_"))
            result = self._run_module_function(module_name, "uninstall")
            self.logger.debug("Plugin Uninstall Output\n {}".format(result))
            remove_result = self._run_plugin_package_remove(plugin)
            self.logger.debug("Package Uninstall Output\n {}".format(remove_result))

            # Remove Plugin Entry
            self.remove_plugin_by_id(plugin_id)
            return True

        except Exception as e:
            self.logger.error(e)
            self.update_plugin_status(plugin, "UNINSTALL_FAILED", msg="{}\n{}".format(e, traceback.format_exc()))
            return False
