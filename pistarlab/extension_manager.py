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
import pistarlab


def open_json_file(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None


def save_to_json_file(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_extension_key(extension_id, extension_version):
    return f'{extension_id}__v{extension_version}'


def get_extension_key_from_extension(extension):
    return get_extension_key(extension['id'], extension['version'])


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


def load_extensions_from_path(path):
    logging.info(f"Reading from{path}")
    syspaths = set(sys.path)
    extensions = []
    for proj_dir in os.listdir(path):
        full_proj_dir = os.path.join(path, proj_dir)
        if not full_proj_dir in syspaths:
            sys.path.append(full_proj_dir)
        for pkg_name in setuptools.find_packages(where=full_proj_dir):
            try:
                extensionmod = importlib.import_module(name="{}.extension".format(pkg_name))
                extension_info_path = pkg_resources.resource_filename(pkg_name, "pistarlab_extension.json")
                with open(extension_info_path, 'r') as f:
                    extension_info = json.load(f)
                extensions.append(extension_info)
            except Exception as e:
                logging.error("Error loading extension {} {}".format(pkg_name,e))
    return extensions


def create_file_from_template(template_path, target_path, var_dict):
    with open(template_path, 'r') as f:
        template = Template(f.read())
    result_text = template.substitute(var_dict)
    with open(target_path, "w") as f:
        f.write(result_text)


def create_new_extension(
        workspace_path, 
        extension_id, 
        extension_name, 
        original_author="", 
        extension_author="", 
        description="", 
        version="0.0.1-dev"):

    extension_id = extension_id.replace(" ", "-").replace("_", "-").lower()
    module_name = extension_id.replace(" ", "_").replace("-", "_").lower()
    extension_path = os.path.join(workspace_path, extension_id)
    module_path = os.path.join(workspace_path, extension_id, module_name)
    os.makedirs(module_path, exist_ok=True)
    extension_info = {
        "id": extension_id,
        "version": version,
        "name": extension_name,
        "categories": [],
        "description": description,
        "extension_author": extension_author,
        "original_author": original_author
    }

    template_vars = {}
    template_vars.update(extension_info)
    template_vars['module_name'] = module_name

    with open(os.path.join(module_path, "pistarlab_extension.json"), "w") as f:
        json.dump(extension_info, f, indent=2)

    Path(os.path.join(module_path, "__init__.py")).touch()

    setup_template_path = pkg_resources.resource_filename('pistarlab', "templates/setup_py.txt")
    setup_target_path = os.path.join(extension_path, "setup.py")
    extension_template_path = pkg_resources.resource_filename('pistarlab', "templates/extension_py.txt")
    extension_target_path = os.path.join(module_path, "extension.py")
    try:
        create_file_from_template(setup_template_path, setup_target_path, template_vars)
        create_file_from_template(extension_template_path, extension_target_path, template_vars)
    except Exception as e:
        logging.info(f"Error building templates, Variables: {template_vars}, Error: {e}")
        raise e
    add_workspace_pkgs_to_path(workspace_path)


SOURCE_FILE_NAME = "extension_sources.json"


class ExtensionManager:
    """
    Naming Requirements:
     - extension package name = extension_id
     - extension module name = extension_id with hyphans replaced with underscores
    """

    def __init__(self, proj_module_name, workspace_path, data_path, logger=None):

        self.proj_module_name = proj_module_name
        self.workspace_path = workspace_path
        self.data_path = data_path
        self.extension_path = os.path.join(data_path, "extensions")
        self.logger = logger or logging

        os.makedirs(self.extension_path, exist_ok=True)

    def get_sources(self):

        sources = {}
        sources["builtin"] = {
            "id": "builtin",
            "type": "file",
            "name": "Built-in",
            "description": "",
            "path": self.get_builtin_extension_src_path(),
        }
        sources["workspace"] = {
            "id": "workspace",
            "type": "workspace",
            "name": "Workspace",
            "description": "",
            "path": self.workspace_path
        }
        sources["main"] = {
            "id": "main",
            "type": "remote",
            "name": "Main Repo",
            "description": "Primary Remote Repo",
            "path": "https://raw.githubusercontent.com/pistarlab/pistarlab-repo/main/extensions/"
            # "path":"https://github.com/pistarlab/pistarlab-repo/raw/main/extensions/"
        }

        extended_sources = open_json_file(os.path.join(self.data_path, SOURCE_FILE_NAME))
        if extended_sources is not None:
            for source in extended_sources:
                self.logger.info(f"Loading Additional Extension Source {source['id']}")
                sources[source['id']] = source

        else:
            Path(os.path.join(self.data_path, SOURCE_FILE_NAME)).touch()
        return sources

    def extension_id_to_module_name(self, extension_id):
        return extension_id.replace("-", "_")

    def get_extensions_from_sources(self):
        all_p = {}
        
        for source in self.get_sources().values():
            extensions = None
            if source["type"] == "remote":
                repo_filename = f"{pistarlab.__version__}.json"
                repo_path = "{}{}".format(source["path"], repo_filename)
                try:
                    with urllib.request.urlopen(repo_path) as url:
                        extensions = json.loads(url.read().decode())
                except Exception as e:
                    self.logger.error(f"Failed to download remote source {repo_path}, {e}")
            elif source["type"] == "test":
                # For testing a repo locally
                repo_filename = f"{pistarlab.__version__}.json"
                repo_path = os.path.join(source["path"], repo_filename)
                extensions = open_json_file(repo_path)
            elif source["type"] == "file":
                repo_filename = "repo.json"
                repo_path = os.path.join(source["path"], repo_filename)
                extensions = open_json_file(repo_path)
            elif source["type"] in ["workspace","path"]:               
                extensions = load_extensions_from_path(source["path"])
                for extension in extensions:
                    extension['full_path'] = os.path.join(source['path'],extension['id'])
            if extensions is not None:
                for extension in extensions:
                    extension["source"] = source
                    extension_key = get_extension_key_from_extension(extension)
                    all_p[extension_key] = extension
                    # TODO: below not in use and not correct for URL
                    # pull in metadata from files
                    if extension.get('metafile', False):
                        meta_filename = "{}.json".format(extension_key)
                        if source["type"] == "remote":
                            meta_path = "{}{}".format(source["path"], meta_filename)
                            with urllib.request.urlopen(meta_path) as url:
                                extension["metadata"] = json.loads(url.read().decode())
                        elif source["type"] == "file":
                            extension["metadata"] = open_json_file(os.path.join(source["path"], meta_filename))

        for extension in all_p.values():
            extension['status'] = "AVAILABLE"
        return all_p

    def get_installed_extensions(self):
        installed_p = open_json_file(
            os.path.join(self.extension_path, "installed.json")
        )
        if installed_p is None:
            installed_p = {}

        return installed_p

    def get_all_extensions(self):
        extensions = self.get_extensions_from_sources()
        installed_p = self.get_installed_extensions()
        for extension in installed_p.values():
            extensions[get_extension_key_from_extension(extension)] = extension
        return extensions

    def get_builtin_extension_src_path(self):
        return pkg_resources.resource_filename(__name__, "extensions")

    

    def _run_module_function(self, module_name, function_name=None, kwargs={}):
        extensionmod = importlib.import_module(name="{}.extension".format(module_name))
        return getattr(extensionmod, function_name)()

    def _run_extension_package_install(self, extension):
        cmd = None
        if extension["source"]["type"] == "file":
            full_path = os.path.join(self.get_builtin_extension_src_path(), extension["id"])
            cmd = "pip install --user -e {}".format(full_path)
        elif extension["source"]["type"] == "workspace":
            full_path = os.path.join(self.workspace_path, extension["id"])
            cmd = "pip install --user -e {}".format(full_path)
        elif extension["source"]["type"] == "path":
            full_path = "{}{}".format(extension['source']['path'],extension["id"])
            cmd = "pip install --user -e {}".format(full_path)
        elif extension["source"]["type"] in ["url","test","remote"]:
            extension_path = extension.get("path")
            self.logger.info(f"Extension Path  {extension_path}, type {extension.get('type')}")
            if extension_path is None:
                raise Exception("No remote path defined in extension. Cannot install with out path.")
            if "rpath" in extension.get("type"):
                extension_path = "{}{}".format(extension['source']['path'],extension_path)
            cmd = "pip install --user {}".format(extension_path)
        else:
            cmd = "pip install --user {}=={}".format(extension["id"], extension["version"])

        self.logger.info(f"Install Command {cmd}")

        # TODO: Should remove below: so far unsuccessful in loading modules installed for first time using pip during runtime
        # https://stackoverflow.com/questions/32478724/cannot-import-dynamically-installed-python-module
        import site
        cmd_result = run_bash_command(cmd)
        self.logger.error(cmd_result)
        if not os.path.exists(site.USER_SITE):
            os.makedirs(site.USER_SITE)
        sys.path.insert(0, site.USER_SITE)
        importlib.invalidate_caches()
        return cmd_result

    def _run_extension_package_remove(self, extension):
        cmd = "pip uninstall -y {}".format(extension["id"])
        self.logger.info("-------------------------------------------")
        self.logger.info(cmd)
        return run_bash_command(cmd)

    def finish_installing_new_extensions(self):
        self.logger.debug("Loading New Extensions")
        lock = FileLock(os.path.join(self.extension_path, ".loading.lock"))
        with lock:
            for extension in self.get_installed_extensions().values():
                if extension['status'] == 'PREPPED_RELOAD':
                    self.logger.info("Finishing Install of {}".format(extension['id']))
                    self.install_extension(extension_id=extension['id'], extension_version=extension['version'])

    def load_extensions(self):
        self.logger.debug("Loading Extensions")
        lock = FileLock(os.path.join(self.extension_path, ".loading.lock"))
        with lock:
            for extension in self.get_installed_extensions().values():
                if extension['status'] == "INSTALLED":
                    try:
                        self.logger.debug("Loading {}".format(extension['id']))
                        module_name = extension.get('module_name', extension['id'].replace('-', "_"))
                        self._run_module_function(module_name, "load", kwargs=extension.get("load_kwargs", {}))
                    except ModuleNotFoundError as e:
                        logging.error(f"Unable to load extension {extension['id']}, {e}")


    def update_extension_status(self, extension, state, msg=""):
        lock = FileLock(os.path.join(self.extension_path, ".updating.lock"))
        old_status = "NA"
        with lock:
            installed_p = self.get_installed_extensions()
            installed_p[extension['id']] = extension
            old_status = extension.get('status')

            extension['status'] = state
            extension['status_msg'] = msg
            extension['status_timestamp'] = time.time()

            save_to_json_file(
                installed_p, os.path.join(self.extension_path, "installed.json")
            )
            self.logger.info("Updated {} status: from {} to {}".format(extension['id'], old_status, extension['status']))

    def cleanup(self):
        for extension in self.get_installed_extensions().values():
            if extension['status'] in ["INSTALLING", "UNINSTALLING"]:
                self.update_extension_status(extension, "INSTALL_FAILED")

    def remove_extension_by_id(self, extension_id):
        lock = FileLock(os.path.join(self.extension_path, ".updating.lock"))
        with lock:
            installed_p = self.get_installed_extensions()
            installed_p.pop(extension_id)
            save_to_json_file(
                installed_p, os.path.join(self.extension_path, "installed.json")
            )

    def get_extensions_by_id(self, extension_id):
        results = []
        for extension in self.get_all_extensions().values():
            if extension['id'] == extension_id:
                results.append(extension)
        return results

    def get_extension(self, extension_id, extension_version):
        extension_key = get_extension_key(extension_id, extension_version)
        return self.get_all_extensions().get(extension_key)

    def install_extension(self, extension_id, extension_version, package_only=False):

        extension_key = get_extension_key(extension_id, extension_version)

        extension = self.get_all_extensions().get(extension_key)

            # xvfb-run pistarlab_extension_tools --action=save_manifest --extension_path /home/brandyn/pistarlab/workspace/pistarlab-landia


        extension_id = extension['id']

        if extension['source']['type'] == "workspace":
            try:
                self.logger.info(f"Attempting to (re)create workspace extension manifest {extension_id}")
                # TODO: xvfb may not be installed
                cmd = f"xvfb-run pistarlab_extension_tools --action=save_manifest --extension_path {extension['full_path']}"
                cmd_result = run_bash_command(cmd)
                self.logger.info(cmd_result)
            except Exception as e:
                self.logger.error(f"Failed to load extension manifest with f{cmd}")
                self.logger.error("Please resolve the issue and try again")

        if extension is None:
            raise Exception("Extension {} not found.".format(extension_key))

        elif extension['status'] in ["INSTALLING", "UNINSTALLING"]:
            raise Exception("Cannot perform action. Extension {} is currently {}.".format(extension_key, extension['status']))

        self.update_extension_status(extension, "INSTALLING")

        # install package
        try:
            self.logger.info("Installing {}".format(extension_key))
            result = self._run_extension_package_install(extension)
            self.logger.info("Package Install Output\n {}".format(result))
        except Exception as e:
            msg = "{}\n{}".format(e, traceback.format_exc())
            self.logger.error(msg)
            self.update_extension_status(extension, "INSTALL_FAILED", msg=msg)
            return False

        if package_only:
            self.update_extension_status(extension, "PREPPED_RELOAD", msg="Boostrap Install")
            return True

        # try to import, if failed, set state to RELOAD_REQUIRED
        try:
            importlib.invalidate_caches()
            importlib.import_module(self.extension_id_to_module_name(extension_id))
        except Exception as e:
            msg = "{}\n{}".format(e, traceback.format_exc())
            self.logger.error(msg)
            self.update_extension_status(extension, "PREPPED_RELOAD", msg=msg)
            return True


        # Run extension install
        try:
            module_name = extension.get('module_name', extension_id.replace('-', "_"))
            result = self._run_module_function(module_name, "install")
            self.logger.info("Extension Install Output\n {}".format(result))
            self.update_extension_status(extension, "INSTALLED")
            # self.update_extension_packages_file()
            return True
        except Exception as e:
            msg = "{}\n{}".format(e, traceback.format_exc())
            self.logger.error(msg)
            self.update_extension_status(extension, "INSTALL_FAILED", msg=msg)
            return False

    # def update_extension_packages_file(self):
    #     with open(os.path.join(self.extension_path, "extension_packages.txt"), "w") as f:
    #         for extension_id in self.get_installed_extensions().keys():
    #             f.write(f"${extension_id}\n")

    def reload_extension_by_id(self, extension_id):
        module_name = self.extension_id_to_module_name(extension_id)
        module = importlib.import_module(module_name)
        importlib.reload(module)
        return True

    def uninstall_extension(self, extension_id):
        self.logger.debug("Uninstalling {}".format(extension_id))
        installed_p = self.get_installed_extensions()
        extension = installed_p.get(extension_id)
        if extension['status'] in ["INSTALLING", "UNINSTALLING"]:
            raise Exception("Cannot perform action. Extension {} is currently {}.".format(extension_id, extension['status']))

        # Mark as INSTALLING
        self.update_extension_status(extension, "UNINSTALLING")

        try:
            # Run Uninstallion
            module_name = extension.get('module_name', extension_id.replace('-', "_"))
            result = self._run_module_function(module_name, "uninstall")
            self.logger.debug("Extension Uninstall Output\n {}".format(result))
            remove_result = self._run_extension_package_remove(extension)
            self.logger.debug("Package Uninstall Output\n {}".format(remove_result))

            # Remove Extension Entry
            self.remove_extension_by_id(extension_id)
            return True
        except ModuleNotFoundError as e:
            self.logger.error(e)
            self.logger.info("Removing anyway")
            self.remove_extension_by_id(extension_id)
            return True
        except Exception as e:
            self.logger.error(e)
            self.logger.info("Removing anyway")
            self.remove_extension_by_id(extension_id)
            return True
            # self.logger.error(e)
            # self.update_extension_status(extension, "UNINSTALL_FAILED", msg="{}\n{}".format(e, traceback.format_exc()))
            # return False
