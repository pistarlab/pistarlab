import copy
import importlib
import json
import logging
import os
import sys

import setuptools


def get_extension_package_name_from_path(path):
    for pkg_name in setuptools.find_packages(where=path):
        importlib.import_module(name="{}.extension".format(pkg_name))
        return pkg_name
    raise Exception("No extension packages found. Note: extension modules must contain an __init__.py and a extension.py file")

def process_env_specs(env_specs,manifest_files_path,replace_images=True,probe_max_count=0):
    from pistarlab.utils import env_helpers
    updated_env_specs = []
    for i,spec_data in enumerate(env_specs):
        if probe_max_count > 0 and probe_max_count<i:
            print(f"Probe exiting early at {i} as requested by 'probe_max_count' argument")
            break
        print("Probing {}".format(spec_data['spec_id']))
        metadata = env_helpers.probe_env_metadata(
                spec_data, 
                image_path=manifest_files_path,
                replace_images = replace_images)
        spec_data['metadata'] = metadata
        updated_env_specs.append(copy.deepcopy(spec_data))
    return updated_env_specs

def create_manifest_files(path,replace_images=True,max_env_count=0,probe_max_count=0):
    print("CREATING MANIFEST")
    sys.path.append(path)
    module_name = get_extension_package_name_from_path(path)
    sys.path.append(path)
    extensionmod = importlib.import_module(name="{}.extension".format(module_name))
    manifest_data = extensionmod.manifest()
    # extension_id = extensionmod.EXTENSION_ID
    # extension_version = extensionmod.EXTENSION_VERSION

    output_path = os.path.join(path, module_name)
    manifest_files_path = os.path.join(output_path, 'manifest_files')
    os.makedirs(manifest_files_path, exist_ok=True)

    # Probe env_spec metadata for both environments
    
    envs= manifest_data.get('environments')
    if envs is not None:
        print("Loading environments")
        updated_envs = []
        count = 0
        for i, env_data in enumerate(envs):
            if max_env_count > 0 and max_env_count<count:
                print(f"Probe exiting early at {count} as requested by 'max_env_count' argument")
                break
            env_data['env_specs'] = process_env_specs(
                env_data.get('env_specs',[]),
                manifest_files_path=manifest_files_path,
                replace_images = replace_images,
                probe_max_count = probe_max_count)
            count += len(env_data['env_specs'])
            updated_envs.append(copy.deepcopy(env_data))
        manifest_data['environments'] = updated_envs    
    else:
        print("No environments found in manifest")       

    #TODO: add support for agent_specs and task_specs
    with open(os.path.join(output_path, "manifest.json"), 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print("Registry saved to {}".format(output_path))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["save_manifest"], help="options = [save_manifest]")
    parser.add_argument("--extension_path", help="Path to extension project root")
    parser.add_argument("--skip_existing_images", action="store_true", help="Replace env image if exist when probing environment metadata")
    parser.add_argument("--probe_max_count", type=int, default=0, help="Max number specs per env to probe. Useful for testing")
    parser.add_argument("--max_env_count", type=int, default=0, help="Max env to probe. Useful for testing")
    
    args = parser.parse_args()
    if args.action == "save_manifest":
        create_manifest_files(args.extension_path,
            replace_images=not args.skip_existing_images,
            max_env_count = args.max_env_count,
            probe_max_count=args.probe_max_count)
    else:
        print(f"Unknown action {args.action}")


if __name__ == "__main__":
    # Setup Logging
    root = logging.getLogger('pistarlab_extension')
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    main()
