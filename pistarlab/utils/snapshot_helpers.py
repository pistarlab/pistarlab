
import json
import logging
import os
import tarfile
from typing import Dict,Any


def make_tarfile(output_filename:str, source_dir:str) -> None:
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname="data")

def extract_tarfile(tarfilename:str,output_path:str)  -> None:
    tar = tarfile.open(tarfilename)
    tar.extractall(output_path)
    tar.close()


def get_snapshots_from_file_repo(data_root:str) -> Dict[str,Dict[str,Any]]:
    logging.info("Loading snapshots {}".format(data_root))
    items = {}
    for (dirpath, dirnames, filenames) in os.walk(data_root):
        for file in filenames:
            if file.endswith(".json"):
                file_path = os.path.join(dirpath, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    # # For backward compatibility
                    # if "snapshot_id" not in data:
                    #     data['snapshot_id'] = "{}_{}_{}_{}".format(data['id'], data['spec_id'], data['seed'], data['snapshot_version'])

                    entry = {
                        "snapshot_id": data['snapshot_id'],
                        'entity_type': data['entity_type'],
                        'spec_id': data['spec_id'],
                        'id': data['id'],
                        'seed': data['seed'],
                        'agent_name': data['agent_name'],
                        'meta': data['meta'],
                        'config': data['config'],
                        'submitter_id': data['submitter_id'],
                        'creation_time': data['creation_time'],
                        'env_stats': data.get('env_stats',{}),
                        'snapshot_version': data['snapshot_version'],
                        'snapshot_description': data['snapshot_description'],
                        'path': dirpath.replace(data_root, "")[1:],
                        'file_prefix': file.replace(".json", ""),

                    }
                    items[entry['snapshot_id']] = entry
    return items
