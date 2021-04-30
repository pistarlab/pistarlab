
import json
import logging
import os
import tarfile


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname="data")


def get_snapshots_from_file_repo(data_root):
    logging.info("Loading snapshots {}".format(data_root))
    items = {}
    for (dirpath, dirnames, filenames) in os.walk(data_root):
        for file in filenames:
            if file.endswith(".json"):
                file_path = os.path.join(dirpath, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    snapshot_id = "{}_{}_{}_{}_{}".format(data['entity_type'], data['spec_id'], data['submitter_id'], data['seed'], data['snapshot_version'])
                    entry = {
                        'entity_type': data['entity_type'],
                        'spec_id': data['spec_id'],
                        'id': data['id'],
                        'seed': data['seed'],
                        'meta': data['meta'],
                        'config': data['config'],
                        'submitter_id': data['submitter_id'],
                        'creation_time': data['creation_time'],
                        'snapshot_version': data['snapshot_version'],
                        'snapshot_description': data['snapshot_description'],
                        'path': dirpath.replace(data_root, "")[1:],
                        'file_prefix': file.replace(".json", ""),
                        "snapshot_id": snapshot_id
                    }
                    items[snapshot_id] = entry
    return items
