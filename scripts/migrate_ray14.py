"""Migrate a pistarlab data.db from ray 1.2-era RLlib configs to ray 1.4.1.
Usage: python migrate_ray14.py <target data.db> <data.db with specs registered under ray 1.4.1>"""
import sqlite3, json, sys
STALE = ['_use_trajectory_view_api', 'memory', 'memory_per_worker', 'object_store_memory',
         'object_store_memory_per_worker', 'replay_sequence_length']
db, new = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
with db:
    for i, c, p in new.execute("select id, config, params from agent_spec where id like 'rllib%'"):
        db.execute("update agent_spec set config=?, params=? where id=?", (c, p, i))
    for a, c in list(db.execute("select id, config from agent where spec_id like 'rllib%'")):
        cfg = json.loads(c)
        for k in STALE:
            cfg.get('trainer_config', {}).pop(k, None)
        db.execute("update agent set config=? where id=?", (json.dumps(cfg), a))
        print(a, 'cleaned')
print('remaining stale refs:',
      db.execute("select count(*) from agent where config like '%_use_trajectory_view_api%'").fetchone()[0],
      db.execute("select count(*) from agent_spec where config like '%_use_trajectory_view_api%'").fetchone()[0])
