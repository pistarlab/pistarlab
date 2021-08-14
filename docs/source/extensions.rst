Extensions
=======

Extensions are the primary mechanism for adding new Agents, Environments, Tasks and Components.


Development Notes
-----------------

Creating a manifest
~~~~~~~~~~~~~~~~~~~

Manfiest files are used to speed up the installation of Extensions. They are especially useful for Environments where probing is required.

**Example of creating a manifest**

   .. code-block:: bash
   
    xvfb-run python pistarlab/extensions_tools.py --action=save_manifest --extension_path PATH_TO_EXTENSION/pistarlab-envs-gym-main

