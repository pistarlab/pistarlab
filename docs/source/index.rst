.. piSTAR Lab documentation master file, created by
   sphinx-quickstart on Tue May  4 10:08:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

piSTAR Lab Documentation
==========================
.. image:: images/pistar_logo_black.png


piSTAR Lab is an open source modular deep reinforcement learning development platform built to make AI development accessible and fun. 

* Github: http://github.com/pistarlab/pistarlab


.. note::
   piSTAR Lab is in early development and **not ready for public use**.

Features
--------
* Web UI
* Plugin System for adding new agents, environments or tasks types
* Python API, anthing you can do in the UI, you can do in Python as well
* Run agents in single and multi player environments
* Experiment tracking
* Built in web-based IDE (via Theia (https://theia-ide.org/))
* Uses Ray Project (https://ray.io/) under the hood for distributed processing


.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2
   
   overview
   installation
   usage
   plugins
   advanced
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Development

   design
   development
   roadmap

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   modules
