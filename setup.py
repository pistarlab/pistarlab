from distutils.command.install_data import install_data
from setuptools import setup, find_packages
import glob
import os
import shutil
import sys
import re
from setuptools.command.install import install
with open("README.md", "r") as f:
    long_description = f.read()

# Package Data
ui_files = [name.replace("pistarlab/", "", 1) for name in glob.glob("pistarlab/uidist/**", recursive=True)]
plugin_files = [name.replace("pistarlab/", "", 1) for name in glob.glob("pistarlab/plugins/**", recursive=True)]
template_files = [name.replace("pistarlab/", "", 1) for name in glob.glob("pistarlab/templates/**", recursive=True)]

package_files = ui_files + plugin_files + template_files

additional_files = ["thirdparty_lib/redis-server", "thirdparty_lib/redis-server.exe"]


def read_version(path):
    with open(path, 'r') as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


class post_install(install_data):
    def run(self):
        pass

setup(
    name="pistarlab",
    version=read_version(os.path.join("pistarlab","__init__.py"),
    author="Brandyn Kusenda",
    author_email="pistar3.14@gmail.com",
    description="A modular AI agent experimentation tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        'Documentation': 'https://pistarlab.readthedocs.io/',
        'Changelog': 'https://pistarlab.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/pistarlab/pistarlab/issues',
    },
    url="https://github.com/pistarlab/pistarlab/",
    license='Apache-2.0',
    install_requires=[
        'Flask',
        'Flask-Cors',
        'Flask-GraphQl',
        'graphene',
        'graphene-sqlalchemy',
        'graphene_sqlalchemy_filter',
        'aiohttp_cors',
        'aiortc',
        'SQLAlchemy>=1.3, < 1.4',
        'shortuuid',
        'simplejson',
        'pyinstrument',
        'sh',
        'ffmpeg-python',
        'gym',
        'matplotlib',
        'gym',
        'colorama',
        'gputil',
        'psutil',
        "msgpack",
        "msgpack_numpy",
        "pytest",
        "psycopg2-binary",
        'zmq',
        'redis',
        'pyyaml',
        'filelock',
        'opencv_python',
        'xvfbwrapper',  # TODO: MSWIN not compatible
        'ray[all]==1.2.0',
        'pip'
        ],
    package_data={'pistarlab': package_files + additional_files},
    entry_points={
        'console_scripts': [
            'pistarlab_launcher = pistarlab.launcher:main',
            'pistarlab_plugin_tools = pistarlab.plugin_tools:main'
        ]
    },
    extras_require={
        'all': ['tensorflow==2.3.1','torch==1.7.1','torchvision==0.8.2']
    },
    packages=find_packages(),
    include_data_files=True,
    include_package_data=True,
    cmdclass={"install_data": post_install},
    classifiers=[
        'Topic :: Software Development',
        'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.7',
    zip_safe=False)
