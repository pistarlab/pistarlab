from distutils.command.install_data import install_data
from setuptools import setup, find_packages
import glob
import os
import shutil
import sys

from setuptools.command.install import install
with open("README.md", "r") as f:
    long_description = f.read()

# Package Data
ui_files = [name.replace("pistarlab/", "", 1) for name in glob.glob("pistarlab/uidist/**", recursive=True)]
plugin_files = [name.replace("pistarlab/", "", 1) for name in glob.glob("pistarlab/plugins/**", recursive=True)]
template_files = [name.replace("pistarlab/", "", 1) for name in glob.glob("pistarlab/templates/**", recursive=True)]

package_files = ui_files + plugin_files + template_files

additional_files = ["thirdparty_lib/redis-server"]


class post_install(install_data):
    def run(self):
        # Call parent
        #install_data.run(self)
        # Execute commands
        print("Running")


setup(
    name="pistarlab",
    version="0.0.1-dev",
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
    url="https://github.com/bkusenda/pistarlab/",
    license='Apache-2.0',
    install_requires=[
        'Flask',
        'Flask-Cors',
        'Flask-GraphQl',
        'graphene',
        'graphene-sqlalchemy',
        'graphene_sqlalchemy_filter'
        'aiohttp_cors',
        'aiortc',
        'SQLAlchemy',
        'shortuuid',
        'simplejson',
        'pyinstrument',
        'sh',
        'xvfbwrapper',  # TODO: MSWIN not compatible
        'opencv_python',
        'ffmpeg-python',
        'gym',
        # TODO: MSWIN issues
        'matplotlib',
        'gym',
        'colorama',
        'gputil',
        "msgpack",
        "msgpack_numpy",
        "pytest",
        "psycopg2-binary",
        'ipykernel',
        'zmq'

        # 'torch',
        # 'torchvision'
        # 'https://download.pytorch.org/whl/cu101/torch-1.7.1%2Bcu101-cp38-cp38-linux_x86_64.whl'
        # 'torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html',
        # 'torchaudio==0.7.2 -f  https://download.pytorch.org/whl/torch_stable.html',
        # 'torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html',
    ],
    package_data={'pistarlab': package_files + additional_files},
    entry_points={
        'console_scripts': [
            'pistarlab = pistarlab.launcher:main',
            'pistarlab_plugin_tools = pistarlab.plugin_tools:main'
        ]
    },
    extras_require={
        'main': [
            'ray[all]==1.2.0',
            'tensorflow==2.3.1',  # TODO Numpy version isssue
            'torch==1.7.1',
            'torchvision==0.8.2'],
        'extras':[ 'pygame']
    },
    packages=find_packages(),
    include_data_files=True,
    include_package_data=True,
    cmdclass={"install_data": post_install},
    classifiers=[
        'Framework :: PiSTAR Lab',
        'Topic :: Software Development',
        'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
    ],
    python_requires='>=3.7',
    zip_safe=False)
