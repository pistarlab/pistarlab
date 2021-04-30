from setuptools import setup, find_packages

setup(
    name="pistarlab-rllib",
    version="0.0.1-dev",
    author="piSTAR",
    author_email="pistar3.14@gmail.com",
    description="rllib",
    long_description='This is a pistarlab plugin for RLLIB',
    url="https://github.com/bkusenda/pistarlab/plugins",
    license='',
    install_requires=['ray==1.2.0'],
    package_data={'pistarlab-rllib': ['README.md',"*.json"]
      },
    packages=find_packages(),
    entry_points={
    'pistarlab_plugin' : [
      "install =  pistarlab_rllib.plugin:install",
      "load =  pistarlab_rllib.plugin:load",
      "uninstall =  pistarlab_rllib.plugin:uninstall"]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)