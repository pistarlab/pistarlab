from setuptools import setup, find_packages

setup(
    name="pistarlab-agents-stable-baselines",
    version="0.0.1-dev",
    author="",
    author_email="",
    description="Stable Baselines",
    long_description='This is a pistarlab plugin',
    url="https://github.com/bkusenda/pistarlab/plugins",
    license='',
    install_requires=['stable-baselines>=2.10.1'],
    package_data={'pistarlab-agents-stable-baselines': ['README.md']
      },
    packages=find_packages(),
    entry_points={
    'pistarlab_plugin' : [
      "install =  pistarlab_agents_stable_baselines.plugin:install",
      "load =  pistarlab_agents_stable_baselines.plugin:load",
      "uninstall =  pistarlab_agents_stable_baselines.plugin:uninstall"]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)