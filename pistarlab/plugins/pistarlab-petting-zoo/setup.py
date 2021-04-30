from setuptools import setup, find_packages
import glob

registry_files = [name.replace("pistarlab_petting_zoo/","",1) for name in glob.glob("pistarlab_petting_zoo/registry_files/**",recursive=True)]

setup(
    name="pistarlab-petting-zoo",
    version="0.0.1-dev",
    author="pistar",
    author_email="",
    description="https://github.com/PettingZoo-Team/PettingZoo",
    long_description='https://github.com/PettingZoo-Team/PettingZoo',
    url="https://github.com/bkusenda/pistarlab/plugins",
    license='',
    install_requires=[
        'pettingzoo==1.5.0',
        'autorom','multi_agent_ale_py','chess','magent'],
    package_data={'pistarlab_petting_zoo': ['README.md',"*.json","*.jpg", "registry_files", "registry.json"] + registry_files
      },
    packages=find_packages(),
    entry_points={ },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)