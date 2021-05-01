from setuptools import setup, find_packages

setup(
    name="pistarlab-envs-minecraft",
    version="0.0.1-dev",
    author="piSTAR",
    author_email="pistar3.14@gmail.com",
    description="ATARI games from OpenAI's gym",
    long_description='This is a pistarlab plugin',
    url="https://github.com/pistarlab/pistarlab/plugins",
    license='',
    install_requires=['gym-minigrid'],
    package_data={'pistarlab-envs-gym-minigrid': ['README.md']
      },
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)