from setuptools import setup, find_packages

setup(
    name="pistarlab-agents-stable-baselines",
    version="0.0.1-dev",
    author="",
    author_email="",
    description="Stable Baselines",
    long_description='This is a pistarlab plugin',
    url="https://github.com/pistarlab/pistarlab/plugins",
    license='',
    install_requires=['stable-baselines3'],
    package_data={'pistarlab-agents-stable-baselines': ['README.md']
      },
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)