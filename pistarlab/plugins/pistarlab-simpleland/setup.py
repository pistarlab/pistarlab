from setuptools import setup, find_packages

setup(
    name="pistarlab_simpleland",
    version="0.0.1-dev",
    author="piSTAR",
    author_email="pistar3.14@gmail.com",
    description="https://github.com/pistarlab/simpleland",
    long_description='This is a pistarlab plugin',
    url="https://github.com/pistarlab/pistarlab/plugins",
    license='',
    install_requires=[],
    package_data={'pistarlab-simpleland': ['README.md']
      },
    packages=find_packages(),
    entry_points={},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)