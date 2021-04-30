## Development Setup

To set up `pistarlab` for local development:

1. Fork https://github.com/pistarlab/pistarlab
   (look for the "Fork" button).
   
2. Clone your fork locally:

    ```bash
    git clone git@github.com:YOURGITHUBNAME/pistarlab.git
    ```

3. Create a branch for local development:

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```
   Now you can make your changes locally.

4. When you're done making changes run all the checks and docs builder with [tox](https://tox.readthedocs.io/en/latest/install.html) using one command:

    ```bash
    tox
    ```

5. Commit your changes and push your branch to GitHub::

    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

6. Submit a pull request through the GitHub website.

## Pull Request Guidelines

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests run ```tox```.
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.md`` about the changes.
4. Add yourself to ``AUTHORS.md``.

## Tips

To run a subset of tests::

```bash
tox -e envname -- pytest -k test_myfeature
```

To run all the test environments in *parallel*::

```bash
tox -p auto
```
