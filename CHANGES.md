# Problem 1
Issue with ollama local. Moved to branch kannan/fix-tooling
Tried
```
pip install .
```
Issue with dotenv. Replaced dotenv with python-dotenv in .toml file

Added support for claude.

# Problem 2
The `crewai-tools` package breaks a lot of dependencies. \
Removed `crewai-tools` from `pyproject.toml`