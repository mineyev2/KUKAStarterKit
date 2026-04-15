# Installation

## Motion Planning Environment Setup

This repo uses Poetry for dependency management. To set up this project, first install
[Poetry](https://python-poetry.org/docs/#installation) and, make sure to have Python3.10
installed on your system.

Then, configure poetry to set up a virtual environment that uses Python 3.10:
```
poetry env use python3.10
```

Next, install all the required dependencies to the virtual environment with the
following command:
```bash
poetry install
```

To run any code using this venv, you can do this:
```bash
poetry run python3 <PYTHON_FILE>
```

On the other hand, you can also just activate the venv:
```bash
eval $(poetry env activate)
```

## KUKA Hardware Setup

Refer to our lab's `#hardware_kuka_iiwa` channel for setup details.

# Usage