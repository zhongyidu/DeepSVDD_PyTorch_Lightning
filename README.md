# Deep SVDD Lightning

This repository provides a PyTorch Lightning implementation of [Deep SVDD](https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf), based on the original paper and official [PyTorch implementation](https://github.com/lukasruff/Deep-SVDD-PyTorch) by Lukas Ruff.

## 📦 Project Setup

This project uses [`pyenv`](https://github.com/pyenv/pyenv) to manage Python versions and [`poetry`](https://python-poetry.org/) for dependency and environment management.

### 1. Install `pyenv`

Follow the instructions for your OS in the official installation guide:  
<https://github.com/pyenv/pyenv#installation>

### 2. Install Python 3.11.9

```bash
pyenv install 3.11.9
```

### 3. Set the Local Python Version

Navigate to the root directory of this project and run:

```bash
pyenv local 3.11.9
```

This command creates or updates a `.python-version` file in the current directory, ensuring that `pyenv` uses Python 3.11.9 for this project.

### 4. Install `poetry`

Follow the official installation instructions here:  
<https://python-poetry.org/docs/#installation>

### 5. Install Project Dependencies

In the project root directory, run:

```bash
poetry install
```

This creates a virtual environment and installs all dependencies listed in `pyproject.toml`.

---

## 🚀 Running the Experiment

### 1. Activate the Virtual Environment

```bash
poetry shell
```

### 2. Run the Main Script

```bash
python main.py
```

To customize the training configuration, edit the `config.yaml` file located in the `config` directory.

---

