# Building Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) with the [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/) for maintaining and generating our documentation.

> **Note:** To avoid dependency conflicts, we strongly recommend using a Python virtual environment to isolate the required dependencies from your system's global Python environment.

## Current-Version Documentation

This section describes how to build the documentation for the current version of the project.

<details open>
<summary><strong>Linux</strong></summary>

```bash
# 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

# 2. Build the current documentation
make current-docs

# 3. Open the current docs
xdg-open _build/current/index.html
```
</details>

<details> <summary><strong>Windows</strong></summary>

```batch
:: 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

:: 2. Build the current documentation
make current-docs

:: 3. Open the current docs
start _build\current\index.html
```
</details>


## Multi-Version Documentation

This section describes how to build the multi-version documentation, which includes previous tags and the main branch.

<details open> <summary><strong>Linux</strong></summary>

```bash
# 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

# 2. Build the multi-version documentation
make multi-docs

# 3. Open the multi-version docs
xdg-open _build/index.html
```
</details>

<details> <summary><strong>Windows</strong></summary>

```batch
:: 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

:: 2. Build the multi-version documentation
make multi-docs

:: 3. Open the multi-version docs
start _build\index.html
```
</details>



for annotating the datset

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
--device cpu --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 --auto \
--input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5

for generating additional datset

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cpu --num_envs 10 --generation_num_trials 10 \
--input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset_small.hdf5

for full dataset collection

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cpu --headless --num_envs 10 --generation_num_trials 1000 \
--input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5  sdfhj