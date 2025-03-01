# MLOPS_Assignment by Edward and Hongsin

## Overview

Folder Structure

--| config/ – Contains configuration files required for the project.

--| data/ – Stores raw and processed data used in the project.

--| dvctest/ – Used for data version control (DVC) testing.

--| models/ – Includes machine learning or statistical models utilized in data analysis.

--| notebooks/ – Jupyter notebooks for exploratory data analysis and prototyping.

--| src/ – Source code for the main functionality of the project.

--| static/ – Contains static assets like images, CSS, or JavaScript files.

--| .gitignore – Specifies files and directories to be ignored by Git.

--| README.md – Documentation for the project.

--| logs.log – Log file containing runtime logs and debugging information.

--| requirements.txt – List of dependencies required to run the project.

## Potential Requisites

To ensure smooth setup and execution, the following prerequisites may be required:

- Python (>=3.8) – Ensure you have Python installed.

- Git – For version control and repository management.

- DVC (Data Version Control) – For managing data changes and reproducibility.

- Hydra - A flexible framework for managing complex configurations dynamically.

## Usage

- Modify and add datasets to the data/ directory.

- Access Notebooks to retrain models

- Update configuration settings in config/ as required.

- Use models/ to store trained models and related files.

## Deployment

**After unzipping your file, open Visual Studio Code and create your python environment** (do this by creating a new terminal on the top-left of the window): python -m venv it3385_venv

**Proceed to activate the environment**: it3385_venv\Scripts\activate

**Then install the requirements.txt file with**: pip install -r requirements.txt

**Lastly, run the app on streamlit with**: streamlit run src/main_app.py


