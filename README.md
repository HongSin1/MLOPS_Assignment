# House Price Prediction App

Overview

This repository contains a House Price Prediction App, which leverages machine learning to estimate house prices based on various features. The application is built using Streamlit for the frontend and utilizes Hydra for configuration management. Additionally, DVC (Data Version Control) is used to handle datasets efficiently.

Features

- Machine Learning Model: Predicts house prices based on input features.

- Streamlit UI: Provides an interactive web-based interface.

- Hydra Configuration: Manages application configurations in a structured manner.

- DVC for Data Management: Ensures version control for datasets.

- Modular Code Structure: Organized using best practices for maintainability.

## Repository Structure

├── .dvc/                           # DVC tracking files

├── config/Wheat_Classification_Model/  # Configuration files managed by Hydra

├── data/                            # Dataset storage (tracked via DVC)

├── models/                          # Trained machine learning models

├── notebooks/                       # Jupyter notebooks for experimentation

├── src/                             # Source code for the app

├── static/css/                      # Static files for styling (if applicable)

├── .gitignore                       # Git ignore file

├── .pre-commit-config.yaml          # Configuration for pre-commit hooks

├── Makefile                         # Automation scripts for building, running, and testing

├── README.md                        # Project documentation (this file)

├── poetry.lock                      # Poetry dependency lock file

├── pyproject.toml                   # Python project dependencies

├── requirements.txt                  # Dependencies for running the project
