# Loan Prediction System

A machine learning project focused on predicting loan outcomes for banking services.

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Overview

This project addresses a key challenge in banking: identifying good clients for additional services versus high-risk clients requiring careful monitoring. Using machine learning techniques on historical banking data, we developed a predictive system to determine whether a loan will end successfully or not.

## Features

- Data preparation and preprocessing pipeline
- Exploratory data analysis with visualizations
- Implementation of various machine learning models:
  - Random Forest Classifier
  - Decision Tree Classifier
  - Gaussian Naive Bayes
  - Logistic Regression
  - Support Vector Machines
- Model evaluation using AUC and accuracy metrics
- Dataset balancing techniques

## Project Structure

- `data/`: Contains all datasets (client information, account details, transactions, etc.)
- `pipeline/`: Core processing components and machine learning implementation
  - `main.py`: Main execution script
  - `data_preparation.py`: Data preprocessing functions
  - `data_splitting.py`: Training/testing data split utilities
  - Various model implementations
- `docs/`: Documentation and reports
- `data_visualizations.py`: Scripts for exploratory data analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/loan-prediction-system.git
cd loan-prediction-system
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

For data exploration and visualization:
```bash
python data_visualizations.py
```

To run the prediction model:
```bash
cd pipeline
python main.py
```

## Results

The model achieves reliable prediction of loan outcomes, allowing banks to:
- Identify high-risk applicants before loan approval
- Offer tailored services to low-risk clients
- Optimize loan approval processes
- Reduce financial losses from defaulted loans

Detailed results and analysis can be found in the documentation under `docs/`.

## Contributors

- Afonso Caiado
- Elias Lambrecht
- José Miguel Maçães
- Luís Miguel Afonso Pinto

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.