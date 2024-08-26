# Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.1-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-green)
![Status](https://img.shields.io/badge/project--status-work--in--progress-orange)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

This project involves building a machine learning model to predict the survival of passengers aboard the Titanic. The project follows a structured machine learning workflow including data exploration, preprocessing, model training, evaluation, serialization, and deployment. Additionally, a real-time monitoring system is conceptualized to track model performance and data drift over time.

**Note:** The deployment to Heroku was conducted as a proof of concept and has since been removed to avoid unnecessary costs.

---

## Project Structure

- **src/**: Contains model training and data preprocessing docs
- **data/**: Contains raw and processed datasets.
- **notebooks/**: Jupyter notebooks for EDA.
- **app/**: Flask application for serving the model.
- **monitoring/**: Conceptual dashboard for real-time monitoring using Plotly Dash.
- **docs/**: Contains graphs from EDA stage
- **Dockerfile**: For containerizing the application.
- **README.md**: Project documentation.


## Dataset

The dataset used is the [Titanic Passenger Data](https://www.kaggle.com/c/titanic/data) from Kaggle. It contains information about passengers such as age, sex, class, etc., along with their survival status.

**Features:**
- `PassengerId`: Unique ID for each passenger.
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Name`: Name of the passenger.
- `Sex`: Gender of the passenger.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings/spouses aboard.
- `Parch`: Number of parents/children aboard.
- `Ticket`: Ticket number.
- `Fare`: Passenger fare.
- `Cabin`: Cabin number.
- `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

**Target Variable:**
- `Survived`: Survival status (0 = No, 1 = Yes).

## Results

Upload model to kaggle for results

## Future Work
- **Enahnced Feature Engineering**: Incorporate more sophisticated features such as title etraction form names or gamily size categorization.
- **Advanced Modeling Techniques**: Experiment with ensemble methods like Gradient Boosting Machines or stacking models.
- **Improved Deployment**: This Project was deployed using Heroku as a proof of concept, it has since been deleted as to not incur any fees. In the future i would like to deploy with a cloud servid with scalability such as GCP
- **Automated Monitoring**: Fully implement and deploy the real time monitoring dashboard for conintuous performance tracking.
- **CI/CD**: Set up automated pipelines using tools like Github Actions for streamlined development and deployment process
