# Model Evaluation Report

## 1. Introduction
- **Objective**: Predicting survival on the Titanic using various machine learning models.
- **Dataset**: [Link to Dataset](https://www.kaggle.com/c/titanic/data)
- **Model Evaluated**: Logistic, Regression, Random Forest, Gradient Boosting

## 2. Models and Hyperparameters
- **Logistic Regression**
  - Hyperparamaters: `C = [0.1, 1, 10]}`
- **Random Forest**
  - Hyperparamters: `n_estimators = [100, 200]`, `max_depth = [None, 10, 20]`
- **Gradient Boosting**
  - Hyperparameters: `n_estimators = [100, 200]`, `learning_rate = [0.01, 0.1, 0.2]`


## 3. Performance Metrics
### Accuracy
|Model | Accuracy |
|---|----------|
|LogisticRegression | 0.79     |
|Random Forest | 0.80     |
|Gradient Boosting| 0.80     |

### Precision, Recall, F1-Score
- **Logisic Regression**
  - Precision: 0.76
  - Recall: 0.72
  - F1-Score: 0.74

- **Random Forest**
  - Precision: 0.78
  - Recall: 0.72
  - F1-Score: 0.75

- **Gradient Boosting**
  - Precision: 0.81
  - Recall: 0.68
  - F1-Score: 0.74


## 4. Model Comparison and Selection
- **Best Model**: Random Forest
- **Reason for Selection**:
  - Joint highest accuracy
  - Overall balanced performance
  - Robust to overfitting
  - Maintaining better recall compared to Gradient Boosting


## 5. Insights and Next Steps
- **Feature Importance**:
  - Describe key features and their importance in the Random Fores model.
- **Limitations**:
  - Discuss any limitations in the current approach.
- **Next Steps**:
  - Suggestions for further tuning, additional models to explore, or more sophisticated techniques like ensemble methods.
