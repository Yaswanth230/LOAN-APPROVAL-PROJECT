🎯 Project Title: Loan Approval Prediction Using Decision Tree Classifier
📌 Problem Statement
Banks receive many personal loan applications, but not all applicants are eligible. Processing every application manually is time-consuming and inefficient.
This project aims to build a machine learning model that can automatically predict whether a customer will be approved for a Personal Loan based on their personal and financial attributes.
Objective:
To use a Decision Tree Classifier to predict if a customer will take a personal loan.
📂 Dataset Description
The dataset contains information about 5,000 customers, including:
Age: Age of the customer
Experience: Work experience in years
Income: Annual income in $000
Family: Number of family members
CCAvg: Average credit card spending
Education: Education level (1: Undergrad, 2: Graduate, 3: Advanced/Professional)
Mortgage: Value of house mortgage
Securities.Account, CD.Account, Online, CreditCard: Binary flags (0 or 1) indicating account/product holding
Personal.Loan: Target variable (1 if loan accepted, 0 otherwise)
(Note: ID and ZIP.Code columns will be dropped as they do not help in prediction.)
🧠 Machine Learning Task
Type: Supervised Learning (Classification)
Algorithm: Decision Tree Classifier
Goal: Predict Personal.Loan (0 or 1)


📈 Steps to Follow
Data Exploration and Visualization
Data Preprocessing (Cleaning, Feature Selection)
Model Building with Decision Tree
Evaluation using Accuracy, Confusion Matrix, and Classification Report
Visualizing the Decision Tree \


🔍 Key Steps Followed:
Performed Exploratory Data Analysis (EDA) to understand data distribution and relationships.
Built a basic Decision Tree model, which showed signs of overfitting (Train Accuracy: 100%, Test Accuracy: 98.07%).
Applied GridSearchCV for hyperparameter tuning to improve model generalization.
After tuning:
Train Accuracy: 98.69%
Test Accuracy: 98.47%
Best Parameters: criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=1
🚀 Project Highlights
Cleaned and explored a real-world loan dataset (loan_data.csv)


Handled missing values and dropped irrelevant features


Visualized feature relationships with the target variable


Trained and tuned a Decision Tree Classifier


Evaluated performance using accuracy, precision, recall, and F1-score


Visualized the decision tree structure and confusion matrix



🛠 Tech Stack
Tool / Library
Purpose
Pandas
Data loading & preprocessing
NumPy
Numerical operations
Matplotlib, Seaborn
Data visualization
Scikit-learn
Model training, tuning & evaluation


📊 Visualizations
🔥 Correlation Heatmap


🧮 Age vs Credit Card Usage Scatter Plot


✅ Confusion Matrix


🌳 Decision Tree Visualization



✅ Model Performance
Metric
Class 0 (Not Approved)
Class 1 (Approved)
Precision
0.99
0.90
Recall
0.99
0.95
F1-Score
0.99
0.92


Train Accuracy (Tuned): 98.69%


Test Accuracy (Tuned): 98.47%


Overall Accuracy: 98.47%


Macro Avg F1-Score: 0.96


Weighted Avg F1-Score: 0.98


✨ These results show that the model is highly accurate and effectively identifies both approved and rejected loan applications.

💡 Conclusion
This beginner-friendly machine learning project demonstrates how to:
Handle and clean real-world financial data


Perform Exploratory Data Analysis (EDA)


Build and tune a classification model


Evaluate with comprehensive metrics


Interpret and visualize model outputs


⚡ Next Steps: Consider using ensemble techniques like Random Forest or XGBoost for potentially even better performance.


🏁 Final Conclusion
In this project, I developed a Decision Tree Classification model to predict whether a customer will accept a personal loan offer based on features like income, age, credit card usage, and other financial indicators.
✅ Final Outcome:
The tuned model achieves high and balanced performance on both training and test sets, indicating that it has learned meaningful patterns in the data without overfitting.
This project demonstrates the importance of:
Model evaluation at multiple stages
Hyperparameter tuning
Understanding how tree complexity affects performance
📚 Future Improvements:
Try other classifiers like Random Forest, XGBoost, or Logistic Regression for comparison.
Apply cross-validation for even more robust evaluation.
Explore feature importance for deeper business insights.
