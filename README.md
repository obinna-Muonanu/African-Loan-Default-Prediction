# African-Loan-Default-Prediction
## Project Overview
Financial institutions need to predict loan defaults to reduce risks and make better lending decisions. In Africa's growing financial markets, with different types of customers and changing economic conditions, it is very important to accurately assess the risk of default.
The goal of this project is to develop a generalisable and robust machine learning model to predict the likelihood of loan defaults for both new and existing customers. This project is of great importance as it will help financial institutions make better decisions by predicting the likelihood of loan defaults. This can lead to:
- Identifying high-risk loans early allows institutions to take preventive measures, reducing financial losses.
- By managing risks effectively, institutions can confidently expand their operations in dynamic and growing markets, such as those in Africa.
- Reducing defaults contributes to a more stable financial system, benefiting both lenders and borrowers.
## Data Handling
The data used for this project was provided by a private asset manager that operates in several financial markets across Africa. You can access the data here: [Data Link](https://zindi.africa/competitions/african-credit-scoring-challenge/data)
### Data Preprocessing

1. **Loan Amount Adjustment**:  
   - Set the minimum values for "Loan Amount" and "Loan Amount to Repay" to 1000.  
   - This was based on findings during EDA, where some loans were unrealistically small (e.g., 2 KHS or 2 Ghana Cedis).  

2. **Merging External Data**:  
   - Added external data from the Federal Reserve Bank related to the countries of interest to improve the model's accuracy.  

3. **Handling Missing Values**:  
   - Filled in missing values to ensure the dataset was complete.  

4. **Encoding Categorical Features**:  
   - Used label encoding to convert categorical variables into numerical form.  

5. **Feature Engineering**:  
   - Created new features to add more information to the model:  
     - **Loan to Interest Rate**: Total loan amount × country-specific lending interest rate (from external data).  
     - **Repayment Ratio**: Total loan amount to repay ÷ total loan amount.  
     - **Interest Rate**: (Total loan amount to repay - total loan amount) ÷ total loan amount.  
     - **Interest Rate of Lender**: (Lender portion to be repaid - amount funded by lender) ÷ amount funded by lender.  
     - **Amount Difference**: Total loan amount to repay - total loan amount.  
     - **Lender Difference**: Lender portion to be repaid - amount funded by lender.  

6. **Log Transformation**:  
   - Applied log transformation to address skewness in the features.  

7. **Clustering**:  
   - Created a new "Cluster" feature using unsupervised learning (KMeans) to identify potential patterns in the data.  
   - Clustering was based on: `lender_id`, `duration`, `loan_type`, `disbursement_year`, `Interest Rate`, and `Interest Rate of Lender`.  

8. **Feature Reduction**:  
   - Dropped irrelevant features that didn’t add value to the model.  
These steps ensured the dataset was clean, enriched, and ready for model building.

## Modelling
The primary model used in this project was the XGBoost algorithm.
XGBoost (Extreme Gradient Boosting) is a machine learning algorithm based on decision trees, designed for speed and performance. It works by building a series of decision trees sequentially, where each new tree focuses on correcting the errors of the previous ones.

XGBoost outperformed the following algorithms used in the project:

- Logistic Regression: A simple linear model, which struggled to capture complex patterns in the data.
- Random Forest: A bagging method, less efficient compared to XGBoost’s boosting approach.
- CatBoost: Another gradient boosting algorithm, but XGBoost was better tuned for this dataset.
- LightGBM: Similar to XGBoost but didn’t perform as well due to dataset specifics.
- SVC (Support Vector Classifier): Worked well but struggled with the dimensionality of the data.
- AdaBoost: A simpler boosting method, less powerful than XGBoost for this use case.

## HyperParameter Tuning

Hyperparameter tuning was performed using **Optuna**, an efficient and automated framework for hyperparameter optimization. Optuna systematically explored different combinations of hyperparameter values to find the optimal settings for the XGBoost model.

The final values of the hyperparameters used were:  
- **n_estimators**: 900  
- **max_depth**: 12  
- **learning_rate**: 0.007  
- **alpha**: 0.7 (L1 regularization term to prevent overfitting)  
- **subsample**: 0.668 (proportion of data used for training each tree to introduce randomness)  
- **random_state**: 20 (ensures reproducibility of results)  

These tuned parameters helped improve the model's performance by finding the best balance between underfitting and overfitting, ensuring accurate predictions on the test set.

## Training and Evaluation

#### Data Splitting:  
- The dataset was split into **training (80%)** and **validation (20%)** sets using the `train_test_split` function from `sklearn`.  
- The `stratify` parameter was set to the target variable (`y`) to ensure an even distribution of classes in the train and validation sets.  

#### Group Labels for Cross-Validation:  
- **Stratified GroupKFold** cross-validation was used to ensure that each customer group (identified by `customer_id`) is represented in all folds, preventing data leakage and ensuring that the model is evaluated fairly.
- The `customer_id` column was dropped before fitting the model.  

#### Addressing Class Imbalance:  
To handle the class imbalance, a custom **class weight** function was defined and used. The function calculates the **inverse frequency weights**, and assigns higher weights to underrepresented classes based on their frequency in the dataset. The function is as follows:  
```python
def calculate_class_weights(y):
    # Function to calculate class weights based on class distribution in the target variable
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    class_weights = {}

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)  
        class_weights[class_label] = class_weight

    return class_weights
```

#### Model Training:  
- Cross-validation was performed to validate the model's robustness across 10 different splits.  
- The **XGBoost model** was fitted on the training data using the optimal hyperparameters obtained from tuning.  

#### Evaluation Metrics:  
The model was evaluated using the following metrics:  
1. **F1 Score**: The primary metric for evaluation, as it balances precision and recall, which is critical in imbalanced datasets.  
2. **Accuracy**: Provided a general idea of overall performance.  
3. **Precision**: Assessed how many predicted defaults were actual defaults.  
4. **Recall**: Measured how many of the actual defaults were correctly identified.  
5. **AUC (Area Under the Curve)**: Ensured the model could distinguish between customers who default and those who don’t.  

Focusing on the **F1 Score** and **AUC**, ensures the model's ability to identify defaults accurately and differentiate between defaulters and non-defaulters effectively. 

## Results and Insights

#### Cross-Validation Performance (10 Folds):

The model was evaluated using **10-fold cross-validation**, and the **F1 scores** for each fold are shown below:

- **Fold 1/10**: F1 Score = 0.8541
- **Fold 2/10**: F1 Score = 0.8791
- **Fold 3/10**: F1 Score = 0.9128
- **Fold 4/10**: F1 Score = 0.8136
- **Fold 5/10**: F1 Score = 0.7867
- **Fold 6/10**: F1 Score = 0.8866
- **Fold 7/10**: F1 Score = 0.7742
- **Fold 8/10**: F1 Score = 0.7879
- **Fold 9/10**: F1 Score = 0.8370
- **Fold 10/10**: F1 Score = 0.7805

The **mean F1 score** across all 10 folds was:  
**Mean F1 Score: 0.8312**

This indicates that the model consistently performs well across different subsets of the data, balancing precision and recall effectively.

#### Evaluation on the Validation Set:

The model was also evaluated on the **validation set**, and the classification report is as follows:


| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| **0**     | 1.00          | 1.00       | 1.00         | 12,210      |
| **1**     | 0.92          | 0.85       | 0.89         | 239         |
| **Accuracy** |               |            | **1.00**     | **12,449**  |
| **Macro Avg** | 0.96          | 0.93       | 0.94         | 12,449      |
| **Weighted Avg** | 1.00          | 1.00       | 1.00         | 12,449      |


#### Performance Metrics on the Validation Set:

- **F1 Score**: 0.8850  
- **AUC (Area Under the Curve)**: 0.9260  
- **Accuracy**: 0.9957  
- **Precision**: 0.9189  
- **Recall**: 0.8536  

### Insights:
- The **F1 score** on the validation set is 0.885, which indicates good performance in balancing precision and recall, especially considering the class imbalance.
- The **AUC score** of 0.926 suggests that the model is good at distinguishing between the two classes (loan default vs no default).
- The **accuracy** of 99.57% is very high, but this high accuracy score can be misleading considering the imbalance in the data. Therefore it's important to consider both precision and recall due to the class imbalance.
- The model demonstrates **strong precision (91.89%)**, meaning most of the predicted defaults are correct, but slightly lower recall (85.36%), meaning that some actual defaults are missed.

#### Confusion Matrix
![confusion matrix-african loan prediction](https://github.com/user-attachments/assets/219505d1-3bc4-4bb6-b440-2eb344d10283)
#### ROC-AUC curve
![Roc curve - african loan prediction](https://github.com/user-attachments/assets/85f609d3-1335-4cf8-a086-196a4441634a)


## Some Images from EDA
![class imbalance- african loan prediction](https://github.com/user-attachments/assets/2f8d709a-c492-45b3-8c88-d755e13bbfb5)
![New vs Repeat - African loan prediction](https://github.com/user-attachments/assets/5ae9deb2-3316-4a5d-b192-881b7340f630)
![Loan Default Count- african loan prediction](https://github.com/user-attachments/assets/e0b4e3ac-6b1e-4346-aeac-f24a437a3b54)
![Average loan amount - african loan prediction](https://github.com/user-attachments/assets/99e912f2-36e8-4849-8d8e-ce891f5a0624)
![Feature importance - african loan prediction](https://github.com/user-attachments/assets/1cf550e6-ef17-477f-aca5-0c673fb81d1f)
![Heatmap - african loan prediction](https://github.com/user-attachments/assets/1d2a2e6d-cd2e-4dc1-a8c1-b3e3bec22700)

## App Preview


