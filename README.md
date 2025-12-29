# Customer Churn Prediction Model

Customer churn is a critical problem for banks, as acquiring new customers is significantly more expensive than retaining existing ones.
This project builds an end-to-end machine learning pipeline to predict whether a bank customer is likely to churn, enabling proactive retention strategies.
The focus of this project is not just accuracy, but correct evaluation under class imbalance and business-aware decision making. The goal is to identify customers at high risk of churning so that the bank can intervene early.

## Dataset
**Source:** Kaggle – Bank Customer Churn Dataset    
**Rows:** ~10,000 customers    
**Target Variable:** Exited   
1 → Customer churned          
0 → Customer stayed    

### Key Features     
- **Demographics:** Age, Gender, Geography    
- **Financial:** Balance, CreditScore, EstimatedSalary    
- **Engagement:** NumOfProducts, IsActiveMember, Tenure     

## Analysis
### Exploratory Data Analysis (EDA)
EDA was performed to understand churn patterns across customer segments:
- Overall churn distribution
- Churn rates across geographical regions
- Relationship between age and churn

These insights informed preprocessing and model evaluation strategies.

### Pre-processing
- Removed identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- Encoded categorical variables:
  - Gender → binary encoding
  - Geography → one-hot encoding
- Applied **feature scaling** for Logistic Regression
- Preserved unscaled features for tree-based models


## Models Used

### Logistic Regression (Baseline)
- Interpretable baseline model
- Performs well on the majority class
- Limited ability to capture complex churn patterns

### Random Forest (Final Model)
- Captures non-linear relationships and feature interactions
- Handles class imbalance using `class_weight="balanced"`
- Significantly improves churn detection performance

## Project Structure
```text
bank-customer-churn/
│
├── data/
│   └── Churn_Modelling.csv
│
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── main.py
├── requirements.txt
└── README.md

```

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bank-customer-churn.git
cd bank-customer-churn
```

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```
### 2. Install dependencies
```bash
pip install -r requirements.txt

```
### 3. Run 
```bash
python main.py
```

## Conclusion
### Logistic Regression
<p> <img width="451" height="234" alt="1498BE5A-B372-4D83-A012-D457EDCD4898" src="https://github.com/user-attachments/assets/da147f5c-b25d-4c0b-b7c4-e3af736b6134" />  </p>         
Logistic Regression served as a useful baseline model, achieving an accuracy of 80.8% and a ROC-AUC of 0.77. However, it performed poorly in identifying churned customers, with a recall of only 19%, missing the majority of customers who actually exited the bank. This behavior reflects the model’s tendency to favor the majority class in imbalanced datasets and its inability to capture complex, non-linear relationships. As a result, while Logistic Regression offers interpretability and simplicity, it is not suitable as a standalone model for churn prediction where minimizing missed churners is critical.

### Random Forest
<p><img width="451" height="234" alt="Screenshot 2025-12-29 at 13 51 34" src="https://github.com/user-attachments/assets/c918e612-a3a2-4709-a65b-60eb56b373ac" />   </p>      
The Random Forest model significantly outperformed Logistic Regression across all business-relevant metrics. It achieved a higher accuracy of 86%, improved churn recall of 44%, and a strong ROC-AUC score of 0.85, indicating better ranking and separation of churn risk. By capturing non-linear patterns and feature interactions, Random Forest was able to identify more than twice the number of churned customers compared to the baseline model, without a substantial increase in false positives. This makes Random Forest a more effective and practical choice for customer churn prediction in real-world banking scenarios.

### Top 10 Features
<p><img width="449" height="175" alt="Screenshot 2025-12-29 at 13 52 04" src="https://github.com/user-attachments/assets/20fe16fd-b0f6-4a41-9f73-54245f325099" />        </p>
Feature importance analysis revealed that customer age, activity status, balance, number of products, and geographical location were the most influential factors in churn prediction. In particular, older and inactive customers showed a higher likelihood of churning, especially when associated with higher account balances and lower product engagement. These insights suggest that churn is driven not by a single factor, but by a combination of demographic, financial, and behavioral attributes.
