# 🎓 College Placement Prediction using Machine Learning

## 📘 Overview
This project analyzes and predicts **engineering students' placement outcomes** based on academic and personal attributes using **machine learning algorithms**.  

It performs **Exploratory Data Analysis (EDA)** to uncover trends and relationships among variables like CGPA, internships, backlogs, and hostel status, and applies multiple ML models to predict whether a student will be placed or not.

---

## 🧠 Objectives
- Perform **EDA** on student placement data  
- Identify key factors influencing placement outcomes  
- Build and compare **classification models** for placement prediction  
- Visualize insights using **Seaborn** and **Matplotlib**

---

## 🧰 Tech Stack
- **Python 3.x**
- **NumPy**, **Pandas** – Data processing  
- **Matplotlib**, **Seaborn** – Data visualization  
- **Scikit-learn** – Machine learning models and evaluation  
- **Jupyter Notebook / Google Colab**

---

## 📊 Dataset Information
File: `Placement.csv`

| Column | Description |
|---------|-------------|
| Age | Age of student |
| Gender | Male / Female |
| Stream | Engineering stream (CS, IT, ECE, etc.) |
| Internships | Number of internships completed |
| CGPA | Academic performance (out of 10) |
| Hostel | Whether the student stayed in hostel (1 = Yes, 0 = No) |
| HistoryOfBacklogs | 1 = Has backlogs, 0 = No backlogs |
| PlacedOrNot | Target variable (1 = Placed, 0 = Not Placed) |

---

## 🔍 Exploratory Data Analysis (EDA)
Key visualizations include:
- Distribution of **Age**, **CGPA**, and **Internships**
- Placement rates by **Gender** and **Stream**
- Impact of **Backlogs** and **Hostel stay** on placement
- Correlation heatmaps and count plots


## ⚙️ Model Building
Algorithms used:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVC)**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

Cross-validation was performed using 3 folds to ensure model reliability.

---

## 🧪 Model Evaluation

| Model | Avg Cross-Validation Accuracy |
|--------|-------------------------------|
| SVC | 80.3% |
| Decision Tree | 84.4% |
| Logistic Regression | 71.5% |
| Random Forest | 84.7% |
| KNN | 80.7% |

**Best Model:** 🎯 `Decision Tree Classifier`  
- **Training Accuracy:** 92.4%  
- **Testing Accuracy:** 87.8%  

---

## 📈 Confusion Matrix
A heatmap of predictions was plotted to visualize true positives and false negatives using Seaborn.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
