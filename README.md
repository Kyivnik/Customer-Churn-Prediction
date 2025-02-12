# Customer Churn Prediction  

This project is a **machine learning model for predicting customer churn**. The goal is to help businesses identify customers who are likely to leave, so they can take action to retain them.  

I built this as part of my ongoing learning process, applying **deep learning (TensorFlow)** along with standard data science techniques. I recently graduated, but I already have some hands-on experience in ML projects and documentation, so this is a good example of how I work.  

---

## 📌 **Project Overview**  

- **Problem:** Predict whether a customer will churn based on various features (contract type, monthly charges, internet service, etc.).  
- **Data:** Uses the **Telco Customer Churn dataset** from Kaggle.  
- **Model:** A **deep learning model (TensorFlow, Keras)** trained on structured data.  
- **Deployment:** Web app built with **Streamlit** to demonstrate real-time predictions.  

---

## 🛠 **Tech Stack**  

```
🔹 Python (NumPy, Pandas, Scikit-learn)  
🔹 TensorFlow / Keras (Deep Learning)  
🔹 Streamlit (Web App for Predictions)  
🔹 Matplotlib, Seaborn (Data Visualization)  
🔹 Git & GitHub (Version Control)  
```

---

## 🚀 **How It Works**  

### **1. Data Preprocessing & Feature Engineering**  
- Cleaned and transformed categorical features into numerical values (one-hot encoding).  
- Standardized numerical data using **StandardScaler**.  
- Addressed missing values and imbalances in the dataset.  

### **2. Model Architecture**  
- Multi-layer neural network with **Batch Normalization, Dropout** for better generalization.  
- **Adam optimizer** with **binary cross-entropy loss** for classification.  
- Used **EarlyStopping & ReduceLROnPlateau** to improve training efficiency.  

### **3. Model Evaluation**  
- Achieved around **82% accuracy** on test data.  
- Analyzed performance with **confusion matrix, ROC curve, and classification report**.  

---

## 📊 **Results & Insights**  

- **Churn is strongly correlated** with contract type and monthly charges.  
- Customers on **month-to-month contracts** have the **highest churn rate**.  
- **Deep learning improved accuracy** compared to traditional ML models.  

---

## 🖥 **Run the Web App**  

To test the model interactively:  

```
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

This will launch the **Streamlit UI**, where you can input customer data and get a prediction in real time.  

---

## 🔧 **Future Improvements**  

- Try alternative models (**XGBoost, LGBM**) to compare with deep learning.  
- Improve interpretability using **SHAP or LIME**.  
- Fine-tune hyperparameters further for better performance.  

---

## 👨‍💻 **About Me**  

I’m a recent graduate in **Data Science & Machine Learning**, passionate about AI and building practical ML solutions. This project helped me **strengthen my deep learning skills**, and I’m always looking to improve.  

If you have any feedback or ideas for improvement – feel free to reach out! 🚀  

---

## 📩 **Contact Me**  

🔗 **LinkedIn:** [linkedin.com/in/kyivnik](https://linkedin.com/in/kyivnik)

---

# ⭐ **If you like this project, give it a star!**

