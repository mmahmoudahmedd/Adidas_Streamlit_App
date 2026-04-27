# 👟 Adidas US Sales Prediction App

An end-to-end **machine learning web application** that predicts Adidas product sales across the United States based on retailer, region, product type, and pricing inputs. Built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 🎯 Project Overview

This project demonstrates a complete data science workflow — from raw data exploration to a deployed, interactive machine learning model. Users can input business parameters (retailer, region, product, price, units sold, sales method) and instantly receive a predicted sales value powered by a trained **Random Forest Regression** model.

### Business Use Case
Helps sales managers and category planners forecast expected sales performance and make data-driven decisions about pricing, inventory allocation, and regional strategy.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Random Forest Regressor) |
| **Preprocessing** | LabelEncoder, StandardScaler |
| **Web Framework** | Streamlit |
| **Model Persistence** | Pickle |
| **Dataset** | Adidas US Sales Dataset (Excel) |

---

## ✨ Features

- 🎨 **Interactive UI** — Clean Streamlit interface with dropdowns and inputs for all model features
- 🤖 **Trained ML Model** — Random Forest Regressor trained on real Adidas US sales data
- 🔄 **Encoded Categorical Variables** — Pre-fitted encoders for Retailer, Region, State, City, Product, and Sales Method
- 📊 **Real-Time Predictions** — Instant sales forecasts based on user inputs
- 📦 **Modular Design** — Separate `.pkl` files for model, scaler, and each encoder for easy maintenance

---

## 📂 Project Structure

```
Adidas_Streamlit_App/
│
├── app.py                              # Main Streamlit application
├── Adidas US Sales Datasets.csv.xlsx   # Source dataset
├── random_forest_sales_model.pkl       # Trained Random Forest model
├── scaler.pkl                          # Fitted StandardScaler
├── Retailer_encoder.pkl                # Encoder for Retailer column
├── Region_encoder.pkl                  # Encoder for Region column
├── State_encoder.pkl                   # Encoder for State column
├── City_encoder.pkl                    # Encoder for City column
├── Product_encoder.pkl                 # Encoder for Product column
├── Sales Method_encoder.pkl            # Encoder for Sales Method column
└── README.md                           # Project documentation
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/mmahmoudahmedd/Adidas_Streamlit_App.git
cd Adidas_Streamlit_App

# 2. Install dependencies
pip install streamlit pandas numpy scikit-learn openpyxl

# 3. Run the Streamlit app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Model Details

- **Algorithm:** Random Forest Regressor
- **Target Variable:** Total Sales
- **Features Used:** Retailer, Region, State, City, Product, Price per Unit, Units Sold, Sales Method
- **Preprocessing Pipeline:**
  1. Label encoding for categorical variables
  2. Standard scaling for numerical features
  3. Train-test split for validation
- **Model Persistence:** Saved via Pickle for deployment

---

## 📊 Dataset

The model was trained on the **Adidas US Sales Dataset**, which includes transactional data across multiple retailers, regions, and product categories in the United States.

---

## 🎓 What I Learned

- Building end-to-end ML pipelines from raw data to deployment
- Handling categorical encoding consistently between training and inference
- Designing intuitive UIs with Streamlit for non-technical users
- Persisting and reloading ML artifacts (models, scalers, encoders)
- Translating a business problem into a data science solution

---

## 🔮 Future Improvements

- [ ] Deploy the app on **Streamlit Community Cloud** for public access
- [ ] Add model performance metrics (R², MAE, RMSE) to the UI
- [ ] Integrate visualizations of historical trends alongside predictions
- [ ] Compare multiple models (XGBoost, Gradient Boosting) for benchmarking
- [ ] Add input validation and error handling

---

## 👤 Author

**Mahmoud Ahmed Omar**
Certified Data Scientist | BIS Senior @ AAST

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahmoud-ahmed-9454a8253/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mahmouddaamarr@gmail.com)

---

⭐ If you found this project useful, please consider giving it a star!
