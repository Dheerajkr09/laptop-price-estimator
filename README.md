# Laptop Price Estimator

A machine learning web app that predicts laptop prices based on hardware specifications.  
Built end-to-end — from raw data to a deployed web application.

🔗 **Live Demo:** [laptop-price-estimator-69ye.onrender.com](https://laptop-price-estimator-69ye.onrender.com/)

---

## ML Pipeline

| Step | What was done |
|---|---|
| 1. Data Cleaning | Handled missing values, corrected dtypes, removed duplicates |
| 2. EDA | Analyzed price distribution, brand trends, feature correlations |
| 3. Feature Engineering | Extracted PPI from resolution, encoded categorical features |
| 4. Modelling | Trained & compared regression models, selected best pipeline |
| 5. Website | Built interactive UI with Streamlit |
| 6. Deployment | Deployed on Render |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn |
| Web App | Streamlit |
| Deployment | Render |

---

## Features

- Predicts laptop price based on brand, CPU, RAM, GPU, storage, display, and OS
- Log-transformed target variable for better regression performance
- PPI (Pixels Per Inch) derived from screen resolution as an engineered feature
- Sklearn Pipeline with preprocessing and model bundled together
- Clean and interactive Streamlit UI

---

## Project Structure

```
laptop-price-predictor/
│
├── laptop_data.csv          # Raw dataset
├── notebook.ipynb           # EDA + Feature Engineering + Model Training
├── app.py                   # Streamlit web application
├── pipe.pkl                 # Trained model pipeline (serialized)
├── df.pkl                   # Processed dataframe (for UI dropdowns)
└── requirements.txt         # Dependencies
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/laptop-price-predictor.git
cd laptop-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## How It Works

1. User selects laptop specs via dropdowns and sliders
2. Input is passed into a trained **Sklearn Pipeline**
3. Pipeline applies preprocessing (OneHotEncoding via ColumnTransformer) then predicts
4. Prediction is in log scale — `np.exp()` converts it back to actual price in INR

## App Preview
---
       <img width="1853" height="852" alt="Screenshot" src="https://github.com/user-attachments/assets/07766c7c-949b-4f97-b88d-a2d031d3d7f2" />" />

