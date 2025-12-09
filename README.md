# GreenAI-Carbon-Emission-Benchmarking
Benchmarking the carbon footprint, runtime, and energy consumption of classical machine-learning models using CodeCarbon and Streamlit.

# Green AI – Carbon Footprint Benchmarking of Machine Learning Models

This project evaluates multiple machine-learning models not only based on accuracy but also on **CO₂ emissions**, **energy consumption**, and **runtime**, using the CodeCarbon tracker.  
Two real-world datasets—**Air Quality** and **Electricity Consumption**—were used to benchmark supervised, regression, and unsupervised learning models.

---

# 1. Project Overview

Modern ML development must focus not only on accuracy but also on environmental sustainability.  
This project builds a **Green AI benchmarking pipeline** that:

- Trains classical ML models  
- Measures **carbon emissions** per model  
- Compares emissions vs runtime vs accuracy  
- Builds an interactive **Streamlit dashboard**  
- Performs statistical analysis to understand emission drivers  

---

# 2. Key Features

- End-to-end reproducible ML workflow  
- Automatic carbon tracking using **CodeCarbon**  
- Feature engineering for Air & Electricity datasets  
- Supervised, regression, and unsupervised model training  
- Statistical comparison (correlation, linear regression)  
- Visual analytics (scatter plots, bubble charts, PCA)  
- Streamlit dashboard for interactive insights  

---

# 3. Research Problem

ML models vary significantly in **energy consumption** and **carbon emissions**, yet most studies focus on predictive accuracy alone.  
There is a lack of:

- Standardised CO₂ benchmarking for classical ML models  
- Visual tools for analysing model emissions  
- Reproducible sustainability-aware ML workflows  

This project addresses these gaps.

---

# 4. Research Questions

**RQ1:** How do classical machine-learning algorithms differ in energy use and CO₂ emissions?  
**RQ2:** What is the relationship between model runtime and carbon emissions?  
**RQ3:** Which models offer the best trade-off between predictive accuracy and environmental impact?

---

# 5. Hypothesis

> “Complex models (e.g., XGBoost, MLP) will consume more energy and emit more CO₂ than simpler models (e.g., Logistic Regression, Random Forest).”

---

# 6. Installation

Install Dependencies
pip install -r requirements.txt

If CodeCarbon fails on Windows:
pip install codecarbon --no-build-isolation

# 7. Running the Project

Run the Notebook
jupyter notebook x23426918_Codefile.ipynb

Run the Streamlit Dashboard
streamlit run streamlit_app.py


Dashboard launches at:
👉 http://localhost:8501/

# 8. Datasets Used
Air Quality Dataset
-Includes sensor readings (CO, NO₂, O₃, temperature, humidity)
-Preprocessing includes cleaning, timestamp creation, scaling

Electricity Dataset
-Includes 370 meter readings
-Feature‐engineered using:
-Total consumption
-Average usage
-Lag features

# 9. Feature Engineering Summary

**Scaling:** StandardScaler for numeric features
**Lag Features:** Added for electricity dataset
**Sensor Derivations:** Air quality variables normalized
**Hardware Specs:**
-CPU: Intel i5-1335U
-RAM: 16 GB
-GPU: None (CPU-only training)
-Missing values removed, time fields combined
These transformations improved interpretability and reduced noise.

# 10. Models Trained
**Supervised & Regression**
-Logistic Regression
-Random Forest
-XGBoost
-MLP (Neural Network)
**Unsupervised**
-KMeans
-Gaussian Mixture Model
Unsupervised models were included to explore clustering patterns with minimal emissions.

# 11. Carbon Emission Formula (CodeCarbon)

CodeCarbon estimates emissions using:

Energy Consumption
Energy = CPU Power × Time + GPU Power × Time + RAM Power × Time

CO₂ Emissions
Emissions = Energy × Carbon Intensity of Region

Regression Relationship Found
Emissions ≈ 8.21 × 10⁻⁸ + (5.276 × 10⁻⁶ × Runtime)
➡ Runtime is a dominant predictor of CO₂ emissions.

# 12. Statistical Analysis
Correlation Test
r ≈ 1.0, p < 0.001


Runtime strongly correlates with emissions.

Linear Regression Results

R² = 1.000

p-value < 0.001

Longer runtime → higher CO₂ footprint

Visuals Produced

Runtime vs CO₂ scatter plot (log scale)

Bubble chart for emission magnitude

Regression line visualization

# 13. Results Summary
Model	Runtime (s)	CO₂ Emissions (g)	Notes
Random Forest (Air)	~1.27	6.87e-06	Best accuracy + low emissions
Random Forest (Elec)	~1.04	5.61e-06	Efficient performance
XGBoost	High	Highest emissions	Computation-heavy
MLP	Medium	Higher emissions	Many iterations
Logistic Regression	Very low	Small footprint	Simple & clean
KMeans / GMM	Very low	Minimal emissions	Lightweight models

➡ Random Forest consistently offers the best balance of sustainability and accuracy.

# 14. Streamlit Dashboard

The dashboard provides:

Emissions summary

Energy usage ranking

Runtime comparison

Emission rate visualization

Model-wise efficiency scoring

Include screenshots in GitHub:

Dashboard Home

Emissions Comparison Chart

Efficiency Ranking

Model Summary Table

# 15. Tech Stack

Python

Scikit-Learn

CodeCarbon

Pandas, NumPy

Seaborn, Matplotlib

Statsmodels

Streamlit

Jupyter Notebook

# 16. Limitations

CodeCarbon accuracy varies by hardware

CPU-only training → cannot generalize to GPU workloads

Air & electricity datasets small → limited variation

Dashboard runs locally (not deployed online)

Real-time carbon intensity not integrated

# 17. Future Scope

Add GPU-based model comparisons

Train transformer models (BERT, GPT)

Deploy Streamlit dashboard online

Integrate live carbon-intensity APIs

Add automated model selection based on emissions

Expand datasets for stronger generalisability

# 18. Author

Lakshmi Meena Manivannan
MSc Data Analytics – National College of Ireland

📧 Email: x23426918@student.ncirl.ie

🔗 LinkedIn: https://www.linkedin.com/in/lakshmi-meena-manivannan/
