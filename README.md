# 🩺 DataDoctor – Smart Data Cleaning & Data Analyzer

DataDoctor is an interactive AI-powered web application that simplifies the entire data preprocessing and analysis workflow. It is designed to help students, researchers, and data analysts clean, explore, and prepare datasets without heavy coding.

The application is built using **Streamlit** and integrates **Machine Learning**, **NLP**, and **Retrieval-Augmented Generation (RAG)** using **Gemini API + LlamaIndex** for intelligent, dataset-aware assistance.

---

## 🔍 Problem Statement

Real-world datasets are often:
- Incomplete (missing values)
- Noisy (outliers, inconsistencies)
- Redundant (duplicate rows)
- Difficult to analyze without technical skills

Manual data cleaning is time-consuming and error-prone. Existing tools lack intelligent automation and easy-to-use interfaces.

**DataDoctor automates and simplifies this process using AI/ML and interactive tools.** :contentReference[oaicite:1]{index=1}

---

## 🎯 Objectives

- Build a Streamlit-based web platform for automated data cleaning and EDA  
- Allow users to upload CSV datasets easily  
- Detect and fix:
  - Missing values  
  - Duplicates  
  - Outliers  
  - Inconsistent formats  
- Provide interactive visualizations  
- Enable feature engineering (encoding, scaling)  
- Provide a dataset-aware AI chatbot  
- Allow exporting cleaned datasets and reports :contentReference[oaicite:2]{index=2}

---

## ✨ Key Features

✅ Upload and preview CSV files  
✅ Automatic missing value handling (mean, median, mode, KNN, etc.)  
✅ Duplicate detection and removal  
✅ Outlier detection using IQR and treatment methods  
✅ Interactive EDA (heatmaps, histograms, boxplots)  
✅ Feature Engineering:
- Label Encoding  
- One-Hot Encoding  
- Scaling (StandardScaler, MinMaxScaler)  

✅ NLP text processing  
✅ RAG-powered AI chatbot (Gemini + LlamaIndex)  
✅ AI-based Data Quality Score  
✅ Export cleaned data:
- CSV
- Excel
- Summary Reports :contentReference[oaicite:3]{index=3}

---

## 🧠 Technologies Used

| Category | Tools / Libraries |
|----------|-------------------|
| Programming | Python |
| UI Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly, Seaborn, Matplotlib |
| NLP | NLTK, TextBlob |
| Vector Embeddings | HuggingFace |
| RAG Framework | LlamaIndex |
| LLM | Google Gemini API |

:contentReference[oaicite:4]{index=4}



---

## ⚙️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/sidnikam20/FinalYearProject
cd FinalYearProject
