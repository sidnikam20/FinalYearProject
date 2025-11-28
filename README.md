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

---

## 👥 Project Implementation & Contributions

This project was developed collaboratively by a team of three members, with clearly defined roles and responsibilities.

| Member        | Role                                      | Key Contributions |
|--------------|-------------------------------------------|-------------------|
| **Yogesh Dige**   | Lead Developer & System Architect | - Designed overall system architecture and data cleaning workflow<br>- Implemented core Streamlit application (`app.py`) and multi-step UI flow<br>- Integrated RAG-based Cleaning Chatbot using Gemini API + LlamaIndex<br>- Implemented data cleaning modules (missing values, duplicates, outliers, data type optimization)<br>- Developed EDA & visualization components, feature engineering, NLP text processing and export modules (CSV/Excel/report)<br>- Set up SQLite database for session storage and chat history, and handled end-to-end debugging and optimization |
| **Siddhi Nikam**  | Data Analyst & UI/UX Contributor | - Assisted in designing user flow and layout for Streamlit pages<br>- Contributed to configuration of visualization plots and summary views<br>- Helped in preparing and testing multiple sample datasets for validation of cleaning and EDA pipeline<br>- Supported literature survey mapping (Pandas Profiling, Sweetviz, RAG-based approaches) to implemented features<br>- Contributed to writing and formatting the final project report and review presentations |
| **Vaibhav Fuke**  | Testing, Evaluation & Documentation | - Performed module-wise testing of data cleaning, EDA, and export functionalities<br>- Helped verify correctness of feature engineering and text processing outputs<br>- Assisted in preparing result screenshots, workflow diagrams, and output sections for the PPT/report<br>- Contributed to writing and formatting the project synopsis and research paper <br>- Supported debugging by reporting edge cases and UI issues during trials |

---

## ⚙️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/Yogeshdige2003/DataDoctor.git
cd DataDoctor
