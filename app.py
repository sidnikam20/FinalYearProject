import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import io
import warnings
import hashlib
import re
import json
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv

# RAG Architecture Imports
from llama_index.core import VectorStoreIndex, Document, Settings, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
import google.generativeai as genai

# NLP Processing
import nltk
from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="DataDoctor - Smart Data Cleaning & Analysis with RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f8f9fa;
        color: #333;
        border: 1px solid #dee2e6;
    }
    .rag-indicator {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def show_welcome_page():
    # Style for container and Streamlit button
    st.markdown(
        """
        <style>
        .datadoctor-center-container {
            max-width: 700px;
            margin: 5% auto 2% auto;
            background: #fff;
            border-radius: 20px;
            box-shadow: 0 8px 40px rgba(102,126,234,0.12), 0 1.5px 3px #764ba2;
            padding: 40px 40px 30px 40px;
        }
        .datadoctor-title {
            text-align: center;
            font-size: 2.9rem;
            font-weight: 700;
            color: #5836b7;
            margin-bottom: 0.6em;
            letter-spacing: 1px;
        }
        .datadoctor-subtitle {
            text-align: center;
            font-size: 1.3rem;
            font-weight: 500;
            color: #4c67e2;
            margin-bottom: 1.2em;
        }
        .datadoctor-section-title {
            color:#764ba2;
            margin-top:30px;
            font-size:1.5rem;
            font-weight:700;
        }
        .datadoctor-list {
            font-size:1.16rem;
            margin: 8px 0 0 20px;
            color:#222;
        }
        .datadoctor-metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            display: inline-block;
            min-width: 150px;
            margin: 7px 8px 7px 0;
            padding: 18px 20px;
            font-size: 1.12rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(102,126,234,0.13);
            font-weight: 600;
            letter-spacing: 0.2px;
        }
        .datadoctor-guide {
            color:#636363;
            font-size:1.08rem;
            margin: 6px 0 0 18px;
            line-height:1.95;
        }
        .datadoctor-footer {
            text-align:center;
            margin:40px 0 0 0;
            color: #586fa1;
            font-size:1.10rem;
        }
        .stButton > button {
            background: linear-gradient(90deg, #6d8bdb 40%, #ea80fc 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 32px !important;
            font-size: 1.18rem !important;
            font-weight: 700 !important;
            padding: 14px 40px !important;
            margin: 18px 0 0 0 !important;
            transition: 0.18s all;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main centered HTML content
    st.markdown(
        """
        <div class="datadoctor-center-container">
            <div class="datadoctor-title">Welcome to DataDoctor</div>
            <div class="datadoctor-subtitle">
                Smart Data Cleaning & Analysis Platform<br>
                with <b>RAG Architecture</b>
            </div>
            <hr style="border:1px solid #764ba2; margin-bottom: 35px;">
            <div class="datadoctor-section-title">What is DataDoctor?</div>
            <div style="font-size:1.14rem;margin: 16px 0 16px 0; color:#222;">
                DataDoctor is an intelligent data preprocessing and analysis platform that lets you:
            </div>
            <ul class="datadoctor-list">
                <li>Clean messy datasets automatically</li>
                <li>Perform exploratory data analysis</li>
                <li>Get AI-powered data cleaning recommendations</li>
                <li>Engineer features for machine learning</li>
                <li>Process text data with NLP</li>
                <li>Export cleaned data in multiple formats</li>
            </ul>
            <div class="datadoctor-section-title" style="margin-top:26px;">Key Features:</div>
            <div class="datadoctor-metric-card">RAG Chatbot (Gemini + LlamaIndex)</div>
            <div class="datadoctor-metric-card">Automated Data Cleaning</div>
            <div class="datadoctor-metric-card">Interactive Visualizations</div>
            <div class="datadoctor-metric-card">Feature Engineering</div>
            <div class="datadoctor-metric-card">NLP Text Processing</div>
            <div class="datadoctor-section-title" style="margin-top:28px;">Quick Start Guide:</div>
            <ol class="datadoctor-guide">
                <li><b>Upload Data:</b> Start with uploading your CSV dataset</li>
                <li><b>Clean Data:</b> Use automated cleaning tools</li>
                <li><b>Explore:</b> Visualize and analyze your data</li>
                <li><b>Transform:</b> Apply feature engineering</li>
                <li><b>Ask AI:</b> Chat with RAG assistant for guidance</li>
                <li><b>Export:</b> Download cleaned data and reports</li>
            </ol>
            
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered Streamlit navigation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started with DataDoctor", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()



# Initialize database
def init_database():
    """Initialize SQLite database for storing sessions and chat history"""
    conn = sqlite3.connect('datadoctor.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cleaning_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        original_filename TEXT,
        operations_performed TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_message TEXT,
        bot_response TEXT,
        intent TEXT,
        rag_used BOOLEAN DEFAULT FALSE,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dataset_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        filename TEXT,
        metadata_json TEXT,
        embeddings_stored BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

init_database()

def unique_key(prefix: str, col_name: str, extra: str = "") -> str:
    """Generate unique keys to avoid duplicate key errors"""
    token = f"{prefix}::{col_name}::{extra}"
    return f"{prefix}_{hashlib.sha1(token.encode()).hexdigest()[:10]}"

def fix_dtypes_for_plotly(df):
    """Convert nullable dtypes to standard dtypes for Plotly compatibility"""
    df_fixed = df.copy()
    
    for col in df_fixed.columns:
        try:
            if str(df_fixed[col].dtype) in ['Int64', 'Int32', 'Int16', 'Int8']:
                df_fixed[col] = df_fixed[col].astype('int64')
            elif str(df_fixed[col].dtype) in ['Float64', 'Float32']:
                df_fixed[col] = df_fixed[col].astype('float64')
            elif str(df_fixed[col].dtype) == 'boolean':
                df_fixed[col] = df_fixed[col].astype('bool')
            elif str(df_fixed[col].dtype) == 'string':
                df_fixed[col] = df_fixed[col].astype('object')
        except Exception as e:
            continue
    
    return df_fixed

class TextProcessor:
    """Advanced text preprocessing and NLP utilities"""
    
    def __init__(self):
        self.stop_words = set()
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except:
            pass
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text == '':
            return text
        
        text = str(text)
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.lower().strip()
        
        return text
    
    def get_sentiment(self, text):
        """Get sentiment analysis of text"""
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'Positive'
            elif polarity < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        except:
            return 'Neutral'
    
    def extract_keywords(self, text, num_keywords=5):
        """Extract keywords from text"""
        try:
            blob = TextBlob(str(text))
            words = [word.lower() for word in blob.words if len(word) > 2 and word.lower() not in self.stop_words]
            
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_keywords]
            return [kw[0] for kw in keywords]
        except:
            return []

class DataQualityAssistant:
    """AI-powered data quality assessment and recommendations"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def assess_data_quality(self, df):
        """Comprehensive data quality assessment"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            
            completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
            consistency = self._calculate_consistency(df)
            validity = self._calculate_validity(df)
            
            quality_score = (completeness + consistency + validity) / 3
            
            return {
                'overall_score': round(quality_score, 2),
                'completeness': round(completeness, 2),
                'consistency': round(consistency, 2),
                'validity': round(validity, 2),
                'missing_ratio': round((missing_cells / total_cells) * 100, 2),
                'duplicate_ratio': round((duplicate_rows / df.shape[0]) * 100, 2)
            }
        except Exception as e:
            return {
                'overall_score': 0,
                'completeness': 0,
                'consistency': 0,
                'validity': 0,
                'missing_ratio': 100,
                'duplicate_ratio': 0
            }
    
    def _calculate_consistency(self, df):
        """Calculate data consistency score"""
        try:
            consistent_cols = 0
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    if not df[col].isna().all():
                        consistent_cols += 1
                elif df[col].dtype == 'object':
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        consistent_cols += 1
                else:
                    consistent_cols += 1
            
            return (consistent_cols / len(df.columns)) * 100 if len(df.columns) > 0 else 0
        except:
            return 50
    
    def _calculate_validity(self, df):
        """Calculate data validity score"""
        try:
            valid_cols = 0
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    if not df[col].isna().all():
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outlier_ratio = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]) / len(df)
                        
                        if outlier_ratio < 0.1:
                            valid_cols += 1
                        else:
                            valid_cols += 0.5
                    else:
                        valid_cols += 0.5
                else:
                    valid_cols += 1
            
            return (valid_cols / len(df.columns)) * 100 if len(df.columns) > 0 else 0
        except:
            return 75

class RAGDataCleaningAssistant:
    """RAG-powered chatbot for data cleaning assistance using Gemini API"""
    def __init__(self):
        """Initialize RAG system with Gemini API"""
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.use_rag = False
        self.knowledge_index = None
        self.query_engine = None
        
        if not self.gemini_api_key:
            st.error("🚨 Gemini API key not found! Please add GEMINI_API_KEY to your .env file")
            return
        
        try:
            # Configure Gemini API
            genai.configure(api_key=self.gemini_api_key)
            
            # Initialize LlamaIndex settings with Gemini
            Settings.llm = Gemini(
                api_key=self.gemini_api_key,
                model="models/gemini-2.0-flash",  # or your preferred model
                temperature=0.7
            )
            
            # Initialize HuggingFace embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize vector store (Chroma) with proper collection handling
            self.chroma_client = chromadb.EphemeralClient()
            
            # Generate unique collection name to avoid conflicts
            import uuid
            collection_name = f"datadoctor_knowledge_{uuid.uuid4().hex[:8]}"
            
            # Alternative: Handle existing collection properly
            try:
                # Try to delete existing collection if it exists
                existing_collections = [col.name for col in self.chroma_client.list_collections()]
                if "datadoctor_knowledge" in existing_collections:
                    self.chroma_client.delete_collection("datadoctor_knowledge")
            except Exception:
                pass  # Collection doesn't exist or can't be deleted, which is fine
            
            # Create new collection (use unique name OR the cleaned standard name)
            self.chroma_collection = self.chroma_client.create_collection("datadoctor_knowledge")
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Build knowledge base
            self._build_knowledge_base()
            self.use_rag = True
            
            st.success("🧠 RAG System initialized successfully with Gemini API!")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            st.info("💡 Falling back to rule-based responses")
            self.use_rag = False

    
    
    
    def _build_knowledge_base(self):
        """Build comprehensive knowledge base for data cleaning"""
        
        knowledge_documents = [
            Document(
                text="""
                Data Cleaning Best Practices and Methods:
                
                1. Missing Value Imputation Strategies:
                - Mean Imputation: Use for normally distributed numerical data. Best for continuous variables without outliers.
                - Median Imputation: Use for skewed numerical data or data with outliers. More robust than mean.
                - Mode Imputation: Use for categorical data or discrete numerical data.
                - KNN Imputation: Use when relationships between variables matter. Good for complex missing patterns.
                - Forward/Backward Fill: Use for time series data with temporal dependencies.
                
                Commands: "fill missing values with mean", "use median imputation", "apply KNN imputation"
                
                2. Outlier Detection and Treatment:
                - IQR Method: Standard method using Q1-1.5*IQR and Q3+1.5*IQR as boundaries
                - Z-Score Method: Use when data follows normal distribution (|z| > 3 indicates outlier)
                - Treatment: Remove (risky), Cap (set to boundaries), Transform (log/sqrt), Keep (if valid)
                
                Commands: "detect outliers", "remove outliers", "cap outliers at boundaries"
                
                3. Duplicate Handling:
                - Exact Duplicates: Rows with identical values across all columns
                - Near Duplicates: Similar rows that might represent same entity
                - Action: Remove exact duplicates, investigate near duplicates manually
                
                Commands: "remove duplicates", "find duplicate rows", "check for near duplicates"
                """,
                metadata={"category": "data_cleaning", "priority": "high"}
            ),
            
            Document(
                text="""
                Feature Engineering Techniques:
                
                1. Categorical Encoding Methods:
                - Label Encoding: Convert categories to numbers (0, 1, 2...). Use for ordinal data.
                - One-Hot Encoding: Create binary columns for each category. Use for nominal data.
                - Target Encoding: Use target variable to encode categories. Good for high-cardinality.
                - Frequency Encoding: Replace categories with their frequency counts.
                
                2. Numerical Feature Scaling:
                - StandardScaler: Transform to mean=0, std=1. Use for normally distributed data.
                - MinMaxScaler: Scale to [0,1] range. Use when you know the bounds.
                - RobustScaler: Use median and IQR, robust to outliers.
                - Normalizer: Scale individual samples to unit norm.
                
                3. Feature Creation:
                - Polynomial Features: Create x², x³, x1*x2 interactions
                - Date Features: Extract year, month, day, weekday, season
                - Text Features: Length, word count, sentiment score
                - Binning: Convert continuous to categorical (age groups, income brackets)
                
                Commands: "encode categorical variables", "scale numerical features", "create polynomial features"
                """,
                metadata={"category": "feature_engineering", "priority": "high"}
            ),
            
            Document(
                text="""
                Data Quality Assessment Framework:
                
                1. Data Quality Dimensions:
                - Completeness: Percentage of non-missing values (Target: >95% critical, >80% optional)
                - Accuracy: Correctness of values against true/expected values
                - Consistency: Uniform format and conformance to business rules
                - Validity: Data conforms to defined schema and constraints
                - Uniqueness: No unwanted duplicate records
                - Timeliness: Data is up-to-date and relevant
                
                2. Quality Metrics Calculation:
                - Completeness Score = (Total Cells - Missing Cells) / Total Cells * 100
                - Consistency Score = Consistent Columns / Total Columns * 100
                - Validity Score = Valid Values / Total Values * 100
                - Overall Score = Average of all dimension scores
                
                3. Quality Thresholds:
                - Excellent: 90-100% - Data is production ready
                - Good: 75-89% - Minor issues, mostly usable
                - Fair: 60-74% - Significant issues, needs attention
                - Poor: <60% - Major quality problems, extensive cleaning needed
                
                Commands: "assess data quality", "check completeness", "validate data consistency"
                """,
                metadata={"category": "data_quality", "priority": "high"}
            ),
            
            Document(
                text="""
                Text Data Processing and NLP:
                
                1. Text Cleaning Steps:
                - Remove HTML tags, URLs, email addresses
                - Handle special characters and punctuation
                - Normalize case (convert to lowercase)
                - Remove extra whitespace and line breaks
                - Handle encoding issues (UTF-8, ASCII)
                
                2. Text Preprocessing:
                - Tokenization: Split text into words/sentences
                - Stop Words Removal: Remove common words (the, and, or)
                - Stemming: Reduce words to root form (running → run)
                - Lemmatization: Convert to dictionary form (better → good)
                - N-gram Generation: Create word pairs/triplets
                
                3. Text Feature Extraction:
                - Sentiment Analysis: Positive, negative, neutral classification
                - Keyword Extraction: Identify important terms
                - Text Statistics: Length, word count, average word length
                - Language Detection: Identify text language
                - Topic Modeling: Discover themes in text collections
                
                Commands: "clean text data", "analyze sentiment", "extract keywords", "detect language"
                """,
                metadata={"category": "text_processing", "priority": "medium"}
            ),
            
            Document(
                text="""
                Data Cleaning Workflows and Best Practices:
                
                1. Standard Data Cleaning Pipeline:
                Step 1: Data Loading and Initial Inspection
                - Load data with proper encoding
                - Check data types and basic statistics
                - Identify potential issues early
                
                Step 2: Data Quality Assessment
                - Calculate completeness, consistency, validity scores
                - Identify missing value patterns
                - Detect outliers and anomalies
                
                Step 3: Missing Value Treatment
                - Analyze missing value patterns (MCAR, MAR, MNAR)
                - Choose appropriate imputation strategy
                - Apply imputation and validate results
                
                Step 4: Outlier Detection and Treatment
                - Use appropriate detection method (IQR, Z-score)
                - Investigate outliers (errors vs valid extreme values)
                - Apply suitable treatment method
                
                Step 5: Data Type Optimization
                - Convert strings to appropriate types (datetime, category)
                - Optimize memory usage
                - Ensure consistency across dataset
                
                Step 6: Feature Engineering
                - Encode categorical variables
                - Scale numerical features
                - Create derived features
                
                Step 7: Final Validation and Export
                - Validate cleaned data quality
                - Generate cleaning report
                - Export in required format
                
                Commands: "run standard pipeline", "start data cleaning workflow", "validate final data"
                """,
                metadata={"category": "workflows", "priority": "high"}
            ),
            
            Document(
                text="""
                Common Data Issues and Solutions:
                
                1. Data Type Issues:
                - Mixed types in columns: Convert to consistent type or split column
                - Incorrect data types: String numbers → numeric, date strings → datetime
                - Encoding problems: Handle UTF-8, Latin-1, ASCII issues
                
                2. Formatting Issues:
                - Inconsistent date formats: Standardize to single format
                - Inconsistent categories: Map variations to standard values
                - Leading/trailing spaces: Strip whitespace
                - Case inconsistency: Standardize to upper/lower case
                
                3. Value Issues:
                - Invalid values: Replace with NaN or correct value
                - Out of range values: Cap or remove based on business rules
                - Placeholder values: Treat as missing (9999, -1, 'N/A')
                
                4. Structure Issues:
                - Wrong delimiter: Re-read with correct separator
                - Header issues: Fix column names
                - Merged cells: Unmerge and fill properly
                
                Solutions:
                - Use pandas.to_numeric() with errors='coerce' for type conversion
                - Apply regex patterns for consistent formatting
                - Use replace() methods for value standardization
                - Validate data against business rules
                
                Commands: "fix data types", "standardize formats", "handle invalid values"
                """,
                metadata={"category": "troubleshooting", "priority": "medium"}
            )
        ]
        
        try:
            # Create vector index from documents
            self.knowledge_index = VectorStoreIndex.from_documents(
                knowledge_documents,
                storage_context=self.storage_context
            )
            
            # Create query engine with custom settings
            self.query_engine = self.knowledge_index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact",
                streaming=False
            )
            
            st.success("📚 Knowledge base built successfully with {} documents".format(len(knowledge_documents)))
            
        except Exception as e:
            st.error(f"❌ Failed to build knowledge base: {str(e)}")
            self.use_rag = False
    
    def add_dataset_context(self, df, filename="dataset"):
        """Add current dataset metadata to RAG knowledge base"""
        try:
            # Generate comprehensive dataset metadata
            dataset_metadata = self._generate_dataset_metadata(df, filename)
            
            # Create dataset document
            dataset_doc = Document(
                text=dataset_metadata,
                metadata={"category": "current_dataset", "filename": filename, "timestamp": str(datetime.now())}
            )
            
            # Add to existing index
            if self.knowledge_index:
                self.knowledge_index.insert(dataset_doc)
                st.info(f"📊 Added dataset context for {filename} to knowledge base")
            
            # Store in database
            self._store_dataset_metadata(df, filename, dataset_metadata)
            
        except Exception as e:
            st.warning(f"⚠️ Could not add dataset context: {str(e)}")
    
    def _generate_dataset_metadata(self, df, filename):
        """Generate comprehensive dataset metadata"""
        try:
            # Basic statistics
            metadata = f"""
Dataset Analysis Report: {filename}

BASIC INFORMATION:
- Dataset Name: {filename}
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- Missing Values: {df.isnull().sum().sum()} total ({df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%)
- Duplicate Rows: {df.duplicated().sum()} ({df.duplicated().sum()/df.shape[0]*100:.1f}%)

COLUMN ANALYSIS:
"""
            
            for col in df.columns:
                col_metadata = f"""
Column: {col}
- Data Type: {df[col].dtype}
- Missing Values: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)
- Unique Values: {df[col].nunique()} ({df[col].nunique()/len(df)*100:.1f}% cardinality)
"""
                
                if df[col].dtype in ['int64', 'float64']:
                    try:
                        col_metadata += f"""
- Statistical Summary:
  * Mean: {df[col].mean():.2f}
  * Median: {df[col].median():.2f}
  * Std Dev: {df[col].std():.2f}
  * Range: {df[col].min():.2f} to {df[col].max():.2f}
  * Potential Outliers: {self._count_outliers(df[col])} values
"""
                    except:
                        pass
                elif df[col].dtype == 'object':
                    try:
                        top_values = df[col].value_counts().head(5)
                        col_metadata += f"""
- Top Values: {', '.join([f'{val}({count})' for val, count in top_values.items()])}
- Average Length: {df[col].astype(str).str.len().mean():.1f} characters
"""
                    except:
                        pass
                
                metadata += col_metadata
            
            # Data quality assessment
            quality_assistant = DataQualityAssistant()
            quality_metrics = quality_assistant.assess_data_quality(df)
            
            metadata += f"""

DATA QUALITY ASSESSMENT:
- Overall Quality Score: {quality_metrics['overall_score']}/100
- Completeness: {quality_metrics['completeness']}%
- Consistency: {quality_metrics['consistency']}%
- Validity: {quality_metrics['validity']}%

RECOMMENDATIONS:
{self._generate_cleaning_recommendations(df, quality_metrics)}
"""
            
            return metadata
            
        except Exception as e:
            return f"Error generating dataset metadata: {str(e)}"
    
    def _count_outliers(self, series):
        """Count outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
            return len(outliers)
        except:
            return 0
    
    def _generate_cleaning_recommendations(self, df, quality_metrics):
        """Generate specific cleaning recommendations"""
        recommendations = []
        
        try:
            # Missing value recommendations
            if quality_metrics['completeness'] < 90:
                missing_cols = df.columns[df.isnull().any()].tolist()
                recommendations.append(f"High missing data detected in {len(missing_cols)} columns: {', '.join(missing_cols[:5])}. Consider appropriate imputation strategies.")
            
            # Duplicate recommendations
            if df.duplicated().sum() > 0:
                recommendations.append(f"Found {df.duplicated().sum()} duplicate rows. Consider removing duplicates unless they represent valid repeated observations.")
            
            # Outlier recommendations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            high_outlier_cols = []
            for col in numeric_cols:
                outlier_count = self._count_outliers(df[col])
                if outlier_count > len(df) * 0.05:  # More than 5% outliers
                    high_outlier_cols.append(col)
            
            if high_outlier_cols:
                recommendations.append(f"High outlier concentration in columns: {', '.join(high_outlier_cols[:3])}. Investigate whether these are data errors or valid extreme values.")
            
            # Data type recommendations
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                if 'date' in col.lower() or 'time' in col.lower():
                    recommendations.append(f"Column '{col}' appears to contain date/time data. Consider converting to datetime type.")
                elif df[col].nunique() / len(df) < 0.05:
                    recommendations.append(f"Column '{col}' has low cardinality ({df[col].nunique()} unique values). Consider converting to categorical type.")
            
            return '\n'.join([f"- {rec}" for rec in recommendations]) if recommendations else "- Dataset quality is good. No major issues detected."
            
        except Exception as e:
            return f"- Error generating recommendations: {str(e)}"
    
    def _store_dataset_metadata(self, df, filename, metadata):
        """Store dataset metadata in database"""
        try:
            conn = sqlite3.connect('datadoctor.db')
            cursor = conn.cursor()
            
            session_id = st.session_state.get('session_id', 'default')
            metadata_json = json.dumps({
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'metadata_text': metadata
            })
            
            cursor.execute('''
            INSERT INTO dataset_metadata (session_id, filename, metadata_json, embeddings_stored)
            VALUES (?, ?, ?, ?)
            ''', (session_id, filename, metadata_json, True))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            st.warning(f"Could not store metadata: {str(e)}")
    
    def query_rag(self, user_question, df=None, filename=None):
        """Query RAG system for intelligent responses"""
        if not self.use_rag or not self.query_engine:
            return self._fallback_response(user_question)
        
        try:
            # Add current dataset context if available
            if df is not None:
                self.add_dataset_context(df, filename or "current_dataset")
            
            # Enhance query with context
            enhanced_query = f"""
User Question: {user_question}

Context: You are DataDoctor AI, an expert data cleaning and analysis assistant. 
Provide specific, actionable advice based on the user's question and current dataset context.

Instructions:
1. Be concise but comprehensive
2. Provide step-by-step guidance when appropriate
3. Reference specific columns or data characteristics when relevant
4. Suggest specific commands or functions to use
5. Explain the reasoning behind recommendations
6. If the question is about the current dataset, use the dataset metadata

User Question: {user_question}
"""
            
            # Query the RAG system
            with st.spinner("🧠 Processing with RAG..."):
                response = self.query_engine.query(enhanced_query)
            
            return str(response)
            
        except Exception as e:
            st.error(f"❌ RAG query failed: {str(e)}")
            return self._fallback_response(user_question)
    
    def _fallback_response(self, question):
        """Fallback response when RAG is unavailable"""
        return f"I understand you're asking about: '{question}'. While my advanced RAG system is temporarily unavailable, I can still help with basic data cleaning guidance. Please try rephrasing your question or check the specific data cleaning sections in the app."

class IntelligentChatbot:
    """Enhanced chatbot with hybrid RAG/Direct API capabilities"""
    
    def __init__(self):
        self.rag_assistant = RAGDataCleaningAssistant()
        self.direct_gemini = None
        
        # Initialize direct Gemini client for general questions
        try:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                self.direct_gemini = genai.GenerativeModel('models/gemini-2.0-flash')
        except Exception as e:
            st.warning(f"Could not initialize direct Gemini: {e}")
        
        # Fallback intent patterns
        self.intent_patterns = {
            'upload': ['upload', 'load', 'import', 'file', 'csv'],
            'clean': ['clean', 'fix', 'handle', 'missing', 'duplicate', 'outlier'],
            'visualize': ['plot', 'chart', 'graph', 'visualize', 'show', 'display'],
            'export': ['download', 'export', 'save', 'output'],
            'help': ['help', 'how', 'what', 'explain', 'guide'],
            'quality': ['quality', 'score', 'assess', 'evaluate'],
            'encoding': ['encode', 'transform', 'categorical', 'dummy'],
            'scaling': ['scale', 'normalize', 'standardize']
        }
    
    def classify_question_type(self, message):
        """Classify if question needs RAG or can use direct Gemini"""
        
        # Dataset-specific keywords that REQUIRE RAG
        dataset_keywords = ['my dataset', 'current dataset', 'this dataset', 'uploaded data', 
                        'current data', 'my data', 'dataset quality', 'assess my data',
                        'clean my data', 'my columns', 'this column', 'current column']
        
        # Data cleaning action keywords that need RAG with current data context
        cleaning_action_keywords = ['handle missing values in', 'remove outliers from', 
                                'clean column', 'impute values in', 'assess quality of my',
                                'duplicate rows in my', 'scale my features']
        
        # General data science questions that should use Direct Gemini
        general_keywords = ['what is', 'what are', 'define', 'explain', 'difference between',
                        'how does', 'why is', 'advantages of', 'disadvantages of',
                        'when to use', 'types of', 'methods of', 'concept of',
                        'benefits of', 'drawbacks of', 'comparison', 'versus', 'vs']
        
        message_lower = message.lower()
        
        # First check for general questions
        if any(keyword in message_lower for keyword in general_keywords):
            # Double check it's not about current dataset  
            if any(keyword in message_lower for keyword in dataset_keywords):
                return "rag"  # It's about current dataset, use RAG
            else:
                return "direct"  # It's a general question, use direct Gemini
        
        # Check for dataset-specific or cleaning action keywords
        if any(keyword in message_lower for keyword in dataset_keywords + cleaning_action_keywords):
            return "rag"  # Use RAG for dataset-specific questions
        
        # For ambiguous cases, default to direct Gemini for better answers
        return "direct"

    
    def generate_response(self, message, current_data=None, filename=None):
        """Generate intelligent response using hybrid approach"""
        
        message_lower = message.lower()
        
        # Simple check: if it's a general "what is" question, use Direct Gemini
        general_patterns = ['what is ', 'what are ', 'define ', 'explain ', 'difference between', 
                        'compare ', 'versus', ' vs ', 'advantages of', 'disadvantages of',
                        'when to use', 'how does', 'types of']
        
        # Dataset-specific patterns that need RAG
        dataset_patterns = ['my dataset', 'current dataset', 'this dataset', 'my data', 
                        'assess my', 'clean my', 'handle missing', 'remove outliers']
        
        # Check for dataset-specific questions first (higher priority)
        use_rag = any(pattern in message_lower for pattern in dataset_patterns)
        
        # Check for general questions (only if not dataset-specific)
        use_direct = (not use_rag) and any(pattern in message_lower for pattern in general_patterns)
        
        # DEBUG: Show which system is being used
        if use_direct:
            st.info("🤖 **Using Direct Gemini** for general question")
        elif use_rag:
            st.info("🧠 **Using RAG System** for dataset-specific question")
        else:
            st.info("🔄 **Using Default** - will try Direct Gemini")
        
        # Try Direct Gemini for general questions
        if (use_direct or not use_rag) and self.direct_gemini:
            try:
                prompt = f"""
                        You are DataDoctor AI, a helpful data science assistant.
                        Please provide a comprehensive, clear answer to: {message}

                        Be educational and include examples if helpful. Use simple language but be technically accurate.
                        """
                response = self.direct_gemini.generate_content(prompt)
                return response.text
                
            except Exception as e:
                st.warning(f"Direct Gemini failed: {e}")
                # Fall back to RAG
                if self.rag_assistant.use_rag:
                    return self.rag_assistant.query_rag(message, current_data, filename)
        
        # Use RAG for dataset-specific questions or as fallback
        if self.rag_assistant.use_rag:
            return self.rag_assistant.query_rag(message, current_data, filename)
        
        # Final fallback
        return self._rule_based_response(message, current_data)

    def _rule_based_response(self, message, current_data):
        """Rule-based fallback response"""
        message_lower = message.lower()
        
        # Simple intent classification
        if any(keyword in message_lower for keyword in self.intent_patterns['clean']):
            response = "I can help you clean your data! Common cleaning tasks include handling missing values, removing duplicates, and treating outliers."
        elif any(keyword in message_lower for keyword in self.intent_patterns['visualize']):
            response = "For data visualization, check the 'EDA & Visualization' section where you can create histograms, correlation heatmaps, and more!"
        elif any(keyword in message_lower for keyword in self.intent_patterns['quality']):
            response = "I can assess data quality by checking completeness, consistency, and validity metrics."
        else:
            response = "I'm here to help with data cleaning and analysis. What specific task would you like assistance with?"
        
        # Add dataset context if available
        if current_data is not None:
            df = current_data
            response += f"\n\n📊 Current Dataset: {df.shape[0]} rows, {df.shape[1]} columns, {df.isnull().sum().sum()} missing values"
        
        return response
    
    def save_chat_history(self, session_id, user_message, bot_response, rag_used=False):
        """Save chat interaction to database"""
        try:
            conn = sqlite3.connect('datadoctor.db')
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO chat_history (session_id, user_message, bot_response, rag_used)
            VALUES (?, ?, ?, ?)
            ''', (session_id, user_message, bot_response, rag_used))
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.warning(f"Could not save chat history: {str(e)}")


def display_chatbot():
    """Display RAG-powered chatbot interface"""
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'rag_chatbot' not in st.session_state:
        st.session_state.rag_chatbot = IntelligentChatbot()
    
    # Floating chat button with RAG indicator
    col1, col2, col3, col4, col5 = st.columns([3.5, 1, 1, 1, 0.5])
    with col5:
        rag_status = "🧠" if st.session_state.rag_chatbot.rag_assistant.use_rag else "🤖"
        if st.button(f"{rag_status} RAG AI", key="chat_toggle", help="RAG-powered AI Assistant"):
            st.session_state.show_chat = not st.session_state.show_chat
    
    # Chat interface
    if st.session_state.show_chat:
        st.markdown("---")
        
        # Chat header with RAG status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🧠 DataDoctor RAG Assistant")
        with col2:
            if st.session_state.rag_chatbot.rag_assistant.use_rag:
                st.markdown('<p class="rag-indicator">🟢 RAG Active</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: orange;">🟡 Fallback Mode</p>', unsafe_allow_html=True)
        
        st.caption("Powered by Retrieval-Augmented Generation with Gemini API & LlamaIndex")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for i, chat in enumerate(st.session_state.chat_history[-10:]):  # Show last 10 messages
                    if chat['type'] == 'user':
                        st.markdown(f"**👤 You:** {chat['message']}")
                    else:
                        rag_indicator = "🧠" if chat.get('rag_used', False) else "🤖"
                        st.markdown(f"**{rag_indicator} DataDoctor:** {chat['message']}")
                    st.markdown("---")
        
        # Chat input area
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Ask me anything about data cleaning and analysis:",
                key="chat_input",
                placeholder="e.g., 'How should I handle missing values in my dataset?'"
            )
        with col2:
            send_button = st.button("Send 📤", key="send_message", type="primary")
        
        # Quick suggestion buttons
        st.write("💡 **Quick Questions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Assess Quality", key="quick_quality"):
                user_input = "Please assess the quality of my current dataset"
                send_button = True
        with col2:
            if st.button("Handle Missing", key="quick_missing"):
                user_input = "How should I handle missing values in my dataset?"
                send_button = True
        with col3:
            if st.button("Remove Outliers", key="quick_outliers"):
                user_input = "How can I detect and remove outliers?"
                send_button = True
        with col4:
            if st.button("Feature Engineering", key="quick_features"):
                user_input = "What feature engineering should I apply?"
                send_button = True
        
        # Process user input
        if send_button and user_input:
            # Get current data context
            current_data = st.session_state.get('current_data', None)
            filename = st.session_state.get('uploaded_filename', 'current_dataset')
            
            # Show typing indicator
            # Generate response with type detection
            question_type = st.session_state.rag_chatbot.classify_question_type(user_input)

            with st.spinner(f"🧠 {'RAG AI' if question_type == 'rag' else 'Direct AI'} is thinking..."):
                response = st.session_state.rag_chatbot.generate_response(
                    user_input, 
                    current_data, 
                    filename
                )

            # Determine which system was used
            rag_used = (question_type == "rag" and st.session_state.rag_chatbot.rag_assistant.use_rag)
            direct_used = (question_type == "direct" and st.session_state.rag_chatbot.direct_gemini)

            # Add to chat history
            st.session_state.chat_history.append({
                'type': 'user', 
                'message': user_input,
                'rag_used': False
            })
            st.session_state.chat_history.append({
                'type': 'bot', 
                'message': response,
                'rag_used': rag_used
            })
            
            # Save to database
            session_id = st.session_state.get('session_id', 'default')
            st.session_state.rag_chatbot.save_chat_history(session_id, user_input, response, rag_used)
            
            # Clear input and rerun
            st.rerun()

# Rest of your existing functions (main, upload_data, data_cleaning, etc.) remain the same
# Just replace the chatbot initialization calls

def main():
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True
    
    if st.session_state.show_welcome:
        show_welcome_page()
        return
    # Initialize session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:10]
    
    # Main header
    st.markdown('<h1 class="main-header">🩺 DataDoctor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Smart Data Cleaning & Analysis Platform with RAG Architecture</p>', unsafe_allow_html=True)
    
    # RAG status indicator
    # RAG status indicator
    if 'rag_chatbot' not in st.session_state:
        st.session_state.rag_chatbot = IntelligentChatbot()

    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Fix dtypes globally when data is loaded
    if 'current_data' in st.session_state:
        try:
            df = st.session_state['current_data']
            st.session_state['current_data'] = fix_dtypes_for_plotly(df)
        except:
            pass
    
    # Display chatbot
    display_chatbot()
    
    # Sidebar for navigation (rest of your existing sidebar code)
    with st.sidebar:
        st.title("🧭 Navigation")
        st.markdown("---")
        
        # RAG System Status
        # Enhanced AI System Status
        st.subheader("🧠 AI System Status")
        if st.session_state.rag_chatbot.rag_assistant.use_rag:
            st.success("🟢 RAG System Active")
            st.caption("Powered by Gemini API + LlamaIndex")
        else:
            st.warning("🟡 RAG Offline")
            st.caption("Check your Gemini API key")

        if st.session_state.rag_chatbot.direct_gemini:
            st.success("🟢 Direct Gemini Active")
            st.caption("For general questions")
        else:
            st.warning("🟡 Direct API Offline")
            st.caption("Using fallback responses")

    
        
        st.markdown("---")
        
        # Progress indicator
        st.subheader("📊 Progress")
        progress_value = (st.session_state.step - 1) * 16.66
        st.progress(progress_value / 100)
        st.write(f"Step {st.session_state.step} of 6")
        
        st.markdown("---")
        
        option = st.selectbox(
            "🔍 Choose Operation:",
            ["1. 📁 Upload Data", "2. 🧹 Data Cleaning", "3. 📈 EDA & Visualization", 
             "4. 🔧 Feature Engineering", "5. 📝 Text Processing", "6. 💾 Export Results"],
            index=st.session_state.step - 1
        )
        
        # Update step based on selection
        st.session_state.step = int(option.split('.')[0])
        
        st.markdown("---")
        
        # AI Quality Assessment with RAG
        if 'current_data' in st.session_state:
            st.subheader("🤖 AI Quality Assessment")
            if st.button("📊 RAG Analysis"):
                df = st.session_state['current_data']
                filename = st.session_state.get('uploaded_filename', 'current_dataset')
                
                # Use RAG for quality assessment
                with st.spinner("🧠 RAG Analysis..."):
                    quality_query = "Please provide a comprehensive quality assessment of my current dataset including specific recommendations for improvement."
                    assessment = st.session_state.rag_chatbot.generate_response(quality_query, df, filename)
                
                st.info("🧠 **RAG Assessment:**")
                st.write(assessment)
        
        # Dataset info sidebar (existing code)
        if 'current_data' in st.session_state:
            try:
                df = st.session_state['current_data']
                st.subheader("📋 Dataset Info")
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                st.metric("Missing Values", df.isnull().sum().sum())
                
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory (MB)", f"{memory_usage:.2f}")
            except:
                st.subheader("📋 Dataset Info")
                st.write("No data loaded")
        else:
            st.subheader("📋 Dataset Info")
            st.write("No data loaded")
    
    # Route to appropriate function based on selection
    if st.session_state.step == 1:
        upload_data()
    elif st.session_state.step == 2:
        data_cleaning()
    elif st.session_state.step == 3:
        data_visualization()
    elif st.session_state.step == 4:
        feature_engineering()
    elif st.session_state.step == 5:
        text_processing()
    elif st.session_state.step == 6:
        export_results()

# Add all your existing functions here (upload_data, data_cleaning, data_visualization, 
# feature_engineering, text_processing, export_results, etc.)
# They remain exactly the same as before

def upload_data():
    st.header("📁 Data Upload & Initial Exploration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your CSV dataset for cleaning and analysis"
        )
        
        if uploaded_file is not None:
            try:
                encoding_option = st.selectbox("Select file encoding:", ["utf-8", "latin-1", "cp1252"], index=0)
                
                df = pd.read_csv(uploaded_file, encoding=encoding_option)
                df = fix_dtypes_for_plotly(df)
                
                st.session_state['original_data'] = df.copy()
                st.session_state['current_data'] = df.copy()
                st.session_state['uploaded_filename'] = uploaded_file.name
                
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                
                # RAG-powered initial analysis
                if st.session_state.rag_chatbot.rag_assistant.use_rag:
                    with st.spinner("🧠 RAG analyzing your dataset..."):
                        initial_analysis = st.session_state.rag_chatbot.generate_response(
                            "Please provide an initial analysis of this newly uploaded dataset and suggest the first steps for data cleaning.",
                            df,
                            uploaded_file.name
                        )
                    
                    st.info("🧠 **RAG Initial Analysis:**")
                    st.write(initial_analysis)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("Try selecting a different encoding option.")
                return
    
    with col2:
        if 'current_data' in st.session_state:
            df = st.session_state['current_data']
            st.subheader("📊 Quick Stats")
            st.markdown(f"""
            <div class="metric-card">
                <h3>{df.shape[0]:,}</h3>
                <p>Rows</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{df.shape[1]:,}</h3>
                <p>Columns</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Rest of upload_data function remains the same...
    if 'current_data' in st.session_state:
        df = st.session_state['current_data']
        
        st.markdown("---")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("🔍 Column Information")
            try:
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Missing': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum() / len(df) * 100).round(1).values
                })
                st.dataframe(col_info, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying column info: {str(e)}")
            
            # RAG-powered quality overview
            st.subheader("🧠 RAG Quality Assessment")
            if st.button("Get RAG Assessment", key="rag_quality_upload"):
                with st.spinner("🧠 Analyzing with RAG..."):
                    quality_assessment = st.session_state.rag_chatbot.generate_response(
                        "Assess the data quality of this dataset and provide a quality score with specific recommendations.",
                        df,
                        st.session_state.get('uploaded_filename', 'dataset')
                    )
                st.info("🧠 **RAG Assessment:**")
                st.write(quality_assessment)
        
        if st.button("🚀 Proceed to Data Cleaning", type="primary", key="proceed_cleaning_btn"):
            st.session_state.step = 2
            st.rerun()



def data_cleaning():
    if 'current_data' not in st.session_state:
        st.warning("Please upload a dataset first!")
        return
    
    st.header("🧹 Advanced Data Cleaning")
    
    df = st.session_state['current_data']
    df = fix_dtypes_for_plotly(df)
    
    # Cleaning options tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Missing Values", "🗂️ Duplicates", "📊 Outliers", "🔧 Data Types"])
    
    with tab1:
        st.subheader("Missing Values Treatment")
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            st.success("🎉 No missing values found!")
        else:
            st.write(f"**Found missing values in {len(missing_cols)} columns:**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if len(missing_cols) > 0:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(df[missing_cols].isnull(), cbar=True, yticklabels=False, cmap='viridis')
                        plt.title("Missing Values Heatmap")
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.warning(f"Could not generate heatmap: {str(e)}")
            
            with col2:
                st.write("**Missing Value Summary:**")
                missing_summary = pd.DataFrame({
                    'Column': missing_cols,
                    'Missing Count': [df[col].isnull().sum() for col in missing_cols],
                    'Missing %': [(df[col].isnull().sum() / len(df) * 100).round(1) for col in missing_cols]
                })
                st.dataframe(missing_summary, use_container_width=True)
            
            st.subheader("🎯 Column-wise Imputation")
            
            imputation_methods = {}
            
            for idx, col in enumerate(missing_cols):
                with st.expander(f"Configure imputation for: **{col}** ({df[col].dtype})"):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**Missing:** {df[col].isnull().sum()} ({(df[col].isnull().sum()/len(df)*100):.1f}%)")
                        if df[col].dtype in ['object', 'category']:
                            st.write(f"**Unique values:** {df[col].nunique()}")
                        else:
                            try:
                                st.write(f"**Range:** {df[col].min():.2f} - {df[col].max():.2f}")
                            except:
                                st.write("**Range:** Cannot calculate")
                    
                    with col_info2:
                        if df[col].dtype in ['object', 'category']:
                            method_options = ["Skip", "Drop Rows", "Fill with Mode", "Fill with Custom Value"]
                        else:
                            method_options = ["Skip", "Drop Rows", "Fill with Mean", "Fill with Median", 
                                            "Fill with Mode", "Forward Fill", "Backward Fill", "KNN Imputation", "Fill with Custom Value"]
                        
                        method_key = unique_key("method", str(col), str(idx))
                        method = st.selectbox(f"Method for {col}:", method_options, key=method_key)
                        
                        if method == "Fill with Custom Value":
                            custom_key = unique_key("custom", str(col), str(idx))
                            custom_val = st.text_input(f"Custom value for {col}:", key=custom_key)
                            imputation_methods[col] = ("custom", custom_val)
                        elif method == "KNN Imputation":
                            knn_key = unique_key("knn", str(col), str(idx))
                            k_neighbors = st.slider(f"K neighbors for {col}:", 1, 10, 5, key=knn_key)
                            imputation_methods[col] = ("knn", k_neighbors)
                        else:
                            imputation_methods[col] = (method.lower().replace(" ", "_"), None)
            
            if st.button("🔧 Apply Missing Value Treatment", type="primary", key="apply_imputation_btn"):
                df_cleaned = apply_missing_value_treatment(df, imputation_methods)
                if df_cleaned is not None:
                    st.session_state['current_data'] = fix_dtypes_for_plotly(df_cleaned)
                    st.success("✅ Missing value treatment applied successfully!")
                    st.rerun()
    
    with tab2:
        st.subheader("Duplicate Row Management")
        
        try:
            duplicate_count = df.duplicated().sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Duplicate Rows", duplicate_count)
            with col2:
                if duplicate_count > 0:
                    if st.button("🗑️ Remove Duplicates", key="remove_duplicates_btn"):
                        df_no_duplicates = df.drop_duplicates()
                        st.session_state['current_data'] = fix_dtypes_for_plotly(df_no_duplicates)
                        st.success(f"✅ Removed {duplicate_count} duplicate rows!")
                        st.rerun()
                else:
                    st.success("No duplicates found!")
            
            if duplicate_count > 0:
                st.write("**Sample duplicate rows:**")
                st.dataframe(df[df.duplicated()].head())
        except Exception as e:
            st.error(f"Error in duplicate detection: {str(e)}")
    
    with tab3:
        st.subheader("Outlier Detection & Treatment")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns found for outlier detection.")
        else:
            selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols, key="outlier_col_select")
            
            if selected_col:
                try:
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Outliers Detected", len(outliers))
                        
                        try:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.boxplot(df[selected_col].dropna())
                            ax.set_title(f"Box Plot: {selected_col}")
                            ax.set_ylabel(selected_col)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.warning(f"Could not create box plot: {str(e)}")
                    
                    with col2:
                        if len(outliers) > 0:
                            outlier_treatment = st.selectbox(
                                "Choose outlier treatment:",
                                ["None", "Remove Outliers", "Cap at Bounds", "Transform to Bounds"],
                                key="outlier_treatment_select"
                            )
                            
                            if st.button("Apply Outlier Treatment", key="apply_outlier_btn") and outlier_treatment != "None":
                                df_treated = treat_outliers(df, selected_col, outlier_treatment, lower_bound, upper_bound)
                                st.session_state['current_data'] = fix_dtypes_for_plotly(df_treated)
                                st.success(f"✅ Applied {outlier_treatment} to {selected_col}")
                                st.rerun()
                        else:
                            st.success("No outliers detected!")
                except Exception as e:
                    st.error(f"Error in outlier analysis: {str(e)}")
    
    with tab4:
        st.subheader("Data Type Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Data Types:**")
            try:
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Current Type': df.dtypes.astype(str),
                    'Memory Usage (KB)': (df.memory_usage(deep=True)[1:] / 1024).round(2).values
                })
                st.dataframe(dtype_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying data types: {str(e)}")
        
        with col2:
            st.write("**Type Conversion Options:**")
            for idx, col in enumerate(df.columns):
                if df[col].dtype == 'object':
                    try:
                        if col.lower() in ['date', 'time', 'created', 'updated'] or 'date' in col.lower():
                            dt_key = unique_key("dt", str(col), str(idx))
                            if st.button(f"Convert {col} to datetime", key=dt_key):
                                try:
                                    df[col] = pd.to_datetime(df[col])
                                    st.session_state['current_data'] = fix_dtypes_for_plotly(df)
                                    st.success(f"✅ Converted {col} to datetime")
                                    st.rerun()
                                except:
                                    st.error(f"Failed to convert {col} to datetime")
                        
                        if df[col].nunique() / len(df) < 0.05:
                            cat_key = unique_key("cat", str(col), str(idx))
                            if st.button(f"Convert {col} to category", key=cat_key):
                                df[col] = df[col].astype('category')
                                st.session_state['current_data'] = fix_dtypes_for_plotly(df)
                                st.success(f"✅ Converted {col} to category")
                                st.rerun()
                    except:
                        pass
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Upload", key="back_to_upload_btn"):
            st.session_state.step = 1
            st.rerun()
    with col3:
        if st.button("➡️ Proceed to EDA", type="primary", key="proceed_to_eda_btn"):
            st.session_state.step = 3
            st.rerun()

def apply_missing_value_treatment(df, imputation_methods):
    """Apply various imputation methods based on user choices"""
    try:
        df_cleaned = df.copy()
        
        for col, (method, param) in imputation_methods.items():
            if method == "skip":
                continue
            elif method == "drop_rows":
                df_cleaned = df_cleaned.dropna(subset=[col])
            elif method == "fill_with_mean":
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif method == "fill_with_median":
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            elif method == "fill_with_mode":
                mode_val = df_cleaned[col].mode()
                if len(mode_val) > 0:
                    df_cleaned[col].fillna(mode_val[0], inplace=True)
            elif method == "forward_fill":
                df_cleaned[col].fillna(method='ffill', inplace=True)
            elif method == "backward_fill":
                df_cleaned[col].fillna(method='bfill', inplace=True)
            elif method == "knn_imputation":
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    imputer = KNNImputer(n_neighbors=param)
                    df_cleaned[[col]] = imputer.fit_transform(df_cleaned[[col]])
            elif method == "custom":
                if param:
                    df_cleaned[col].fillna(param, inplace=True)
        
        return df_cleaned
    except Exception as e:
        st.error(f"Error applying imputation: {str(e)}")
        return None

def treat_outliers(df, column, treatment, lower_bound, upper_bound):
    """Apply outlier treatment methods"""
    df_treated = df.copy()
    
    try:
        if treatment == "Remove Outliers":
            df_treated = df_treated[(df_treated[column] >= lower_bound) & (df_treated[column] <= upper_bound)]
        elif treatment == "Cap at Bounds":
            df_treated[column] = df_treated[column].clip(lower=lower_bound, upper=upper_bound)
        elif treatment == "Transform to Bounds":
            df_treated.loc[df_treated[column] < lower_bound, column] = lower_bound
            df_treated.loc[df_treated[column] > upper_bound, column] = upper_bound
    except Exception as e:
        st.error(f"Error in outlier treatment: {str(e)}")
    
    return df_treated

def data_visualization():
    if 'current_data' not in st.session_state:
        st.warning("Please upload a dataset first!")
        return
    
    st.header("📈 Exploratory Data Analysis & Visualization")
    
    df = st.session_state['current_data']
    df = fix_dtypes_for_plotly(df)
    st.session_state['current_data'] = df
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Distributions", "🔗 Correlations", "📋 Summary Report"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
        with col5:
            categorical_cols = len(df.select_dtypes(exclude=[np.number]).columns)
            st.metric("Categorical Columns", categorical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types Distribution")
            try:
                dtype_counts = df.dtypes.astype(str).value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
                ax.set_title("Data Types Distribution")
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error creating pie chart: {str(e)}")
                st.write("**Data Types:**")
                st.dataframe(pd.DataFrame({'Count': dtype_counts}))
        
        with col2:
            st.subheader("Missing Values by Column")
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(range(len(missing_data)), missing_data.values)
                    ax.set_xticks(range(len(missing_data)))
                    ax.set_xticklabels(missing_data.index, rotation=45, ha='right')
                    ax.set_title("Missing Values Count by Column")
                    ax.set_ylabel("Count")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating bar chart: {str(e)}")
                    st.dataframe(pd.DataFrame({'Missing Count': missing_data}))
            else:
                st.success("No missing values in any column!")

    with tab2:
        st.subheader("Data Distributions")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if numeric_cols:
            st.write("**Numeric Columns:**")
            selected_numeric = st.multiselect(
                "Select numeric columns to visualize:", 
                numeric_cols, 
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                key="numeric_cols_multiselect"
            )
            
            if selected_numeric:
                for col in selected_numeric:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        data_clean = df[col].dropna()
                        if len(data_clean) > 0:
                            ax.hist(data_clean, bins=30, alpha=0.7, edgecolor='black')
                            ax.set_title(f"Distribution of {col}")
                            ax.set_xlabel(col)
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.warning(f"No data to plot for {col}")
                    except Exception as e:
                        st.error(f"Could not plot {col}: {str(e)}")
        
        if categorical_cols:
            st.write("**Categorical Columns:**")
            selected_cat = st.selectbox("Select a categorical column:", categorical_cols, key="categorical_col_select")
            
            if selected_cat:
                try:
                    value_counts = df[selected_cat].value_counts().head(15)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(len(value_counts)), value_counts.values)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    ax.set_title(f"Distribution of {selected_cat}")
                    ax.set_ylabel("Count")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating categorical plot: {str(e)}")
                    st.dataframe(df[selected_cat].value_counts().head(15))

    with tab3:
        st.subheader("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation analysis.")
        else:
            try:
                corr_matrix = df[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax)
                ax.set_title("Correlation Matrix Heatmap")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.subheader("High Correlation Pairs")
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': round(float(corr_val), 3)
                            })
                
                if high_corr_pairs:
                    st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
                else:
                    st.info("No highly correlated variable pairs found (threshold: 0.7)")
                    
            except Exception as e:
                st.error(f"Error in correlation analysis: {str(e)}")

    with tab4:
        st.subheader("📋 Automated Summary Report")
        
        try:
            summary_report = generate_summary_report(df)
            
            for section_title, content in summary_report.items():
                st.write(f"**{section_title}:**")
                if isinstance(content, pd.DataFrame):
                    st.dataframe(content, use_container_width=True)
                else:
                    st.write(content)
                st.write("---")
        except Exception as e:
            st.error(f"Error generating summary report: {str(e)}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Cleaning", key="back_to_cleaning_btn"):
            st.session_state.step = 2
            st.rerun()
    with col3:
        if st.button("➡️ Feature Engineering", type="primary", key="proceed_to_feature_btn"):
            st.session_state.step = 4
            st.rerun()

def generate_summary_report(df):
    """Generate automated EDA summary report"""
    report = {}
    
    try:
        report["Dataset Overview"] = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Memory Usage (MB)', 'Duplicates'],
            'Value': [
                f"{df.shape[0]:,}",
                df.shape[1],
                df.isnull().sum().sum(),
                f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
                df.duplicated().sum()
            ]
        })
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["Numeric Variables Summary"] = df[numeric_cols].describe()
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            cat_summary = []
            for col in categorical_cols:
                try:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
                    cat_summary.append({
                        'Column': col,
                        'Unique Values': df[col].nunique(),
                        'Most Frequent': str(mode_val),
                        'Missing %': f"{df[col].isnull().sum() / len(df) * 100:.1f}%"
                    })
                except:
                    cat_summary.append({
                        'Column': col,
                        'Unique Values': 'Error',
                        'Most Frequent': 'Error',
                        'Missing %': 'Error'
                    })
            report["Categorical Variables Summary"] = pd.DataFrame(cat_summary)
        
    except Exception as e:
        report["Error"] = f"Could not generate full report: {str(e)}"
    
    return report

def feature_engineering():
    if 'current_data' not in st.session_state:
        st.warning("Please upload a dataset first!")
        return
    
    st.header("🔧 Feature Engineering")
    
    df = st.session_state['current_data']
    
    try:
        df = fix_dtypes_for_plotly(df)
    except Exception as e:
        st.warning(f"Minor dtype issue: {str(e)}")
    
    # Add 4th tab for Drop Columns
    tab1, tab2, tab3, tab4 = st.tabs(["🔤 Encoding", "📏 Scaling", "➕ New Features", "🗑️ Drop Columns"])
    
    with tab1:
        st.subheader("Categorical Variable Encoding")
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not categorical_cols:
            st.info("No categorical columns found for encoding.")
        else:
            encoding_config = {}
            
            for idx, col in enumerate(categorical_cols):
                with st.expander(f"Configure encoding for: **{col}**", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Unique values:** {df[col].nunique()}")
                        try:
                            sample_values = df[col].dropna().astype(str).unique()[:5]
                            st.write(f"**Sample values:** {', '.join(map(str, sample_values))}")
                        except:
                            st.write("**Sample values:** Could not display")
                    
                    with col2:
                        encoding_key = unique_key("encoding_method", str(col), str(idx))
                        encoding_method = st.selectbox(
                            f"Encoding method for {col}:",
                            ["None", "Label Encoding", "One-Hot Encoding"],
                            key=encoding_key
                        )
                        
                        # NEW: Add option to drop original column
                        drop_key = unique_key("drop_original", str(col), str(idx))
                        drop_original = st.checkbox("Drop original column after encoding", value=True, key=drop_key)
                        
                        encoding_config[col] = {"method": encoding_method, "drop": drop_original}
            
            apply_encoding_key = unique_key("apply_encoding_btn", "all_cols", "main")
            if st.button("🔧 Apply Encoding", type="primary", key=apply_encoding_key):
                df_encoded = apply_encoding(df, encoding_config)
                if df_encoded is not None:
                    st.session_state['current_data'] = fix_dtypes_for_plotly(df_encoded)
                    st.success("✅ Encoding applied successfully!")
                    st.rerun()
    
    with tab2:
        st.subheader("Feature Scaling")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns found for scaling.")
        else:
            scaling_config = {}
            
            st.write("**Select scaling method for numeric columns:**")
            
            for idx, col in enumerate(numeric_cols):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{col}**")
                    try:
                        st.write(f"Range: {df[col].min():.2f} - {df[col].max():.2f}")
                        st.write(f"Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
                    except:
                        st.write("Statistics could not be calculated")
                
                with col2:
                    scaling_key = unique_key("scaling_method", str(col), str(idx))
                    scaling_method = st.selectbox(
                        f"Scaling method:",
                        ["None", "Standard Scaling", "Min-Max Scaling"],
                        key=scaling_key
                    )
                    
                    # NEW: Add option to replace original column
                    replace_key = unique_key("replace_scaled", str(col), str(idx))
                    replace_source = st.checkbox("Replace original column", value=False, key=replace_key)
                    
                    scaling_config[col] = {"method": scaling_method, "replace": replace_source}
            
            apply_scaling_key = unique_key("apply_scaling_btn", "all_cols", "main")
            if st.button("📏 Apply Scaling", type="primary", key=apply_scaling_key):
                df_scaled = apply_scaling(df, scaling_config)
                if df_scaled is not None:
                    st.session_state['current_data'] = fix_dtypes_for_plotly(df_scaled)
                    st.success("✅ Scaling applied successfully!")
                    st.rerun()
    
    with tab3:
        st.subheader("Create New Features")
        st.info("🚧 Advanced feature creation capabilities!")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.write("**Quick Feature Creation:**")
            
            col1, col2 = st.columns(2)
            with col1:
                feature_col1 = st.selectbox("Select first column:", numeric_cols, key="feature_col1_select")
            with col2:
                feature_col2 = st.selectbox("Select second column:", 
                                          [col for col in numeric_cols if col != feature_col1], key="feature_col2_select")
            
            operation = st.selectbox("Select operation:", ["Add", "Subtract", "Multiply", "Divide", "Ratio"], key="operation_select")
            new_feature_name = st.text_input("New feature name:", value=f"{feature_col1}_{operation}_{feature_col2}", key="new_feature_name_input")
            
            if st.button("➕ Create Feature", key="create_feature_btn") and new_feature_name:
                try:
                    if operation == "Add":
                        df[new_feature_name] = df[feature_col1] + df[feature_col2]
                    elif operation == "Subtract":
                        df[new_feature_name] = df[feature_col1] - df[feature_col2]
                    elif operation == "Multiply":
                        df[new_feature_name] = df[feature_col1] * df[feature_col2]
                    elif operation == "Divide":
                        df[new_feature_name] = df[feature_col1] / (df[feature_col2] + 1e-8)
                    elif operation == "Ratio":
                        df[new_feature_name] = df[feature_col1] / (df[feature_col1] + df[feature_col2] + 1e-8)
                    
                    st.session_state['current_data'] = fix_dtypes_for_plotly(df)
                    st.success(f"✅ Created new feature: {new_feature_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating feature: {str(e)}")
    
    # NEW: Drop Columns Tab
    # NEW: Drop Columns Tab
    with tab4:
        st.subheader("🗑️ Column Management - Drop Unwanted Columns")
        
        st.info("💡 Select columns you want to remove from your dataset. Useful for dropping original columns after encoding/scaling or removing irrelevant features.")
        
        # Display current columns with their info
        st.write("**Current Dataset Columns:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)
        
        st.markdown("---")
        
        # Categorize columns for easier selection
        st.write("**📊 Column Categories:**")
        
        # Identify column types
        original_cols = []
        encoded_cols = []
        scaled_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check if it's an encoded column
            if '_encoded' in col_lower or col.endswith(('_0', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9')):
                encoded_cols.append(col)
            # Check if it's a scaled column
            elif '_scaled' in col_lower or '_minmax' in col_lower or '_normalized' in col_lower:
                scaled_cols.append(col)
            # Otherwise treat as original column
            else:
                original_cols.append(col)
        
        # Create columns for organized display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🔤 Original/Base Columns:**")
            if original_cols:
                st.caption(f"{len(original_cols)} columns")
                selected_original = st.multiselect(
                    "Select original columns to drop:",
                    original_cols,
                    key="drop_original_cols"
                )
            else:
                st.caption("No original columns")
                selected_original = []
        
        with col2:
            st.write("**🔧 Encoded Columns:**")
            if encoded_cols:
                st.caption(f"{len(encoded_cols)} columns")
                selected_encoded = st.multiselect(
                    "Select encoded columns to drop:",
                    encoded_cols,
                    key="drop_encoded_cols"
                )
            else:
                st.caption("No encoded columns")
                selected_encoded = []
        
        with col3:
            st.write("**📏 Scaled Columns:**")
            if scaled_cols:
                st.caption(f"{len(scaled_cols)} columns")
                selected_scaled = st.multiselect(
                    "Select scaled columns to drop:",
                    scaled_cols,
                    key="drop_scaled_cols"
                )
            else:
                st.caption("No scaled columns")
                selected_scaled = []
        
        st.markdown("---")
        
        # Alternative: Select from all columns
        st.write("**📝 Or Select From All Columns:**")
        all_selected = st.multiselect(
            "Choose any columns to drop:",
            df.columns.tolist(),
            help="Select any columns you want to remove from the dataset",
            key="drop_all_cols"
        )
        
        # Combine all selections
        columns_to_drop = list(set(selected_original + selected_encoded + selected_scaled + all_selected))
        
        if columns_to_drop:
            st.warning(f"⚠️ **You are about to drop {len(columns_to_drop)} column(s):**")
            
            # Show preview of columns to be dropped
            cols_preview = st.columns(min(len(columns_to_drop), 5))
            for idx, col in enumerate(columns_to_drop[:5]):
                with cols_preview[idx]:
                    st.code(col)
            
            if len(columns_to_drop) > 5:
                st.caption(f"...and {len(columns_to_drop) - 5} more columns")
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🗑️ Drop Selected Columns", type="primary", key="confirm_drop_columns"):
                    try:
                        # Drop columns
                        df_dropped = df.drop(columns=columns_to_drop)
                        
                        # Update session state
                        st.session_state['current_data'] = fix_dtypes_for_plotly(df_dropped)
                        
                        st.success(f"✅ Successfully dropped {len(columns_to_drop)} column(s)!")
                        st.info(f"Dataset now has {df_dropped.shape[1]} columns (was {df.shape[1]})")
                        
                        # Clear selections and rerun
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error dropping columns: {str(e)}")
            
            with col2:
                if st.button("❌ Clear Selection", key="clear_drop_selection"):
                    st.rerun()
        else:
            st.info("👆 Select columns from the lists above to drop them from your dataset")
        
        st.markdown("---")
        
        # Show current dataset shape
        st.write("**📊 Current Dataset Information:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Columns", df.shape[1])
        with col2:
            st.metric("Total Rows", df.shape[0])
        with col3:
            st.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to EDA", key="back_to_eda_btn"):
            st.session_state.step = 3
            st.rerun()
    with col3:
        if st.button("➡️ Text Processing", type="primary", key="proceed_to_text_btn"):
            st.session_state.step = 5
            st.rerun()


def apply_encoding(df, encoding_config):
    """Apply encoding methods to categorical columns with optional original column dropping"""
    try:
        df_encoded = df.copy()
        
        for col, cfg in encoding_config.items():
            # Support both old format (string) and new format (dict)
            if isinstance(cfg, dict):
                method = cfg.get("method", "None")
                drop_src = cfg.get("drop", True)
            else:
                method = cfg
                drop_src = True  # Default behavior for backward compatibility
            
            if method == "None":
                continue
                
            elif method == "Label Encoding":
                try:
                    le = LabelEncoder()
                    encoded_col_name = f"{col}_encoded"
                    
                    # Remove existing encoded column if it exists
                    if encoded_col_name in df_encoded.columns:
                        df_encoded = df_encoded.drop(columns=[encoded_col_name])
                    
                    df_encoded[encoded_col_name] = le.fit_transform(df_encoded[col].astype(str))
                    
                    # Drop original column if requested
                    if drop_src and col in df_encoded.columns:
                        df_encoded = df_encoded.drop(columns=[col])
                        
                except Exception as e:
                    st.warning(f"Could not apply label encoding to {col}: {str(e)}")
                    
            elif method == "One-Hot Encoding":
                try:
                    # Generate dummy variables
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    
                    # Remove existing dummy columns that would conflict
                    existing_dummy_cols = [c for c in dummies.columns if c in df_encoded.columns]
                    if existing_dummy_cols:
                        df_encoded = df_encoded.drop(columns=existing_dummy_cols)
                    
                    # Drop original column if requested
                    if drop_src and col in df_encoded.columns:
                        df_encoded = df_encoded.drop(columns=[col])
                    
                    # Concatenate the new dummy columns
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    
                except Exception as e:
                    st.warning(f"Could not apply one-hot encoding to {col}: {str(e)}")
        
        # Final check: remove any remaining duplicate columns
        df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]
        
        return df_encoded
        
    except Exception as e:
        st.error(f"Error applying encoding: {str(e)}")
        return None


def apply_scaling(df, scaling_config):
    """Apply scaling methods to numeric columns with option to replace original"""
    try:
        df_scaled = df.copy()
        
        for col, cfg in scaling_config.items():
            # Support both old format (string) and new format (dict)
            if isinstance(cfg, dict):
                method = cfg.get("method", "None")
                replace_src = cfg.get("replace", False)
            else:
                method = cfg
                replace_src = False  # Default: create new column
            
            if method == "None":
                continue
                
            elif method == "Standard Scaling":
                try:
                    scaler = StandardScaler()
                    scaled_vals = scaler.fit_transform(df_scaled[[col]]).flatten()
                    
                    if replace_src:
                        df_scaled[col] = scaled_vals
                    else:
                        df_scaled[f"{col}_scaled"] = scaled_vals
                        
                except Exception as e:
                    st.warning(f"Could not apply standard scaling to {col}: {str(e)}")
                    
            elif method == "Min-Max Scaling":
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    scaled_vals = scaler.fit_transform(df_scaled[[col]]).flatten()
                    
                    if replace_src:
                        df_scaled[col] = scaled_vals
                    else:
                        df_scaled[f"{col}_minmax"] = scaled_vals
                        
                except Exception as e:
                    st.warning(f"Could not apply min-max scaling to {col}: {str(e)}")
        
        return df_scaled
        
    except Exception as e:
        st.error(f"Error applying scaling: {str(e)}")
        return None

    
    # text processing

def text_processing():
    """New text processing module for NLP capabilities"""
    if 'current_data' not in st.session_state:
        st.warning("Please upload a dataset first!")
        return
    
    st.header("📝 Advanced Text Processing & NLP")
    
    df = st.session_state['current_data']
    text_processor = TextProcessor()
    
    # Identify text columns
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains substantial text (not just categories)
            sample_text = df[col].dropna().astype(str)
            if len(sample_text) > 0:
                avg_length = sample_text.str.len().mean()
                if avg_length > 10:  # Average length > 10 characters
                    text_columns.append(col)
    
    if not text_columns:
        st.info("No text columns detected in your dataset. Upload a dataset with text data to use this feature.")
        return
    
    st.write(f"**Found {len(text_columns)} text columns:** {', '.join(text_columns)}")
    
    # Text processing options
    tab1, tab2, tab3, tab4 = st.tabs(["🧹 Text Cleaning", "😊 Sentiment Analysis", "🔍 Keyword Extraction", "📊 Text Analytics"])
    
    with tab1:
        st.subheader("Text Cleaning & Preprocessing")
        
        selected_text_col = st.selectbox("Select text column to clean:", text_columns)
        
        if selected_text_col:
            # Show sample before cleaning
            st.write("**Sample text before cleaning:**")
            sample_before = df[selected_text_col].dropna().head(3).tolist()
            for i, text in enumerate(sample_before, 1):
                st.write(f"{i}. {str(text)[:200]}...")
            
            # Cleaning options
            cleaning_options = st.multiselect(
                "Select cleaning operations:",
                ["Remove URLs", "Remove emails", "Remove special characters", "Convert to lowercase", "Remove extra whitespace"],
                default=["Remove URLs", "Remove emails", "Convert to lowercase", "Remove extra whitespace"]
            )
            
            if st.button("🧹 Clean Text", key="clean_text_btn"):
                with st.spinner("Cleaning text data..."):
                    cleaned_text = []
                    
                    for text in df[selected_text_col]:
                        if pd.notna(text):
                            cleaned = text_processor.clean_text(text)
                            cleaned_text.append(cleaned)
                        else:
                            cleaned_text.append(text)
                    
                    # Create new column with cleaned text
                    new_col_name = f"{selected_text_col}_cleaned"
                    df[new_col_name] = cleaned_text
                    st.session_state['current_data'] = df
                    
                    st.success(f"✅ Text cleaning completed! New column '{new_col_name}' created.")
                    
                    # Show sample after cleaning
                    st.write("**Sample text after cleaning:**")
                    sample_after = df[new_col_name].dropna().head(3).tolist()
                    for i, text in enumerate(sample_after, 1):
                        st.write(f"{i}. {str(text)[:200]}...")
    
    with tab2:
        st.subheader("Sentiment Analysis")
        
        selected_sentiment_col = st.selectbox("Select text column for sentiment analysis:", text_columns, key="sentiment_col")
        
        if selected_sentiment_col:
            if st.button("😊 Analyze Sentiment", key="analyze_sentiment_btn"):
                with st.spinner("Analyzing sentiment..."):
                    sentiments = []
                    
                    for text in df[selected_sentiment_col]:
                        if pd.notna(text):
                            sentiment = text_processor.get_sentiment(text)
                            sentiments.append(sentiment)
                        else:
                            sentiments.append('Neutral')
                    
                    # Add sentiment column
                    sentiment_col_name = f"{selected_sentiment_col}_sentiment"
                    df[sentiment_col_name] = sentiments
                    st.session_state['current_data'] = df
                    
                    st.success(f"✅ Sentiment analysis completed! New column '{sentiment_col_name}' created.")
                    
                    # Show sentiment distribution
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
                        ax.set_title("Sentiment Distribution")
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.write("**Sentiment Breakdown:**")
                        st.dataframe(sentiment_counts.reset_index())
    
    with tab3:
        st.subheader("Keyword Extraction")
        
        selected_keyword_col = st.selectbox("Select text column for keyword extraction:", text_columns, key="keyword_col")
        
        if selected_keyword_col:
            num_keywords = st.slider("Number of keywords to extract per text:", 1, 10, 5)
            
            if st.button("🔍 Extract Keywords", key="extract_keywords_btn"):
                with st.spinner("Extracting keywords..."):
                    all_keywords = []
                    
                    for text in df[selected_keyword_col]:
                        if pd.notna(text):
                            keywords = text_processor.extract_keywords(text, num_keywords)
                            all_keywords.append(', '.join(keywords))
                        else:
                            all_keywords.append('')
                    
                    # Add keywords column
                    keywords_col_name = f"{selected_keyword_col}_keywords"
                    df[keywords_col_name] = all_keywords
                    st.session_state['current_data'] = df
                    
                    st.success(f"✅ Keyword extraction completed! New column '{keywords_col_name}' created.")
                    
                    # Show sample keywords
                    st.write("**Sample extracted keywords:**")
                    sample_keywords = df[keywords_col_name].dropna().head(5)
                    for i, keywords in enumerate(sample_keywords, 1):
                        st.write(f"{i}. {keywords}")
    
    with tab4:
        st.subheader("Text Analytics Dashboard")
        
        selected_analytics_col = st.selectbox("Select text column for analytics:", text_columns, key="analytics_col")
        
        if selected_analytics_col:
            # Calculate text statistics
            text_data = df[selected_analytics_col].dropna().astype(str)
            
            if len(text_data) > 0:
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Texts", len(text_data))
                with col2:
                    avg_length = text_data.str.len().mean()
                    st.metric("Avg Length", f"{avg_length:.0f} chars")
                with col3:
                    word_count = text_data.str.split().str.len().mean()
                    st.metric("Avg Words", f"{word_count:.0f}")
                with col4:
                    unique_texts = text_data.nunique()
                    st.metric("Unique Texts", unique_texts)
                
                # Text length distribution
                st.subheader("Text Length Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                text_lengths = text_data.str.len()
                ax.hist(text_lengths, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Text Length (characters)')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Text Lengths')
                st.pyplot(fig)
                plt.close()
                
                # Word cloud simulation (top words)
                st.subheader("Most Common Words")
                try:
                    all_text = ' '.join(text_data.tolist())
                    processor = TextProcessor()
                    all_keywords = processor.extract_keywords(all_text, num_keywords=20)
                    
                    if all_keywords:
                        keywords_df = pd.DataFrame({'Word': all_keywords, 'Frequency': range(len(all_keywords), 0, -1)})
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(keywords_df['Word'][:10], keywords_df['Frequency'][:10])
                        ax.set_xlabel('Frequency')
                        ax.set_title('Top 10 Most Common Words')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                except Exception as e:
                    st.info("Could not generate word frequency analysis.")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Feature Engineering", key="back_to_feature_text_btn"):
            st.session_state.step = 4
            st.rerun()
    with col3:
        if st.button("➡️ Export Results", type="primary", key="proceed_to_export_text_btn"):
            st.session_state.step = 6
            st.rerun()

def export_results():
    if 'current_data' not in st.session_state:
        st.warning("No processed data available for export!")
        return
    
    st.header("💾 Export Results")
    
    df_final = st.session_state['current_data']
    
    # Processing summary
    if 'original_data' in st.session_state:
        df_original = st.session_state['original_data']
        
        st.subheader("📊 Processing Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **Original Dataset:**
            - Rows: {df_original.shape[0]:,}
            - Columns: {df_original.shape[1]}
            - Missing: {df_original.isnull().sum().sum():,}
            """)
        
        with col2:
            st.markdown(f"""
            **Final Dataset:**
            - Rows: {df_final.shape[0]:,}
            - Columns: {df_final.shape[1]}
            - Missing: {df_final.isnull().sum().sum():,}
            """)
        
        with col3:
            rows_change = df_final.shape[0] - df_original.shape[0]
            cols_change = df_final.shape[1] - df_original.shape[1]
            missing_change = df_final.isnull().sum().sum() - df_original.isnull().sum().sum()
            
            st.markdown(f"""
            **Changes:**
            - Rows: {rows_change:+,}
            - Columns: {cols_change:+}
            - Missing: {missing_change:+,}
            """)
    
    # Export options
    st.subheader("📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            csv_data = df_final.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name="datadoctor_cleaned_data.csv",
                mime="text/csv",
                help="Download the cleaned dataset as CSV file"
            )
        except Exception as e:
            st.error(f"Error preparing CSV: {str(e)}")
        
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_final.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                if 'original_data' in st.session_state:
                    st.session_state['original_data'].to_excel(writer, sheet_name='Original_Data', index=False)
            
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="📊 Download as Excel",
                data=excel_data,
                file_name="datadoctor_processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error preparing Excel: {str(e)}")
    
    with col2:
        if st.button("📋 Generate Summary Report", key="generate_report_btn"):
            try:
                report_buffer = io.StringIO()
                report_buffer.write("DATADOCTOR - DATA CLEANING SUMMARY REPORT\n")
                report_buffer.write("="*50 + "\n\n")
                
                if 'original_data' in st.session_state:
                    report_buffer.write("ORIGINAL DATASET:\n")
                    report_buffer.write(f"- Shape: {st.session_state['original_data'].shape}\n")
                    report_buffer.write(f"- Missing values: {st.session_state['original_data'].isnull().sum().sum()}\n\n")
                
                report_buffer.write("FINAL DATASET:\n")
                report_buffer.write(f"- Shape: {df_final.shape}\n")
                report_buffer.write(f"- Missing values: {df_final.isnull().sum().sum()}\n\n")
                
                report_buffer.write("DATA TYPES:\n")
                for col, dtype in df_final.dtypes.items():
                    report_buffer.write(f"- {col}: {dtype}\n")
                
                # AI Quality Assessment
                assistant = DataQualityAssistant()
                quality_metrics = assistant.assess_data_quality(df_final)
                report_buffer.write(f"\nAI QUALITY ASSESSMENT:\n")
                report_buffer.write(f"- Overall Score: {quality_metrics['overall_score']}/100\n")
                report_buffer.write(f"- Completeness: {quality_metrics['completeness']}%\n")
                report_buffer.write(f"- Consistency: {quality_metrics['consistency']}%\n")
                report_buffer.write(f"- Validity: {quality_metrics['validity']}%\n")
                
                report_data = report_buffer.getvalue()
                st.download_button(
                    label="📝 Download Report",
                    data=report_data,
                    file_name="datadoctor_summary_report.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
        
        st.info("📋 Final dataset is ready for download!")
    
    # Final dataset preview
    st.subheader("👀 Final Dataset Preview")
    st.dataframe(df_final.head(10), use_container_width=True)
    
    # Session Storage
    if st.button("💾 Save Session", key="save_session_btn"):
        try:
            session_id = st.session_state.get('session_id', 'default')
            filename = st.session_state.get('uploaded_filename', 'unknown.csv')
            
            # Save to database
            conn = sqlite3.connect('datadoctor.db')
            cursor = conn.cursor()
            
            operations = {
                'original_shape': st.session_state['original_data'].shape if 'original_data' in st.session_state else None,
                'final_shape': df_final.shape,
                'operations': 'Data cleaning, EDA, Feature engineering, Text processing'
            }
            
            cursor.execute('''
            INSERT INTO cleaning_sessions (session_id, original_filename, operations_performed)
            VALUES (?, ?, ?)
            ''', (session_id, filename, json.dumps(operations)))
            
            conn.commit()
            conn.close()
            
            st.success("✅ Session saved successfully!")
        except Exception as e:
            st.error(f"Error saving session: {str(e)}")
    
    st.success("🎉 Data cleaning and processing completed successfully!")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Text Processing", key="back_to_text_processing_btn"):
            st.session_state.step = 5
            st.rerun()
    with col2:
        if st.button("🔄 Start New Project", type="secondary", key="start_new_project_btn"):
        # Clear session state for new project
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Force re-initialization of RAG system
            st.session_state['force_rag_reset'] = True
            
            st.success("✅ Starting new project...")
            st.rerun()


# For brevity, I'm not repeating all the existing functions here, but they remain unchanged
# Just make sure to include them in your complete file
   
if __name__ == "__main__":
    main()
