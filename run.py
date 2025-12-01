import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Hospital Readmission Penalty Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        padding: 15px;
        background: #ecf0f1;
        border-left: 5px solid #3498db;
        margin: 20px 0;
        border-radius: 5px;
    }
    .info-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4F8BF9;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4F8BF9 0%, #3498db 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, classification_report)
import warnings

warnings.filterwarnings('ignore')

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Support Vector Classifier (SVC)": SVC(kernel='rbf', probability=True, random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv")
        return df
    except FileNotFoundError:
        st.error("❌ CSV file not found. Please ensure the file is in the same directory.")
        return None

def get_state_fullnames():
    return {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
        'DC': 'District of Columbia', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands', 
        'GU': 'Guam', 'AS': 'American Samoa', 'MP': 'Northern Mariana Islands'
    }

def preprocess_data(df):
    numeric_cols = ['Number of Discharges', 'Excess Readmission Ratio', 
                   'Predicted Readmission Rate', 'Expected Readmission Rate', 
                   'Number of Readmissions']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Excess Readmission Ratio'])
    df['Is_Penalized'] = (df['Excess Readmission Ratio'] > 1).astype(int)
    
    drop_cols = ['Facility ID', 'Facility Name', 'Excess Readmission Ratio', 
                'Footnote', 'Start Date', 'End Date']
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    return df, df_model

def get_preprocessing_pipeline(X):
    num_features = X.select_dtypes(include='number').columns.tolist()
    cat_features = X.select_dtypes(include='object').columns.tolist()
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    return preprocessor, num_features, cat_features

def show_data_overview():
    st.markdown('<div class="section-header">📊 Step 1: Data Overview</div>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Understand the raw dataset before processing.<br><br>
        <b>Key Concepts:</b>
        <ul>
            <li><b>Dataset:</b> FY-2025 CMS Hospital Readmissions Reduction Program (HRRP)</li>
            <li><b>Rows:</b> Each row represents a hospital's performance for a specific medical condition</li>
            <li><b>Columns:</b> Features include state, condition, discharge counts, and readmission rates</li>
            <li><b>Target Variable:</b> Whether the hospital is penalized (Excess Readmission Ratio > 1)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("📊 Total Columns", df.shape[1])
    with col3:
        st.metric("🏥 Unique Hospitals", df['Facility Name'].nunique())
    
    st.subheader("Sample Data (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True, height=300)
    
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns.tolist(),
        'Data Type': df.dtypes.astype(str).values,
        'Missing Values': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values.astype(float) / len(df) * 100).round(2)
    })
    st.dataframe(col_info, use_container_width=True, height=300)
    
    st.dataframe(col_info, use_container_width=True, height=300)
    
    st.subheader("Statistical Summary (Numeric Columns)")
    st.dataframe(df.describe().T, use_container_width=True)

def show_preprocessing():
    df = load_data()
    if df is None:
        return
    
    df, df_model = preprocess_data(df)
    
    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Clean and prepare data for machine learning models.<br><br>
        <b>Steps Performed:</b>
        <ol>
            <li><b>Handle Missing Values:</b> Use median for numeric, most frequent for categorical</li>
            <li><b>Feature Scaling:</b> Standardize numeric features (mean=0, std=1)</li>
            <li><b>Encoding:</b> Convert categorical variables to numeric using One-Hot Encoding</li>
            <li><b>Target Creation:</b> Create binary target (Penalized=1, Not Penalized=0)</li>
            <li><b>Train-Test Split:</b> Split data into 80% training and 20% testing</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Missing Value Analysis")
    missing = df_model.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing.index.tolist(),
        'Missing Count': missing.values.astype(int),
        'Missing %': (missing.values.astype(float) / len(df_model) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(missing_df.head(10), use_container_width=True)
    with col2:
        if missing[missing > 0].shape[0] > 0:
            fig = px.bar(missing_df.head(10), x='Column', y='Missing Count',
                        title='Top 10 Columns with Missing Values',
                        color='Missing Count', color_continuous_scale='Reds')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values in the dataset!")

    st.subheader("Feature Correlation Matrix")
    numeric_df = df_model.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                   color_continuous_scale='RdBu_r',
                   title='Correlation Heatmap (Numeric Features)')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlier Detection (Boxplots)")
    num_cols = numeric_df.columns[:4]
    cols = st.columns(2)
    for idx, col in enumerate(num_cols):
        with cols[idx % 2]:
            fig = px.box(numeric_df, y=col, title=f'Boxplot: {col}',
                        color_discrete_sequence=['#4F8BF9'])
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Train-Test Split Information")
    y = df_model['Is_Penalized']
    X = df_model.drop('Is_Penalized', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Training Samples", f"{X_train.shape[0]:,}")
    with col2:
        st.metric("🧪 Testing Samples", f"{X_test.shape[0]:,}")
    with col3:
        st.metric("📐 Split Ratio", "80:20")

def show_eda():
    df = load_data()
    if df is None:
        return
    df, df_model = preprocess_data(df)
    
    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Visualize patterns and relationships in the data.<br><br>
        <b>Key Visualizations:</b>
        <ol>
            <li><b>Target Distribution:</b> Balance of penalized vs non-penalized hospitals</li>
            <li><b>Feature Distributions:</b> Histogram analysis of numeric features</li>
            <li><b>Condition Analysis:</b> Which medical conditions have higher penalty rates</li>
            <li><b>Geographic Patterns:</b> State-wise penalty distribution</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("1. Target Variable Distribution")
    col1, col2 = st.columns([1, 2])
    with col1:
        counts = df_model['Is_Penalized'].value_counts()
        st.metric("✅ Not Penalized", f"{counts.get(0,0):,}", delta=f"{(counts.get(0,0)/len(df_model)*100):.1f}%")
        st.metric("⚠️ Penalized", f"{counts.get(1,0):,}", delta=f"{(counts.get(1,0)/len(df_model)*100):.1f}%", delta_color="inverse")
    with col2:
        fig = px.pie(df_model, names='Is_Penalized',
                    title='Penalty Status Distribution',
                    color='Is_Penalized',
                    color_discrete_map={0: '#4CAF50', 1: '#E74C3C'},
                    hole=0.4,
                    labels={'Is_Penalized': 'Penalty Status'},
                    category_orders={'Is_Penalized': [0, 1]})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. Readmission Rates by Penalty Status")
    fig = px.histogram(df_model, x='Predicted Readmission Rate', color='Is_Penalized',
                       barmode='overlay', nbins=30,
                       color_discrete_map={0: '#4CAF50', 1: '#E74C3C'},
                       labels={'Is_Penalized': 'Penalty Status', 'Predicted Readmission Rate': 'Predicted Readmission Rate (%)'},
                       title='Predicted Readmission Rate Distribution', opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Top Medical Conditions with Penalties")
    top_conditions = df_model[df_model['Is_Penalized'] == 1]['Measure Name'].value_counts().nlargest(10).index
    df_top = df_model[df_model['Measure Name'].isin(top_conditions)]
    fig = px.histogram(df_top, x="Measure Name", color="Is_Penalized",
                      barmode="group",
                      color_discrete_map={0: "#4CAF50", 1: "#E74C3C"},
                      title="Top 10 Medical Conditions: Penalty Frequency",
                      labels={'Is_Penalized': 'Penalty Status', 'count': 'Count'})
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("4. Distribution of Numeric Features")
    numeric_cols = df_model.select_dtypes(include='number').columns.tolist()
    selected_feature = st.selectbox("Select a feature to visualize:", numeric_cols)
    fig = px.histogram(df_model, x=selected_feature, color='Is_Penalized',
                      barmode='overlay', nbins=30,
                      color_discrete_map={0: '#4CAF50', 1: '#E74C3C'},
                      title=f'Distribution of {selected_feature}',
                      labels={'Is_Penalized': 'Penalty Status'},
                      opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5. Geographic Distribution of Penalties")
    state_penalty = df_model.groupby('State')['Is_Penalized'].mean().sort_values(ascending=False).head(15) * 100
    fig = px.bar(state_penalty.reset_index(), x='State', y='Is_Penalized',
                title='Top 15 States by Penalty Rate',
                color='Is_Penalized',
                color_continuous_scale='Reds',
                labels={'Is_Penalized': 'Penalty Rate (%)'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    df = load_data()
    if df is None:
        return
    _, df_model = preprocess_data(df)

    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Train machine learning models to predict hospital penalties.<br><br>
        <b>Models Used (INT234 Aligned):</b>
        <ol>
            <li><b>Logistic Regression:</b> Linear model for binary classification</li>
            <li><b>Decision Tree Classifier:</b> Tree-based model for interpretable decisions</li>
            <li><b>Support Vector Classifier (SVC):</b> Finds optimal separating hyperplane</li>
            <li><b>K-Nearest Neighbors (KNN):</b> Instance-based learning using nearest data points</li>
            <li><b>MLP Neural Network:</b> Multi-layer perceptron for complex patterns</li>
        </ol>
        <b>Evaluation Metrics:</b>
        <ul>
            <li><b>Accuracy:</b> Overall correctness of predictions</li>
            <li><b>Precision:</b> Accuracy of positive predictions</li>
            <li><b>Recall:</b> Ability to find all positive cases</li>
            <li><b>F1-Score:</b> Harmonic mean of precision and recall</li>
            <li><b>ROC-AUC:</b> Area under ROC curve (discrimination ability)</li>
            <li><b>5-Fold CV:</b> Cross-validation score for model stability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    X = df_model.drop('Is_Penalized', axis=1)
    y = df_model['Is_Penalized']
    preprocessor, num_features, cat_features = get_preprocessing_pipeline(X)

    st.subheader("Select a Model to Train")
    model_name = st.selectbox("Choose Model:", list(MODELS.keys()), key="single_model_select")
    model = MODELS[model_name]

    if st.button("🚀 Train Model", key="train_single"):
        with st.spinner(f"Training {model_name}..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_train_pred = pipe.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['classifier'], 'predict_proba') else pipe.decision_function(X_test)
            auc = roc_auc_score(y_test, y_prob)
            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
            st.success(f"✅ {model_name} trained successfully!")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("🎯 Train Acc", f"{train_acc:.4f}")
            col2.metric("🧪 Test Acc", f"{test_acc:.4f}")
            col3.metric("⚡ Precision", f"{precision:.4f}")
            col4.metric("🔍 Recall", f"{recall:.4f}")
            col5.metric("🤝 F1-Score", f"{f1:.4f}")
            col6.metric("🎯 ROC-AUC", f"{auc:.4f}")
            st.metric("🔄 5-Fold CV Mean", f"{cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              color_continuous_scale='Blues',
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['No Penalty', 'Penalty'],
                              y=['No Penalty', 'Penalty'])
            fig_cm.update_layout(height=500)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                        name=f'ROC Curve (AUC={auc:.4f})',
                                        line=dict(color='#4F8BF9', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                        name='Random Classifier',
                                        line=dict(dash='dash', color='gray')))
            fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', showlegend=True, height=500)
            st.plotly_chart(fig_roc, use_container_width=True)
            st.subheader("Detailed Classification Report")
            report = classification_report(y_test, y_pred, target_names=['No Penalty', 'Penalty'], output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df.style.highlight_max(axis=0, color='#D2F8D2'), use_container_width=True)

def show_model_comparison():
    df = load_data()
    if df is None:
        return
    df, df_model = preprocess_data(df)
    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Compare all 5 models to find the best performer.<br><br>
        <b>What to Look For:</b>
        <ul>
            <li><b>Highest Accuracy:</b> Best overall performance</li>
            <li><b>Highest AUC:</b> Best discrimination ability</li>
            <li><b>Balanced Metrics:</b> Good precision AND recall</li>
            <li><b>Stable CV Score:</b> Low standard deviation means consistent performance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    X = df_model.drop('Is_Penalized', axis=1)
    y = df_model['Is_Penalized']
    preprocessor, num_features, cat_features = get_preprocessing_pipeline(X)
    if st.button("🚀 Train All Models", key="train_all"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = {}
        for idx, (model_name, model) in enumerate(MODELS.items()):
            status_text.text(f"Training {model_name}... ({idx+1}/{len(MODELS)})")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_train_pred = pipe.predict(X_train)
            y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['classifier'], 'predict_proba') else pipe.decision_function(X_test)
            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
            results[model_name] = {
                'Train Accuracy': accuracy_score(y_train, y_train_pred),
                'Test Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_prob),
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            progress_bar.progress((idx + 1) / len(MODELS))
        status_text.text("✅ All models trained successfully!")
        st.subheader("📊 Model Performance Comparison")
        results_df = pd.DataFrame({
            name: {
                'Train Acc': res['Train Accuracy'],
                'Test Acc': res['Test Accuracy'],
                'Precision': res['Precision'],
                'Recall': res['Recall'],
                'F1-Score': res['F1-Score'],
                'ROC-AUC': res['ROC-AUC'],
                'CV Mean': res['CV Mean'],
                'CV Std': res['CV Std']
            } for name, res in results.items()
        }).T
        results_df = results_df.round(4)
        st.dataframe(results_df.style.highlight_max(axis=0, color='#D2F8D2').format("{:.4f}"), use_container_width=True, height=250)
        st.subheader("📈 Visual Comparison")
        fig = px.bar(results_df.reset_index().melt(id_vars='index'),
                     x='index', y='value', color='variable',
                     barmode='group',
                     title='Model Performance Comparison',
                     labels={'value': 'Score', 'variable': 'Metric', 'index': 'Model'})
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("🔀 Combined ROC Curves")
        fig_roc = go.Figure()
        colors = ['#4F8BF9', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']
        for idx, (model_name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                         name=f'{model_name} (AUC={res["ROC-AUC"]:.4f})',
                                         line=dict(color=colors[idx % len(colors)], width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     name='Random Classifier',
                                     line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(title='All Models - ROC Curve Comparison',
                             xaxis_title='False Positive Rate',
                             yaxis_title='True Positive Rate',
                             showlegend=True, height=600)
        st.plotly_chart(fig_roc, use_container_width=True)
        best_model = max(results.items(), key=lambda x: x[1]['Test Accuracy'])
        st.success(f"""
        🏆 **Best Model:** {best_model[0]}
        - Test Accuracy: {best_model[1]['Test Accuracy']:.2%}
        - ROC-AUC: {best_model[1]['ROC-AUC']:.4f}
        - F1-Score: {best_model[1]['F1-Score']:.4f}
        """)

def show_live_prediction():
    df = load_data()
    if df is None:
        return
    df, df_model = preprocess_data(df)
    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Predict penalty risk for a hypothetical hospital.<br><br>
        <b>Key Terms:</b>
        <ul>
            <li><b>Penalized:</b> Hospital receives Medicare payment reduction due to excessive readmissions</li>
            <li><b>Predicted Readmission Rate:</b> Actual readmission rate observed at the hospital</li>
            <li><b>Expected Readmission Rate:</b> National average readmission rate for the same condition</li>
            <li><b>Excess Readmission Ratio:</b> Predicted ÷ Expected (&gt;1.0 means penalized)</li>
            <li><b>Number of Discharges:</b> Total patients discharged with the condition</li>
            <li><b>Medical Condition:</b> Specific diagnosis (e.g., Heart Attack, Pneumonia, Heart Failure)</li>
        </ul>
        <b>How to Use:</b>
        <ol>
            <li>Select a machine learning model from the dropdown</li>
            <li>Enter hospital information using the input fields below</li>
            <li>Click "Predict Penalty Risk" to get AI-powered results</li>
        </ol>
        <b>Interpretation:</b>
        <ul>
            <li><b>High Risk (Penalized):</b> Hospital likely faces Medicare payment reduction - immediate action required</li>
            <li><b>Low Risk (Not Penalized):</b> Hospital meets quality standards - maintain current practices</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    X = df_model.drop('Is_Penalized', axis=1)
    y = df_model['Is_Penalized']
    preprocessor, num_features, cat_features = get_preprocessing_pipeline(X)
    st.subheader("Select Prediction Model")
    model_name = st.selectbox("Choose Model:", list(MODELS.keys()), key="prediction_model")
    model = MODELS[model_name]
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(X, y)
    st.subheader("Enter Hospital Information")
    state_fullnames = get_state_fullnames()
    state_map = {v: k for k, v in state_fullnames.items()}
    state_options = list(state_map.keys())
    col1, col2 = st.columns(2)
    with col1:
        state_display = st.selectbox("🗺️ State", state_options)
        state = state_map[state_display]
        condition = st.selectbox("🏥 Medical Condition", sorted(df['Measure Name'].unique()))
        discharges = st.slider("📊 Number of Discharges", min_value=0, max_value=5000, value=500, step=50)
    with col2:
        pred_rate = st.slider("📈 Predicted Readmission Rate (%)", min_value=0.0, max_value=30.0, value=16.5, step=0.5, help="Actual readmission rate observed for this hospital")
        exp_rate = st.slider("📉 Expected Readmission Rate (%)", min_value=0.0, max_value=30.0, value=15.0, step=0.5, help="National average readmission rate for this condition")
        num_readm = st.number_input("🔄 Number of Readmissions", min_value=0, max_value=1000, value=50, step=10)
    input_data = pd.DataFrame({
        'State': [state],
        'Measure Name': [condition],
        'Number of Discharges': [discharges],
        'Predicted Readmission Rate': [pred_rate],
        'Expected Readmission Rate': [exp_rate],
        'Number of Readmissions': [num_readm]
    })
    if st.button("🔮 Predict Penalty Risk", key="predict_btn"):
        with st.spinner("Analyzing hospital data..."):
            prediction = pipe.predict(input_data)[0]
            if hasattr(pipe.named_steps['classifier'], 'predict_proba'):
                probability = pipe.predict_proba(input_data)[0][1]
            else:
                from scipy.special import expit
                probability = expit(pipe.decision_function(input_data))[0]
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK: Hospital Likely to be Penalized")
                st.markdown(f"""
                <div style='background:#ffe6e6; padding:20px; border-radius:10px; border-left:5px solid #E74C3C;'>
                <h4>Risk Assessment</h4>
                <p><b>Penalty Probability:</b> {probability:.2%}</p>
                <p><b>Recommendation:</b> Immediate action required to reduce readmission rates.</p>
                <p><b>Suggested Actions:</b></p>
                <ul>
                    <li>Review discharge procedures and patient education</li>
                    <li>Implement post-discharge follow-up protocols</li>
                    <li>Analyze readmission patterns for this condition</li>
                    <li>Coordinate with primary care providers</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("### ✅ LOW RISK: Hospital Unlikely to be Penalized")
                st.markdown(f"""
                <div style='background:#e6ffe6; padding:20px; border-radius:10px; border-left:5px solid #4CAF50;'>
                <h4>Risk Assessment</h4>
                <p><b>Penalty Probability:</b> {probability:.2%}</p>
                <p><b>Status:</b> Current practices are effective.</p>
                <p><b>Recommendations:</b></p>
                <ul>
                    <li>Continue current quality improvement initiatives</li>
                    <li>Monitor readmission trends regularly</li>
                    <li>Share best practices with other departments</li>
                    <li>Maintain documentation standards</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            st.subheader("Visual Risk Indicator")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#E74C3C" if prediction == 1 else "#4CAF50"},
                    'steps': [
                        {'range': [0, 30], 'color': "#D2F8D2"},
                        {'range': [30, 70], 'color': "#FFF9C4"},
                        {'range': [70, 100], 'color': "#FFCDD2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def main():
    st.markdown('<div class="main-header">🏥 Hospital Readmission Penalty Predictor</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="font-size:22px;font-weight:bold;margin-bottom:10px;">🧭 Navigation</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    section_labels = [
        ("📊", "Data Overview"),
        ("🔧", "Data Preprocessing"),
        ("📈", "Exploratory Data Analysis"),
        ("🧑‍💻", "Model Training"),
        ("🏆", "Model Comparison"),
        ("🔮", "Live Prediction")
    ]
    section_options = [f"{icon} {label}" for icon, label in section_labels]
    page = st.sidebar.radio(
        "Select a Section:",
        section_options,
        key="sidebar_radio"
    )
    st.sidebar.info("""
    **Project Structure:**
    1. Load & explore data
    2. Preprocess & clean
    3. Visualize patterns
    4. Train ML models
    5. Compare performance
    6. Make predictions
    """)
    
    if "Data Overview" in page:
        show_data_overview()
    elif "Data Preprocessing" in page:
        show_preprocessing()
    elif "Exploratory Data Analysis" in page:
        show_eda()
    elif "Model Training" in page:
        show_model_training()
    elif "Model Comparison" in page:
        show_model_comparison()
    elif "Live Prediction" in page:
        show_live_prediction()

if __name__ == "__main__":
    main()