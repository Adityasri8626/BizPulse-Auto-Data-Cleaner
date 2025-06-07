import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import io
import re

st.set_page_config(page_title="BizPulse – Auto Data Cleaner", layout="wide")
st.title("🧹 BizPulse: Smart Auto Data Cleaner")
st.markdown("###### 🚀 Powered by **Aditya Srivastava**")

# Upload file
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

def clean_column_names(df):
    return df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)

def standardize_categories(df):
    st.subheader("🧾 Categorical Standardization")
    summary = []

    for col in df.select_dtypes(include='object'):
        if df[col].nunique() < 50:  # only for small cardinality
            original_vals = df[col].unique().tolist()
            df[col] = df[col].astype(str).str.strip().str.lower()

            # Gender Example
            if 'gender' in col:
                df[col] = df[col].replace({
                    'm': 'male', 'male': 'male',
                    'f': 'female', 'female': 'female',
                    'nan': 'unknown', '': 'unknown'
                })

            # Fill empty with 'unknown'
            df[col] = df[col].replace(['', 'nan', 'none', 'null'], 'unknown')

            summary.append((col, original_vals, df[col].unique().tolist()))

    if summary:
        st.success("✅ Categorical Columns Standardized")
        st.dataframe(pd.DataFrame(summary, columns=["Column", "Before", "After"]))
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    st.subheader("🔍 Original Dataset Preview")
    st.dataframe(df.head())
    st.info(f"📊 Original Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Null Handling
    st.subheader("🧼 Null Value Handling")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        st.success("✅ No nulls found.")
    else:
        st.warning("⚠️ Null values found. Handling them...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna("unknown")
                else:
                    df[col] = df[col].fillna(df[col].median())
        st.success("✅ Nulls filled.")

    # Duplicates
    st.subheader("🔁 Duplicate Removal")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        st.warning(f"⚠️ {dup_count} duplicates found. Removing...")
        df = df.drop_duplicates()
        st.success("✅ Duplicates removed.")
    else:
        st.success("✅ No duplicates found.")

    # Constant Columns
    st.subheader("🚫 Constant Columns Removal")
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        st.warning(f"Constant Columns: {const_cols}")
        df.drop(columns=const_cols, inplace=True)
        st.success("✅ Removed.")
    else:
        st.success("✅ No constant columns.")

    # Column Name Cleanup
    df.columns = clean_column_names(df)
    st.success("🧹 Cleaned column names.")

    # Outlier Detection
    st.subheader("📉 Outlier Detection & Removal")
    num_cols = df.select_dtypes(include='number').columns
    z_scores = stats.zscore(df[num_cols])
    outliers = (np.abs(z_scores) > 3).any(axis=1)
    outlier_count = outliers.sum()

    if outlier_count > 0:
        st.warning(f"{outlier_count} outliers removed.")
        df = df[~outliers]
    else:
        st.success("✅ No outliers found.")

    # Auto Data Type Correction
    st.subheader("📐 Auto Data Type Detection & Correction")
    correction_log = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                correction_log.append((col, "object", "datetime"))
            except:
                try:
                    df[col] = pd.to_numeric(df[col])
                    correction_log.append((col, "object", "numeric"))
                except:
                    pass
    if correction_log:
        st.success("✅ Data Type Corrections")
        st.dataframe(pd.DataFrame(correction_log, columns=["Column", "Old Type", "New Type"]))
    else:
        st.info("No data type changes needed.")

    # Categorical Standardization
    df = standardize_categories(df)

    # Heatmap
    st.subheader("🔥 Heatmap of Correlation")
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.markdown("✅ **Interpretation:** Values close to +1 or -1 show strong correlation. Zero means no relation.")
    else:
        st.info("Not enough numeric columns for heatmap.")

    # Final Summary
    st.subheader("📊 Final Summary")
    st.markdown(f"- Rows: `{df.shape[0]}`")
    st.markdown(f"- Columns: `{df.shape[1]}`")
    st.markdown(f"- Removed Duplicates: `{dup_count}`")
    st.markdown(f"- Removed Constant Columns: `{len(const_cols)}`")
    st.markdown(f"- Outliers Removed: `{outlier_count}`")

    # Download
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button("⬇️ Download Cleaned CSV", buffer.getvalue(), file_name="cleaned_dataset.csv", mime="text/csv")
