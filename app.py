import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import io
import time

# === Page Configuration ===
st.set_page_config(page_title="BizPulse – Auto Data Cleaner", layout="wide")

# === Custom Styled Title ===
st.markdown("""
    <style>
        .main-title {
            font-size:40px;
            color:#4CAF50;
            text-align:center;
            font-weight:bold;
            padding: 20px 0;
        }
    </style>
    <div class="main-title">🧹 BizPulse: Smart Auto Data Cleaner</div>
""", unsafe_allow_html=True)

st.markdown("##### 🚀 Powered by **Aditya Srivastava**")

# === Sidebar Branding ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/565/565547.png", width=80)
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Designed by **Aditya Srivastava**")

# === File Upload ===
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    with st.spinner("⏳ Loading and Cleaning your Data..."):
        time.sleep(1.5)
        df = pd.read_csv(uploaded_file, encoding='latin1')

        st.subheader("🔍 Original Dataset Preview")
        st.dataframe(df.head())
        st.info(f"📊 Original Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # === Null Value Handling ===
        st.subheader("🧼 Null Value Handling")
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if null_cols.empty:
            st.success("✅ No null values found.")
        else:
            st.warning("⚠️ Null values found:")
            st.dataframe(null_cols)

            for col in null_cols.index:
                if df[col].dtype == 'object':
                    df[col].fillna("Unknown", inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)

            st.success("✅ Null values filled (Unknown/Median).")

        # === Duplicate Removal ===
        st.subheader("🔁 Duplicate Removal")
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.warning(f"⚠️ {dup_count} duplicate rows found.")
            df.drop_duplicates(inplace=True)
            st.success("✅ Duplicates removed.")
        else:
            st.success("✅ No duplicate rows.")

        # === Constant Columns ===
        st.subheader("🚫 Constant Columns Removal")
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            st.warning(f"Found constant columns: {constant_cols}")
            df.drop(columns=constant_cols, inplace=True)
            st.success("✅ Constant columns removed.")
        else:
            st.success("✅ No constant columns found.")

        # === Column Name Cleaning ===
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)
        st.success("🧹 Column names standardized.")

        # === Outlier Detection ===
        st.subheader("📉 Outlier Detection")
        num_cols = df.select_dtypes(include='number').columns
        z_scores = stats.zscore(df[num_cols])
        outliers = (np.abs(z_scores) > 3).any(axis=1)
        outlier_count = outliers.sum()

        if outlier_count > 0:
            st.warning(f"⚠️ {outlier_count} outliers detected.")
            df = df[~outliers]
            st.success("✅ Outliers removed.")
        else:
            st.success("✅ No outliers found.")

        # === Auto Data Type Detection and Correction ===
        st.subheader("📐 Auto Data Type Detection & Correction")
        correction_summary = []

        for col in df.columns:
            original_dtype = df[col].dtype

            if original_dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                    correction_summary.append((col, original_dtype, df[col].dtype, "Converted to Numeric"))
                    continue
                except:
                    pass
                try:
                    df[col] = pd.to_datetime(df[col])
                    correction_summary.append((col, original_dtype, df[col].dtype, "Converted to Datetime"))
                    continue
                except:
                    pass
            elif original_dtype in ['int64', 'float64']:
                if df[col].astype(str).str.contains('[a-zA-Z]').sum() > 0:
                    df[col] = df[col].astype(str)
                    correction_summary.append((col, original_dtype, 'object', "Converted to Object"))

        if correction_summary:
            summary_df = pd.DataFrame(correction_summary, columns=["Column", "Old Type", "New Type", "Conversion"])
            st.success("✅ Data Type Correction Summary")
            st.dataframe(summary_df)
        else:
            st.info("✅ No data type corrections needed.")

        # === Final Dtypes
        st.subheader("📋 Final Column Data Types")
        st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

        # === Heatmap for Correlation (with guidance)
        st.subheader("🌡️ Correlation Heatmap")
        if len(num_cols) > 1:
            st.markdown("""
                A **heatmap** shows how strongly numerical columns are related to each other.
                - 🔴 Red = strong positive correlation
                - 🔵 Blue = strong negative correlation
                - ⚪️ Near 0 = no correlation
            """)
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("❗ Not enough numeric columns for correlation heatmap.")

        # === Final Cleaning Summary
        st.subheader("📊 Final Cleaning Summary")
        st.markdown(f"- ✅ Rows: `{df.shape[0]}`")
        st.markdown(f"- ✅ Columns: `{df.shape[1]}`")
        st.markdown(f"- 🔁 Duplicates Removed: `{dup_count}`")
        st.markdown(f"- 🚫 Constant Columns Removed: `{len(constant_cols)}`")
        st.markdown(f"- 📉 Outliers Removed: `{outlier_count}`")

        # === Download Button
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=buffer.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            help="Download your cleaned data"
        )
