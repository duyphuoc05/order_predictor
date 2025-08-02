import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Cấu hình giao diện
st.set_page_config(page_title="Order Analytics", layout="wide")
st.title("📦 Order Analytics & Revenue Forecast")

# ---------------------- Load dữ liệu ----------------------
@st.cache
def load_data():
    df = pd.read_csv("orders_sample_with_stock.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Total"] = df["Quantity"] * df["Price"]
    return df

df = load_data()

# ---------------------- Tiền xử lý ----------------------
st.header("🧼 Tiền xử lý dữ liệu")

# Xóa null
st.markdown("**1. Xóa giá trị null**")
df = df.dropna()

# Xoá trùng
st.markdown("**2. Xoá dòng trùng lặp**")
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
st.write(f"✅ Đã xoá `{duplicates}` dòng trùng lặp")

# Chuẩn hóa chữ thường
st.markdown("**3. Chuẩn hóa tên sản phẩm**")
df["Product"] = df["Product"].str.lower()

# Reset index
df = df.reset_index(drop=True)

st.success("✔️ Hoàn tất tiền xử lý")
st.dataframe(df.head())

# ---------------------- Trực quan hóa ----------------------
st.
