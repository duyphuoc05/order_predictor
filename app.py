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

# Cấu hình layout
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

# 1. Xoá null
st.markdown("**1. Xoá giá trị Null**")
df = df.dropna()

# 2. Xoá dòng trùng lặp
st.markdown("**2. Xoá dòng trùng lặp**")
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
st.write(f"✅ Đã xoá `{duplicates}` dòng trùng lặp")

# 3. Chuẩn hóa tên sản phẩm
st.markdown("**3. Chuẩn hóa tên sản phẩm (chữ thường)**")
df["Product"] = df["Product"].str.lower()

# 4. Reset index
df = df.reset_index(drop=True)

st.su
