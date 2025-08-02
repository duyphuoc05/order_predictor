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

st.set_page_config(page_title="Order Analytics", layout="wide")
st.title("📦 Order Analytics & Revenue Forecast")

# Hiển thị hàm load_data
st.code("""
def load_data():
    df = pd.read_csv("orders_sample_with_stock.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Total"] = df["Quantity"] * df["Price"]
    return df
""", language="python")

# Load data
@st.cache
def load_data():
    df = pd.read_csv("orders_sample_with_stock.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Total"] = df["Quantity"] * df["Price"]
    return df

df = load_data()

# ---------------------- TIỀN XỬ LÝ ----------------------
st.subheader("🧼 Tiền xử lý dữ liệu")

# 1. Xóa null
st.markdown("### 🧪 1. Xóa các giá trị Null (nếu có)")
st.code("""
# Xóa hàng chứa null
df = df.dropna()
""", language="python")
df = df.dropna()

# 2. Xoá trùng lặp
st.markdown("### ♻️ 2. Xóa dòng trùng lặp")
st.code("""
# Xóa dòng trùng
df = df.drop_duplicates()
""", language="python")
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
st.write(f"Số dòng trùng lặp đã bị xóa: **{duplicates}**")

# 3. Chuẩn hóa tên sản phẩm
st.markdown("### 🔤 3. Chuẩn hóa tên sản phẩm (chữ thường)")
st.code("""
df["Product"] = df["Product"].str.lower()
""", language="python")
df["Product"] = df["Product"].str.lower()

# 4. Reset index
st.markdown("### 🔁 4. Reset lại chỉ số dòng")
st.code("""
df = df.reset_index(drop=True)
""", language="python")
df = df.reset_index(drop=True)

# ---------------------- TRỰC QUAN HÓA ----------------------
st.subheader("📊 Dữ liệu sau tiền xử lý")
st.dataframe(df.head())

# 1. Orders per day
st.subheader("1️⃣ Đơn hàng theo ngày")
order_count = df["Date"].value_counts().sort_index()
st.line_chart(order_count)

# 2. Total revenue by product
st.subheader("2️⃣ Doanh thu theo sản phẩm")
revenue = df.groupby("Product")["Total"].sum().sort_values()
st.bar_chart(revenue)

# 3. Average stock
st.subheader("3️⃣ Tồn kho trung bình theo sản phẩm")
avg_stock = df.groupby("Product")["Stock"].mean().sort_values()
st.bar_chart(avg_stock)

# 4. Price distribution
st.subheader("4️⃣ Phân bố giá sản phẩm")
fig1, ax1 = plt.subplots()
sns.histplot(df["Price"], kde=True, ax=ax1)
st.pyplot(fig1)

# 5. Scatter Quantity vs Total
st.subheader("5️⃣ Quantity vs Total")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="Quantity", y="Total", hue="Product", ax=ax2)
st.pyplot(fig2)

# ---------------------- MÔ HÌNH HỒI QUY ----------------------
st.subheader("🔍 Dự đoán tổng tiền với hồi quy tuyến tính")

X = df[["Quantity", "Price", "Product"]]
y = df["Total"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Product"])
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

st.write("📉 **MSE:**", round(mean_squared_error(y_test, y_pred), 2))
st.write("📈 **R-squared:**", round(r2_score(y_test, y_pred), 2))

# ---------------------- MÃ NGUỒN ----------------------
with st.expander("📜 Xem toàn bộ mã nguồn"):
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            source_code = f.read()
        st.code(source_code, language="python")
    except:
        st.warning("Không thể hiển thị mã nguồn khi chạy trên nền tảng cloud.")
