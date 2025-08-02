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

# Cấu hình layout Streamlit
st.set_page_config(page_title="Order Analytics", layout="wide")
st.title("📦 Order Analytics & Revenue Forecast")

# ---------------------- Load dữ liệu ----------------------
@st.cache_data
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

# ✔️ Dòng lỗi đã được sửa
st.success("✔️ Hoàn tất tiền xử lý")

# Xem dữ liệu
st.dataframe(df.head())

# ---------------------- Trực quan hoá ----------------------
st.header("📊 Phân tích dữ liệu")

st.subheader("1️⃣ Đơn hàng theo ngày")
st.line_chart(df["Date"].value_counts().sort_index())

st.subheader("2️⃣ Doanh thu theo sản phẩm")
st.bar_chart(df.groupby("Product")["Total"].sum().sort_values())

st.subheader("3️⃣ Tồn kho trung bình theo sản phẩm")
st.bar_chart(df.groupby("Product")["Stock"].mean().sort_values())

st.subheader("4️⃣ Phân bố giá bán")
fig1, ax1 = plt.subplots()
sns.histplot(df["Price"], bins=20, kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("5️⃣ Quantity vs Total")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="Quantity", y="Total", hue="Product", ax=ax2)
st.pyplot(fig2)

# ---------------------- Dự đoán ----------------------
st.header("🤖 Dự đoán tổng tiền (Linear Regression)")

X = df[["Quantity", "Price", "Product"]]
y = df["Total"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Product"])
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# ✅ Dòng lỗi đã được sửa hoàn chỉnh
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

st.metric("📉 MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
st.metric("📈 R-squared", f"{r2_score(y_test, y_pred):.2f}")

# ---------------------- Mã nguồn ----------------------
with st.expander("📜 Xem mã nguồn"):
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            code = f.read()
        st.code(code, language="python")
    except:
        st.info("Không thể hiển thị mã nguồn khi chạy trên nền tảng cloud.")
