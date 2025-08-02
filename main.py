with st.expander("📜 Xem toàn bộ mã nguồn"):
    st.code("""
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

st.title("📦 Order Analytics & Revenue Forecast")

@st.cache
def load_data():
    df = pd.read_csv("orders_sample_with_stock.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Total"] = df["Quantity"] * df["Price"]
    return df

df = load_data()
st.subheader("📊 Dữ liệu đầu vào")
st.dataframe(df.head())

st.subheader("1️⃣ Đơn hàng theo ngày")
order_count = df["Date"].value_counts().sort_index()
st.line_chart(order_count)

st.subheader("2️⃣ Doanh thu theo sản phẩm")
revenue = df.groupby("Product")["Total"].sum().sort_values()
st.bar_chart(revenue)

st.subheader("3️⃣ Tồn kho trung bình theo sản phẩm")
avg_stock = df.groupby("Product")["Stock"].mean().sort_values()
st.bar_chart(avg_stock)

st.subheader("4️⃣ Phân bố giá sản phẩm")
fig1, ax1 = plt.subplots()
sns.histplot(df["Price"], kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("5️⃣ Quantity vs Total")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="Quantity", y="Total", hue="Product", ax=ax2)
st.pyplot(fig2)

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
    """, language="python")
