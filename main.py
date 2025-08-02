import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu
df = pd.read_csv("orders_sample_with_stock.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df["Total"] = df["Quantity"] * df["Price"]

# Biểu đồ 1: Đơn hàng theo ngày
df["Date"].value_counts().sort_index().plot(title="Số lượng đơn hàng theo ngày")
plt.xlabel("Ngày")
plt.ylabel("Số đơn hàng")
plt.show()

# Biểu đồ 2: Doanh thu theo sản phẩm
df.groupby("Product")["Total"].sum().sort_values().plot(kind="barh", title="Tổng doanh thu theo sản phẩm")
plt.xlabel("Tổng doanh thu (USD)")
plt.show()

# Biểu đồ 3: Tồn kho trung bình
df.groupby("Product")["Stock"].mean().sort_values().plot(kind="bar", title="Tồn kho trung bình theo sản phẩm")
plt.xticks(rotation=45)
plt.show()

# Biểu đồ 4: Phân bố giá
sns.histplot(df["Price"], bins=20, kde=True)
plt.title("Phân bố giá bán sản phẩm")
plt.xlabel("Giá sản phẩm")
plt.show()

# Biểu đồ 5: Quantity vs Total
sns.scatterplot(data=df, x="Quantity", y="Total", hue="Product")
plt.title("Mối quan hệ giữa Số lượng và Tổng tiền")
plt.show()

# Mô hình hồi quy tuyến tính dự đoán Total
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

# Dự đoán và đánh giá
y_pred = pipeline.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R2):", r2_score(y_test, y_pred))
