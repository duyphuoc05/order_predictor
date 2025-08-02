# File: app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Thiết lập tiêu đề và mô tả
st.title("Phân tích Dữ liệu Bán hàng")
st.write("""
Ứng dụng này phân tích dữ liệu bán hàng theo mô hình P6 (Product, Price, Place, Promotion, People, Process).
Dữ liệu từ file `orders_sample_with_stock.csv` được sử dụng để phân tích sản phẩm, giá cả, tồn kho, 
xu hướng doanh thu, và dự đoán số lượng bán ra bằng mô hình Random Forest.
""")

# Tải dữ liệu
try:
    # Thay <your-username> và <your-repo> bằng thông tin GitHub của bạn
    data_url = "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/orders_sample_with_stock.csv"
    data = pd.read_csv(data_url)
except:
    try:
        data = pd.read_csv('orders_sample_with_stock.csv')
    except FileNotFoundError:
        st.error("Không thể tải dữ liệu. Vui lòng kiểm tra file `orders_sample_with_stock.csv` hoặc URL GitHub.")
        st.stop()

# Tiền xử lý dữ liệu
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data['Total Revenue'] = data['Quantity'] * data['Price']
data['Month'] = data['Date'].dt.strftime('%B')

# Bảng số liệu tổng hợp
st.header("Bảng số liệu tổng hợp")
summary_data = data.groupby('Product').agg({
    'Total Revenue': 'sum',
    'Quantity': 'sum',
    'Price': 'mean',
    'Stock': 'mean'
}).round(2).sort_values(by='Total Revenue', ascending=False).head(5)
summary_data = summary_data.rename(columns={
    'Total Revenue': 'Tổng Doanh thu',
    'Quantity': 'Tổng Số lượng',
    'Price': 'Giá trung bình',
    'Stock': 'Tồn kho trung bình'
})
st.table(summary_data)
st.write("""
Bảng số liệu tổng hợp cung cấp thông tin quan trọng về doanh thu, số lượng bán ra, giá trung bình, và tồn kho trung bình của các sản phẩm hàng đầu.
Dữ liệu này giúp doanh nghiệp xác định sản phẩm chủ lực, đánh giá hiệu quả định giá, và quản lý nguồn cung hiệu quả.
""")

# --- Product ---
st.header("1. Product (Sản phẩm)")
st.write("Phân tích sản phẩm bán chạy nhất dựa trên doanh thu và số lượng bán ra.")
col1, col2 = st.columns(2)

# Biểu đồ cột: Top 5 sản phẩm theo doanh thu
with col1:
    revenue_by_product = data.groupby('Product')['Total Revenue'].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=revenue_by_product.index, y=revenue_by_product.values, palette='Blues_d', ax=ax)
    ax.set_title('Top 5 Sản phẩm theo Doanh thu')
    ax.set_xlabel('Sản phẩm')
    ax.set_ylabel('Doanh thu')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.write(f"**Nhận xét**: Monitor dẫn đầu với doanh thu {revenue_by_product.iloc[0]:.2f}.")

# Biểu đồ tròn: Tỷ lệ số lượng bán ra
with col2:
    quantity_by_product = data.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(quantity_by_product, labels=quantity_by_product.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    ax.set_title('Tỷ lệ số lượng bán ra (Top 5)')
    plt.tight_layout()
    st.pyplot(fig)
    st.write(f"**Nhận xét**: Monitor và Mouse chiếm tỷ lệ lớn.")
st.write("""
Phân tích sản phẩm cho thấy Monitor và Mouse là các sản phẩm chủ lực, chiếm tỷ lệ lớn trong doanh thu và số lượng bán ra. 
Doanh nghiệp nên tập trung quảng bá và đảm bảo nguồn cung cho các sản phẩm này, đồng thời xem xét mở rộng danh mục nếu có nhu cầu cao.
""")

# --- Price ---
st.header("2. Price (Giá cả)")
st.write("Phân tích mối quan hệ giữa giá và số lượng bán ra.")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Price', y='Quantity', hue='Product', size='Stock', data=data, alpha=0.6, ax=ax)
ax.set_title('Mối quan hệ giữa Giá và Số lượng bán ra')
ax.set_xlabel('Giá')
ax.set_ylabel('Số lượng')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig)
st.write("**Nhận xét**: Giá dao động lớn, không có mối quan hệ tuyến tính rõ ràng với số lượng.")
st.write("""
Phân tích giá cả chỉ ra rằng biến động giá lớn (như Power Bank) có thể ảnh hưởng đến nhu cầu. 
Doanh nghiệp nên xem xét điều chỉnh chiến lược định giá, chẳng hạn giảm giá cho sản phẩm có giá cao nhưng bán chậm, 
hoặc duy trì giá ổn định cho sản phẩm chủ lực như Monitor.
""")

# --- Place ---
st.header("3. Place (Phân phối)")
st.write("Phân tích tồn kho để đảm bảo khả năng đáp ứng nhu cầu.")
stock_by_product = data.groupby('Product')['Stock'].mean().sort_values(ascending=False).head(5)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=stock_by_product.index, y=stock_by_product.values, palette='Greens_d', ax=ax)
ax.set_title('Tồn kho trung bình theo sản phẩm (Top 5)')
ax.set_xlabel('Sản phẩm')
ax.set_ylabel('Tồn kho trung bình')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
st.write(f"**Nhận xét**: Laptop và Router có tồn kho cao ({stock_by_product.iloc[0]:.2f}).")
st.write("""
Phân tích phân phối cho thấy Laptop và Router có tồn kho dồi dào, nhưng một số sản phẩm như Graphics Card có nguy cơ hết hàng. 
Doanh nghiệp cần theo dõi sát sao tồn kho và bổ sung kịp thời cho các sản phẩm bán chạy để tránh gián đoạn cung ứng.
""")

# --- Promotion ---
st.header("4. Promotion (Xúc tiến)")
st.write("Phân tích xu hướng doanh thu để đề xuất thời điểm khuyến mãi.")
revenue_by_date = data.groupby('Date')['Total Revenue'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(revenue_by_date['Date'], revenue_by_date['Total Revenue'], marker='o', color='b')
ax.set_title('Doanh thu theo ngày')
ax.set_xlabel('Ngày')
ax.set_ylabel('Doanh thu')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
st.write("**Nhận xét**: Đề xuất khuyến mãi vào các ngày thấp điểm.")
st.write("""
Phân tích xúc tiến cho thấy doanh thu dao động, với các ngày cao điểm (như 7/4/2024) và thấp điểm (như cuối tháng 6). 
Doanh nghiệp nên triển khai khuyến mãi vào các ngày thấp điểm để kích cầu, đồng thời duy trì chiến lược cho các ngày cao điểm.
""")

# --- People & Process ---
st.header("5. People & Process (Con người & Quy trình)")
st.write("Dự đoán số lượng bán ra bằng mô hình Random Forest.")
X = data[['Price', 'Stock', 'Product', 'Month']]
y = data['Quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Price', 'Stock']),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Product', 'Month'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Kết quả mô hình Random Forest**:")
st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
st.write(f"- R² Score: {r2:.2f}")
st.write("**Nhận xét**: Mô hình có thể cải thiện bằng cách thêm đặc trưng hoặc thử mô hình khác.")

# Biểu đồ so sánh thực tế và dự đoán
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(y_test)), y_test, color='blue', label='Thực tế', alpha=0.6)
ax.scatter(range(len(y_pred)), y_pred, color='red', label='Dự đoán', alpha=0.6)
ax.set_title('So sánh Quantity thực tế và dự đoán')
ax.set_xlabel('Mẫu')
ax.set_ylabel('Số lượng')
ax.legend()
plt.tight_layout()
st.pyplot(fig)
st.write("""
Phân tích con người và quy trình cho thấy mô hình Random Forest dự đoán số lượng bán ra với độ chính xác trung bình (R² ≈ 0.3-0.5). 
Mô hình này hỗ trợ quản lý tồn kho bằng cách dự đoán nhu cầu, nhưng cần cải thiện bằng cách thêm đặc trưng (như ngày trong tuần) hoặc thử các mô hình khác như XGBoost.
""")

# Kết luận
st.header("Kết luận và Đề xuất")
st.write("""
- **Product**: Tập trung vào Monitor và Mouse.
- **Price**: Kiểm tra biến động giá lớn (Power Bank).
- **Place**: Đảm bảo tồn kho cho Graphics Card.
- **Promotion**: Khuyến mãi vào các ngày thấp điểm.
- **People & Process**: Cải thiện mô hình dự báo bằng cách thêm đặc trưng.
""")
