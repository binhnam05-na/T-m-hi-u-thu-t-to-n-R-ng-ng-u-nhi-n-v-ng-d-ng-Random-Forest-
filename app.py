import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as object_graph
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(
    page_title="S&P 500 Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values(['Name', 'date'])
    return df

@st.cache_data
def train_rf_model(df, ticker, start_date, end_date, n_estimators, max_depth, min_samples_split):
    stock_df = df[(df['Name'] == ticker) & (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))].copy()
    
    if len(stock_df) < 15:
        return None, None, None, None, None, stock_df
        
    stock_df['open_lag1'] = stock_df['open'].shift(1)
    stock_df['high_lag1'] = stock_df['high'].shift(1)
    stock_df['low_lag1'] = stock_df['low'].shift(1)
    stock_df['close_lag1'] = stock_df['close'].shift(1)
    stock_df['volume_lag1'] = stock_df['volume'].shift(1)
    stock_df = stock_df.dropna()
    
    if len(stock_df) < 10:
        return None, None, None, None, None, stock_df
        
    features = ['open_lag1', 'high_lag1', 'low_lag1', 'close_lag1', 'volume_lag1']
    X = stock_df[features]
    y = stock_df['close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    rf = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    importance_df = pd.DataFrame({
        'Feature': ['Open Price (T-1)', 'High Price (T-1)', 'Low Price (T-1)', 'Close Price (T-1)', 'Volume (T-1)'],
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    return rf, metrics, importance_df, y_test, y_pred, stock_df

try:
    df = load_and_preprocess_data('all_stocks_5yr.csv')
    tickers = sorted(df['Name'].unique().tolist())
except Exception:
    st.error("Không tìm thấy tệp 'all_stocks_5yr.csv'. Vui lòng đặt tệp dữ liệu cùng cấp với mã nguồn ứng dụng.")
    st.stop()

# --- CSS TOÀN CỤC CHUYỂN CHỮ SANG MÀU SÁNG ĐỂ ĐỌC TRÊN NỀN RÕ ANH ---
st.markdown("""
    <style>
        /* Ép toàn bộ chữ trong app (trừ sidebar) sang màu trắng và đổ bóng nhẹ */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
            color: #FFFFFF !important;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8) !important;
        }
        /* Định dạng các khối nhập liệu thành dạng kính mờ (Glassmorphism) */
        div[data-testid="stBlock"] {
            background-color: rgba(0, 0, 0, 0.4) !important;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        /* Định dạng lại thanh Tabs điều hướng cho nổi bật trên nền ảnh */
        button[data-baseweb="tab"] {
            color: #E2E8F0 !important;
            background-color: rgba(0, 0, 0, 0.5) !important;
            border-radius: 5px 5px 0 0;
            margin-right: 4px;
            padding: 10px 20px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        /* Làm mờ nhẹ nền của bảng Dataframe để vừa thấy ảnh vừa đọc được số */
        .stDataFrame div {
            background-color: rgba(255, 255, 255, 0.85) !important;
        }
        .stDataFrame span, .stDataFrame p {
            color: #1E293B !important;
            text-shadow: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<h2 style="color:#1E3A8A; font-weight:bold; text-shadow:none !important;">🛠️ CẤU HÌNH RANDOM FOREST</h2>', unsafe_allow_html=True)
selected_ticker = st.sidebar.selectbox("Chọn mã cổ phiếu S&P 500", tickers, index=tickers.index('AAL') if 'AAL' in tickers else 0)

ticker_full_data = df[df['Name'] == selected_ticker].sort_values('date')
min_date = ticker_full_data['date'].min().date()
max_date = ticker_full_data['date'].max().date()

st.sidebar.markdown('<h3 style="color:#4B5563; font-weight:600; margin-top:15px; text-shadow:none !important;">📅 LỌC KHOẢNG THỜI GIAN</h3>', unsafe_allow_html=True)
date_range = st.sidebar.date_input(
    "Chọn khoảng thời gian phân tích",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

n_estimators = st.sidebar.slider("Số lượng cây quyết định (n_estimators)", min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider("Độ sâu tối đa của cây (max_depth)", min_value=3, max_value=30, value=15, step=1)
min_samples_split = st.sidebar.slider("Mẫu tối thiểu tách nút (min_samples_split)", min_value=2, max_value=10, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown('<h3 style="color:#4B5563; font-weight:600; text-shadow:none !important;">Thông số Mô hình gốc</h3>', unsafe_allow_html=True)
st.sidebar.info("Mô hình sử dụng cơ chế Trễ 1 Phiên (Lag-1) trên các đặc trưng Open, High, Low, Close, Volume để thực hiện dự báo mức giá Close hiện tại.")

rf_model, metrics, importance_df, y_test, y_pred, filtered_df = train_rf_model(
    df, selected_ticker, start_date, end_date, n_estimators, max_depth, min_samples_split
)

# Chuyển tiêu đề chính sang màu trắng sáng để nổi trên ảnh rõ nét
st.markdown(f'<h1 style="text-align:center; color:#FFFFFF; font-weight:800; margin-bottom:20px; text-shadow: 2px 2px 8px rgba(0,0,0,0.9) !important;">📈 HỆ THỐNG DỰ BÁO GIÁ CỔ PHIẾU S&P 500</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:18px; color:#F1F5F9; text-shadow: 1px 1px 4px rgba(0,0,0,0.9) !important;">Phân tích và dự báo mã cổ phiếu: <strong style="color:#60A5FA;">{selected_ticker}</strong> trong giai đoạn từ <strong style="color:#34D399;">{start_date}</strong> đến <strong style="color:#34D399;">{end_date}</strong></p>', unsafe_allow_html=True)

if rf_model is None:
    st.error("⚠️ Không đủ dữ liệu trong khoảng thời gian đã chọn để huấn luyện mô hình. Vui lòng mở rộng khoảng thời gian ở thanh bên (Sidebar).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Dự báo & Kết quả", "📊 Trực quan hóa dữ liệu", "📋 Dữ liệu mẫu (Top 10)"])

with tab1:
    # 🌟 HIỆN RÕ ẢNH 1: Hạ lớp phủ tối xuống rất thấp (chỉ còn 15% để nhận diện chiều sâu ảnh)
    st.markdown("""
        <style>
            .stApp {
                background-image: linear-gradient(rgba(0, 0, 0, 0.15), rgba(0, 0, 0, 0.15)), 
                                  url('https://images.pexels.com/photos/30915372/pexels-photo-30915372.jpeg?_gl=1*1l16bn4*_ga*MTYxOTc0NDI5NS4xNzgxNzYzMzU3*_ga_8JE65Q40S6*czE3ODE3NjMzNTYkbzEkZzEkdDE3ODE3NjM0MDYkajEwJGwwJGgw');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
        </style>
    """, unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown('<div style="background-color:rgba(0, 0, 0, 0.5); padding:20px; border-radius:10px; border-left: 5px solid #2563EB;">'
                    '<h3 style="margin-top:0; color:#FFFFFF; text-shadow: 1px 1px 2px black !important;">📥 NHẬP DỮ LIỆU ĐẦU VÀO (PHIÊN T-1)</h3>'
                    '</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        
        latest_row = filtered_df.iloc[-1]
        
        open_val = st.number_input("Giá mở cửa phiên trước (open)", value=float(latest_row['open']), format="%.2f")
        high_val = st.number_input("Giá cao nhất phiên trước (high)", value=float(latest_row['high']), format="%.2f")
        low_val = st.number_input("Giá thấp nhất phiên trước (low)", value=float(latest_row['low']), format="%.2f")
        close_val = st.number_input("Giá đóng cửa phiên trước (close)", value=float(latest_row['close']), format="%.2f")
        volume_val = st.number_input("Khối lượng giao dịch phiên trước (volume)", value=int(latest_row['volume']), step=1000)
        
        predict_clicked = st.button("🚀 Bắt đầu dự đoán giá", use_container_width=True)
        
    with col_output:
        st.markdown('<div style="background-color:rgba(0, 0, 0, 0.5); padding:20px; border-radius:10px; border-left: 5px solid #10B981;">'
                    '<h3 style="margin-top:0; color:#FFFFFF; text-shadow: 1px 1px 2px black !important;">📤 THÔNG SỐ ĐẦU RA & ĐÁNH GIÁ MÔ HÌNH</h3>'
                    '</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        
        if predict_clicked:
            input_features = np.array([[open_val, high_val, low_val, close_val, volume_val]])
            prediction = rf_model.predict(input_features)[0]
            
            st.metric(
                label=f"GIÁ ĐÓNG CỬA DỰ BÁO TIẾP THEO ({selected_ticker})",
                value=f"${prediction:.2f}",
                delta=f"{(prediction - close_val):.2f} so với phiên trước"
            )
            st.markdown("---")
        else:
            st.warning("Vui lòng nhấn nút 'Bắt đầu dự đoán giá' ở cột bên trái để xem kết quả dự báo.")
            st.markdown("---")
            
        st.markdown('<p style="font-weight:600; color:#FFFFFF; margin-bottom:5px;">Chỉ số đánh giá độ chính xác trong khoảng thời gian này:</p>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score (Độ chuẩn xác)", f"{metrics['R2']:.4f}")
        m_col2.metric("MAE (Sai số tuyệt đối)", f"{metrics['MAE']:.4f}")
        m_col3.metric("MSE (Sai số bình phương)", f"{
