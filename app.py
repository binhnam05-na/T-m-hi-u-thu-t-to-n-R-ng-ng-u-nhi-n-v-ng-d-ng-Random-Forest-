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

# --- CSS ĐỒNG BỘ DARK MODE TOÀN DIỆN ---
st.markdown("""
    <style>
        .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
            color: #E2E8F0 !important;
            font-family: 'Inter', sans-serif;
        }
        
        section[data-testid="stSidebar"] {
            background-color: #0B0F19 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #94A3B8 !important;
        }

        /* Hộp kính mờ bo góc cao cấp */
        div[data-testid="stBlock"] {
            background-color: rgba(15, 23, 42, 0.7) !important;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        }
        
        /* Menu lựa chọn Tabs điều hướng */
        button[data-baseweb="tab"] {
            color: #64748B !important;
            background-color: rgba(15, 23, 42, 0.8) !important;
            border-radius: 6px 6px 0 0;
            margin-right: 4px;
            padding: 10px 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #1E3A8A !important;
            color: #F8FAFC !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #60A5FA !important;
        }
        
        /* Tinh chỉnh DataFrame cho hợp nền tối */
        .stDataFrame div {
            background-color: rgba(15, 23, 42, 0.95) !important;
        }
        .stDataFrame span, .stDataFrame p {
            color: #E2E8F0 !important;
            text-shadow: none !important;
        }
        
        input {
            color: #FFFFFF !important;
            background-color: #0F172A !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<h2 style="color:#60A5FA; font-weight:bold;">🛠️ CẤU HÌNH RANDOM FOREST</h2>', unsafe_allow_html=True)
selected_ticker = st.sidebar.selectbox("Chọn mã cổ phiếu S&P 500", tickers, index=tickers.index('AAL') if 'AAL' in tickers else 0)

ticker_full_data = df[df['Name'] == selected_ticker].sort_values('date')
min_date = ticker_full_data['date'].min().date()
max_date = ticker_full_data['date'].max().date()

st.sidebar.markdown('<h3 style="color:#64748B; font-weight:600; margin-top:15px;">📅 LỌC KHOẢNG THỜI GIAN</h3>', unsafe_allow_html=True)
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
st.sidebar.markdown('<h3 style="color:#64748B; font-weight:600;">Thông số Mô hình gốc</h3>', unsafe_allow_html=True)
st.sidebar.info("Mô hình sử dụng cơ chế Trễ 1 Phiên (Lag-1) trên các đặc trưng Open, High, Low, Close, Volume để thực hiện dự báo mức giá Close hiện tại.")

rf_model, metrics, importance_df, y_test, y_pred, filtered_df = train_rf_model(
    df, selected_ticker, start_date, end_date, n_estimators, max_depth, min_samples_split
)

st.markdown(f'<h1 style="text-align:center; color:#FFFFFF; font-weight:800; margin-bottom:10px; text-shadow: 2px 2px 12px rgba(0,0,0,0.8) !important;">📈 HỆ THỐNG DỰ BÁO GIÁ CỔ PHIẾU S&P 500</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:16px; color:#94A3B8; margin-bottom:25px;">Phân tích mã cổ phiếu: <strong style="color:#60A5FA;">{selected_ticker}</strong> | Giai đoạn: <strong style="color:#34D399;">{start_date}</strong> đến <strong style="color:#34D399;">{end_date}</strong></p>', unsafe_allow_html=True)

if rf_model is None:
    st.error("⚠️ Không đủ dữ liệu trong khoảng thời gian đã chọn để huấn luyện mô hình. Vui lòng mở rộng khoảng thời gian ở thanh bên (Sidebar).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Dự báo & Kết quả", "📊 Trực quan hóa dữ liệu", "📋 Dữ liệu mẫu (Top 10)"])

with tab1:
    st.markdown("""
        <style>
            .stApp {
                background-image: linear-gradient(rgba(10, 15, 26, 0.55), rgba(10, 15, 26, 0.55)), 
                                  url('https://images.pexels.com/photos/30915372/pexels-photo-30915372.jpeg?_gl=1*1l16bn4*_ga*MTYxOTc0NDI5NS4xNzgxNzYzMzU3*_ga_8JE65Q40S6*czE3ODE3NjMzNTYkbzEkZzEkdDE3ODE3NjM0MDYkajEwJGwwJGgw');
                background-size: cover; background-position: center; background-attachment: fixed;
            }
        </style>
    """, unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown('<div style="background-color:rgba(20, 30, 54, 0.8); padding:20px; border-radius:10px; border-left: 5px solid #3B82F6;">'
                    '<h3 style="margin-top:0; color:#FFFFFF; font-size:18px;">📥 NHẬP DỮ LIỆU ĐẦU VÀO (PHIÊN T-1)</h3>'
                    '</div><br>', unsafe_allow_html=True)
        
        latest_row = filtered_df.iloc[-1]
        open_val = st.number_input("Giá mở cửa phiên trước (open)", value=float(latest_row['open']), format="%.2f")
        high_val = st.number_input("Giá cao nhất phiên trước (high)", value=float(latest_row['high']), format="%.2f")
        low_val = st.number_input("Giá thấp nhất phiên trước (low)", value=float(latest_row['low']), format="%.2f")
        close_val = st.number_input("Giá đóng cửa phiên trước (close)", value=float(latest_row['close']), format="%.2f")
        volume_val = st.number_input("Khối lượng giao dịch phiên trước (volume)", value=int(latest_row['volume']), step=1000)
        
        predict_clicked = st.button("🚀 Bắt đầu dự đoán giá", use_container_width=True)
        
    with col_output:
        st.markdown('<div style="background-color:rgba(20, 30, 54, 0.8); padding:20px; border-radius:10px; border-left: 5px solid #10B981;">'
                    '<h3 style="margin-top:0; color:#FFFFFF; font-size:18px;">📤 THÔNG SỐ ĐẦU RA & ĐÁNH GIÁ MÔ HÌNH</h3>'
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
            
        st.markdown('<p style="font-weight:600; color:#94A3B8; margin-bottom:12px;">Chỉ số đánh giá độ chính xác trong khoảng thời gian này:</p>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score (Độ chuẩn xác)", f"{metrics['R2']:.4f}")
        m_col2.metric("MAE (Sai số tuyệt đối)", f"{metrics['MAE']:.4f}")
        m_col3.metric("MSE (Sai số bình phương)", f"{metrics['MSE']:.4f}")

with tab2:
    st.markdown("""
        <style>
            .stApp {
                background-image: linear-gradient(rgba(10, 15, 26, 0.6), rgba(10, 15, 26, 0.6)), 
                                  url('https://images.pexels.com/photos/6772076/pexels-photo-6772076.jpeg?_gl=1*1jtuazw*_ga*MTYxOTc0NDI5NS4xNzgxNzYzMzU3*_ga_8JE65Q40S6*czE3ODE3NjMzNTYkbzEkZzEkdDE3ODE3NjM0MDYkajEwJGwwJGgw');
                background-size: cover; background-position: center; background-attachment: fixed;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h3 style="color:#FFFFFF; margin-bottom:20px; font-size:20px;">📉 HỆ THỐNG TRỰC QUAN HÓA THÔNG MINH (PASTEL THEME)</h3>', unsafe_allow_html=True)
    
    # --- 🌟 KHUNG GIAO DIỆN MÀU HÀI HÒA CHO MÀN HÌNH TỐI ---
    dark_harmony_layout = dict(
        paper_bgcolor='rgba(22, 28, 45, 0.75)',  # Hộp kính màu xanh đen sâu thẳm dịu mắt
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#CBD5E1', size=11),      # Chữ màu xám trắng mềm mịn không bị chói mắt
        title_font=dict(color='#F1F5F9', size=14),
        hoverlabel=dict(bgcolor='#1E293B', font_size=12, font_color='#F8FAFC'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', title_font=dict(color='#94A3B8'), tickfont=dict(color='#94A3B8'), linecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title_font=dict(color='#94A3B8'), tickfont=dict(color='#94A3B8'), linecolor='rgba(255,255,255,0.1)')
    )

    # Biểu đồ 1: Màu xanh ngọc lục bảo pastel (Dịu hơn màu xanh thuần)
    fig1 = object_graph.Figure()
    fig1.add_trace(object_graph.Scatter(x=filtered_df['date'], y=filtered_df['close'], name='Giá đóng cửa', line=dict(color='#4ADE80', width=2)))
    fig1.add_trace(object_graph.Scatter(x=filtered_df['date'], y=filtered_df['open'], name='Giá mở cửa', line=dict(color='#94A3B8', width=1, dash='dot')))
    fig1.update_layout(title=f"Biểu đồ 1: Biến động giá cổ phiếu {selected_ticker}", hovermode="x unified", **dark_harmony_layout)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Biểu đồ 2: Sử dụng bảng màu Muted Teal khói sang trọng
    fig2 = px.bar(
        importance_df, x='Importance', y='Feature', orientation='h',
        title="Biểu đồ 2: Mức độ quan trọng của các đặc trưng đầu vào",
        labels={'Importance': 'Độ quan trọng', 'Feature': 'Đặc trưng'},
        color='Importance', color_continuous_scale=['#312E81', '#1E40AF', '#60A5FA'] # Gradient xanh thanh lịch từ tối sang sáng
    )
    fig2.update_layout(coloraxis_showscale=False, **dark_harmony_layout)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Biểu đồ 3: Đỏ pastel phối Xanh ngọc mint (Cặp màu đối lập kinh điển nhưng hạ sắc độ)
    fig3 = object_graph.Figure()
    display_length = min(100, len(y_test))
    y_test_plot = y_test.values[-display_length:]
    y_pred_plot = y_pred[-display_length:]
    x_axis = np.arange(len(y_test_plot))
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_test_plot, mode='lines', name='Thực tế', line=dict(color='#2DD4BF', width=2))) # Mint
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_pred_plot, mode='lines', name='Dự đoán', line=dict(color='#F87171', width=1.8, dash='dash'))) # Coral Red pastel
    fig3.update_layout(title=f"Biểu đồ 3: So sánh Thực tế vs Dự đoán ({display_length} phiên cuối)", **dark_harmony_layout)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Biểu đồ 4: Scatter sử dụng dải màu khói mờ Ice & Fire (Hợp nhãn màn hình tối)
    fig4 = px.scatter(
        filtered_df, x='volume', y='close', color='high',
        title="Biểu đồ 4: Mối tương quan giữa Khối lượng giao dịch và Giá đóng cửa",
        labels={'volume': 'Khối lượng', 'close': 'Giá đóng cửa', 'high': 'Giá cao nhất'},
        color_continuous_scale='IceFire'
    )
    fig4.update_layout(**dark_harmony_layout)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Biểu đồ 5: Tím Lavender Pastel lãng mạn giúp làm dịu thị giác
    filtered_df['Price_Range'] = filtered_df['high'] - filtered_df['low']
    fig5 = px.histogram(
        filtered_df, x='Price_Range', 
        title="Biểu đồ 5: Phân phối Biên độ dao động giá trong ngày (High - Low)",
        labels={'Price_Range': 'Biên độ dao động (USD)'},
        color_discrete_sequence=['#C084FC'], nbins=50 # Lavender pastel
    )
    fig5.update_layout(yaxis_title="Tần suất", **dark_harmony_layout)
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.markdown("""
        <style>
            .stApp {
                background-image: linear-gradient(rgba(10, 15, 26, 0.6), rgba(10, 15, 26, 0.6)), 
                                  url('https://cdn.vietnambiz.vn/2019/11/21/947c8cad98ec71b228fd-15743030298821491913597.jpg');
                background-size: cover; background-position: center; background-attachment: fixed;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<h3>📋 10 DÒNG DỮ LIỆU ĐẦU TIÊN CỦA KHOẢNG THỜI GIAN ĐÃ CHỌN ({selected_ticker})</h3>', unsafe_allow_html=True)
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    st.markdown('<h3 style="margin-top:30px;">📊 Thống kê mô tả tổng quan trong khoảng thời gian này</h3>', unsafe_allow_html=True)
    st.dataframe(filtered_df.describe(), use_container_width=True)
