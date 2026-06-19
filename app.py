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

# --- CSS ĐỒNG BỘ NỀN VÀ KHẮC PHỤC LỖI HIỂN THỊ Ô NHẬP LIỆU ---
st.markdown("""
    <style>
        .stApp {
            background-image: linear-gradient(rgba(13, 17, 23, 0.90), rgba(13, 17, 23, 0.93)), 
                              url('https://images.pexels.com/photos/30915372/pexels-photo-30915372.jpeg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        
        .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
            color: #F0F4F8 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        section[data-testid="stSidebar"] {
            background-color: #0D1117 !important;
            border-right: 1px solid #30363D;
        }
        section[data-testid="stSidebar"] * {
            color: #C9D1D9 !important;
        }

        div[data-testid="stBlock"] {
            background-color: rgba(22, 27, 34, 0.85) !important;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #30363D;
        }
        
        button[data-baseweb="tab"] {
            color: #8B949E !important;
            background-color: #161B22 !important;
            border-radius: 6px 6px 0 0;
            margin-right: 6px;
            padding: 10px 20px !important;
            border: 1px solid #30363D !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #21262D !important;
            color: #26A69A !important; 
            font-weight: bold !important;
            border-bottom: 3px solid #26A69A !important;
        }
        
        /* ĐÃ SỬA: CSS chuẩn hóa cho các ô input giúp gõ chữ bằng tay mượt mà */
        input {
            color: #FFFFFF !important;
            background-color: #0D1117 !important;
            border: 1px solid #30363D !important;
        }
        input:focus {
            background-color: #161B22 !important;
            color: #FFFFFF !important;
            border-color: #26A69A !important;
            caret-color: #FFFFFF !important;
        }
    </style>
""", unsafe_allow_html=True)

# Giao diện Sidebar thanh công cụ
st.sidebar.markdown('<h2 style="color:#F2A900; font-weight:bold;">🛠️ CẤU HÌNH RANDOM FOREST</h2>', unsafe_allow_html=True)
selected_ticker = st.sidebar.selectbox("Chọn mã cổ phiếu S&P 500", tickers, index=tickers.index('AAL') if 'AAL' in tickers else 0)

ticker_full_data = df[df['Name'] == selected_ticker].sort_values('date')
min_date = ticker_full_data['date'].min().date()
max_date = ticker_full_data['date'].max().date()

st.sidebar.markdown('<h3 style="color:#8B949E; font-weight:600; margin-top:15px;">📅 LỌC KHOẢNG THỜI GIAN</h3>', unsafe_allow_html=True)
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
st.sidebar.info("Mô hình sử dụng cơ chế Trễ 1 Phiên (Lag-1) trên các đặc trưng Open, High, Low, Close, Volume để thực hiện dự báo mức giá Close hiện tại.")

rf_model, metrics, importance_df, y_test, y_pred, filtered_df = train_rf_model(
    df, selected_ticker, start_date, end_date, n_estimators, max_depth, min_samples_split
)

# Khối Header chính giữa trang
st.markdown(f'<h1 style="text-align:center; color:#FFFFFF; font-weight:800; margin-bottom:5px; text-shadow: 2px 2px 4px #000000;">📈 HỆ THỐNG DỰ BÁO GIÁ CỔ PHIẾU S&P 500</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:16px; color:#8B949E; margin-bottom:25px;">Phân tích mã: <strong style="color:#F2A900;">{selected_ticker}</strong> | Khung thời gian: <strong style="color:#26A69A;">{start_date}</strong> đến <strong style="color:#26A69A;">{end_date}</strong></p>', unsafe_allow_html=True)

if rf_model is None:
    st.error("⚠️ Không đủ dữ liệu trong khoảng thời gian đã chọn để huấn luyện mô hình. Vui lòng mở rộng khoảng thời gian ở thanh bên (Sidebar).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Dự báo & Kết quả", "📊 Trực quan hóa dữ liệu", "📋 Dữ liệu mẫu (Top 10)"])

# --- TAB 1: DỰ BÁO KẾT QUẢ ---
with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown('<div style="background-color:#161B22; padding:15px; border-radius:6px; border-left: 5px solid #F2A900;">'
                    '<h3 style="margin-top:0; color:#F2A900; font-size:16px; font-weight:bold;">📥 NHẬP THÔNG SỐ ĐẦU VÀO (PHIÊN T-1)</h3>'
                    '</div><br>', unsafe_allow_html=True)
        
        latest_row = filtered_df.iloc[-1]
        open_val = st.number_input("Giá mở cửa phiên trước (open)", value=float(latest_row['open']), format="%.2f")
        high_val = st.number_input("Giá cao nhất phiên trước (high)", value=float(latest_row['high']), format="%.2f")
        low_val = st.number_input("Giá thấp nhất phiên trước (low)", value=float(latest_row['low']), format="%.2f")
        close_val = st.number_input("Giá đóng cửa phiên trước (close)", value=float(latest_row['close']), format="%.2f")
        volume_val = st.number_input("Khối lượng giao dịch phiên trước (volume)", value=int(latest_row['volume']), step=1000)
        
    with col_output:
        st.markdown('<div style="background-color:#161B22; padding:15px; border-radius:6px; border-left: 5px solid #26A69A;">'
                    '<h3 style="margin-top:0; color:#26A69A; font-size:16px; font-weight:bold;">📤 KẾT QUẢ ĐẦU RA TỪ MÔ HÌNH DỰ BÁO</h3>'
                    '</div><br>', unsafe_allow_html=True)
        
        # ĐÃ SỬA: Chạy mô hình dự báo trực tiếp thời gian thực khi bất kì ô input nào thay đổi
        input_features = np.array([[open_val, high_val, low_val, close_val, volume_val]])
        prediction = rf_model.predict(input_features)[0]
        price_diff = prediction - close_val
        pct_diff = (price_diff / close_val) * 100
        
        if price_diff > 0:
            box_bg = "rgba(38, 166, 154, 0.15)"
            border_c = "#26A69A"
            text_c = "#00E676"  
            status_text = f"▲ Tăng {price_diff:+.2f} ({pct_diff:+.2f}%)"
        elif price_diff < 0:
            box_bg = "rgba(239, 83, 80, 0.15)"
            border_c = "#EF5350"
            text_c = "#FF1744"  
            status_text = f"▼ Giảm {price_diff:+.2f} ({pct_diff:+.2f}%)"
        else:
            box_bg = "rgba(242, 169, 0, 0.1)"
            border_c = "#F2A900"
            text_c = "#F2A900"
            status_text = "■ Không đổi (Bằng giá tham chiếu)"

        st.markdown(f"""
            <div style="background-color: {box_bg}; padding: 25px; border-radius: 8px; border: 2px solid {border_c}; text-align: center;">
                <p style="color: #C9D1D9; font-size: 13px; margin-bottom: 5px; font-weight: 600;">XU HƯỚNG GIÁ ĐÓNG CỬA PHIÊN TIẾP THEO</p>
                <h1 style="color: {text_c} !important; font-size: 46px !important; font-weight: 800; margin: 0; padding: 5px 0;">${prediction:.2f}</h1>
                <p style="color: {text_c} !important; font-size: 18px; margin: 5px 0 0 0; font-weight: 700; text-shadow: 1px 1px 3px rgba(0,0,0,0.8);">
                    {status_text}
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.markdown('<p style="font-weight:600; color:#8B949E; margin-bottom:12px;">Độ tin cậy của mô hình (Mẫu kiểm thử):</p>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score (Độ khớp)", f"{metrics['R2']:.4f}")
        m_col2.metric("MAE (Sai số tuyệt đối)", f"{metrics['MAE']:.4f}")
        m_col3.metric("MSE", f"{metrics['MSE']:.4f}")

# --- TAB 2: ĐỒ THỊ PHÂN TÍCH ---
with tab2:
    st.markdown('<h3 style="color:#26A69A; margin-bottom:20px; font-size:18px; font-weight:bold;">📊 ĐỒ THỊ PHÂN TÍCH KỸ THUẬT VÀ THUẬT TOÁN</h3>', unsafe_allow_html=True)
    
    chart_layout_config = dict(
        template="plotly_dark",
        paper_bgcolor='rgba(13, 17, 23, 0.7)', 
        plot_bgcolor='#0D1117',
        font=dict(color='#C9D1D9'),
        title_font=dict(color='#FFFFFF', size=18, family='Segoe UI', weight='bold'),
        
        xaxis=dict(
            gridcolor='#30363D', 
            showgrid=True, 
            linecolor='#30363D', 
            title_font=dict(color='#8B949E', size=14),
            tickfont=dict(size=13, color='#C9D1D9')
        ),
        yaxis=dict(
            gridcolor='#30363D', 
            showgrid=True, 
            linecolor='#30363D', 
            title_font=dict(color='#8B949E', size=14),
            tickfont=dict(size=13, color='#C9D1D9')
        )
    )

    # Biểu đồ 1: Biến động giá Line màu Xanh tăng - Đỏ giảm
    fig1 = object_graph.Figure()
    fig1.add_trace(object_graph.Scatter(x=filtered_df['date'], y=filtered_df['close'], name='Giá đóng cửa (Close)', line=dict(color='#00E676', width=2)))
    fig1.add_trace(object_graph.Scatter(x=filtered_df['date'], y=filtered_df['open'], name='Giá mở cửa (Open)', line=dict(color='#FF1744', width=1.5, dash='dot')))
    fig1.update_layout(title="Biểu đồ 1: Lịch sử Biến động Giá cổ phiếu (Xanh tăng / Đỏ giảm)", hovermode="x unified", **chart_layout_config)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Biểu đồ 2: Đặc trưng quan trọng
    fig2 = px.bar(
        importance_df, x='Importance', y='Feature', orientation='h',
        title="Biểu đồ 2: Mức độ quan trọng của các đặc trưng đầu vào",
        labels={'Importance': 'Độ quan trọng', 'Feature': 'Đặc trưng'},
        color='Importance', color_continuous_scale=['#EF5350', '#F2A900', '#26A69A']
    )
    fig2.update_layout(coloraxis_showscale=False, **chart_layout_config)
    fig2.update_yaxes(tickfont=dict(size=13)) 
    st.plotly_chart(fig2, use_container_width=True)
    
    # Biểu đồ 3: So sánh Thực tế vs Dự đoán
    fig3 = object_graph.Figure()
    display_length = min(100, len(y_test))
    y_test_plot = y_test.values[-display_length:]
    y_pred_plot = y_pred[-display_length:]
    x_axis = np.arange(len(y_test_plot))
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_test_plot, mode='lines', name='Thực tế', line=dict(color='#00E676', width=2.5)))
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_pred_plot, mode='lines', name='Dự đoán', line=dict(color='#FF1744', width=2, dash='dash')))
    fig3.update_layout(title=f"Biểu đồ 3: So sánh Giá trị Thực tế vs Dự đoán ({display_length} phiên cuối)", **chart_layout_config)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Biểu đồ 4: CẬP NHẬT HỆ MÀU PHÂN CỰC TRỰC QUAN CAO (ĐỎ - VÀNG - XANH LÁ)
    fig4 = px.scatter(
        filtered_df, x='volume', y='close', color='high',
        title="Biểu đồ 4: Mối tương quan giữa Khối lượng giao dịch và Giá đóng cửa",
        labels={'volume': 'Khối lượng khớp', 'close': 'Giá khớp', 'high': 'Giá cao nhất'},
        color_continuous_scale='RdYlGn' # Đỏ (Giá thấp) -> Vàng (Giá trung bình) -> Xanh lá (Giá cao)
    )
    fig4.update_layout(**chart_layout_config)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Biểu đồ 5: ĐỒNG BỘ MÀU XANH NGỌC + GRIDLINE DỌC SẮC NÉT
    filtered_df['Price_Range'] = filtered_df['high'] - filtered_df['low']
    fig5 = px.histogram(
        filtered_df, x='Price_Range', 
        title="Biểu đồ 5: Phân phối Biên độ dao động giá trong ngày (High - Low)",
        labels={'Price_Range': 'Biên độ dao động (USD)', 'count': 'Tần suất xuất hiện'},
        color_discrete_sequence=['#26A69A'], 
        nbins=50
    )
    fig5.update_layout(yaxis_title="Tần suất xuất hiện", **chart_layout_config)
    fig5.update_xaxes(showgrid=True, gridcolor='#30363D', gridwidth=1.5)
    st.plotly_chart(fig5, use_container_width=True)

# --- TAB 3: HIỂN THỊ DỮ LIỆU ---
with tab3:
    st.markdown(f'<h3 style="color:#F2A900; font-weight:bold;">📋 10 DÒNG DỮ LIỆU ĐẦU TIÊN CỦA KHOẢNG THỜI GIAN ĐÃ CHỌN ({selected_ticker})</h3>', unsafe_allow_html=True)
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    st.markdown('<h3 style="color:#F2A900; margin-top:30px; font-weight:bold;">📊 Thống kê mô tả tổng quan trong khoảng thời gian này</h3>', unsafe_allow_html=True)
    st.dataframe(filtered_df.describe(), use_container_width=True)
