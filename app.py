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

# --- CSS ĐỒNG BỘ NỀN TỐI VỪA (DARK SATIN) & SỬA LỖI HIỂN THỊ DỮ LIỆU ---
st.markdown("""
    <style>
        /* Thiết lập hình nền chung cho toàn bộ ứng dụng kèm lớp phủ mờ tinh tế */
        .stApp {
            background-image: linear-gradient(rgba(11, 14, 20, 0.88), rgba(11, 14, 20, 0.92)), 
                              url('https://images.pexels.com/photos/30915372/pexels-photo-30915372.jpeg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        
        /* Đồng bộ chữ trắng xám có độ tương phản cao, dễ đọc */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
            color: #E2E8F0 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Cấu hình Sidebar màu đen xám của bảng giá */
        section[data-testid="stSidebar"] {
            background-color: #0B0E14 !important;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] * {
            color: #94A3B8 !important;
        }

        /* Các khối bao bọc nội dung (Container) */
        div[data-testid="stBlock"] {
            background-color: rgba(20, 26, 38, 0.8) !important;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #2D3748;
        }
        
        /* Menu Tab sàn chứng khoán */
        button[data-baseweb="tab"] {
            color: #94A3B8 !important;
            background-color: #0F172A !important;
            border-radius: 4px 4px 0 0;
            margin-right: 4px;
            padding: 8px 16px !important;
            border: 1px solid #1E293B !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #1E293B !important;
            color: #00E676 !important; 
            font-weight: bold !important;
            border-bottom: 2px solid #00E676 !important;
        }
        
        /* GIẢI QUYẾT LỖI MẤT DỮ LIỆU TAB 3: Giúp bảng rõ nét tuyệt đối */
        .stDataFrame div {
            background-color: #0F172A !important;
        }
        .stDataFrame td, .stDataFrame th, .stDataFrame span {
            color: #F8FAFC !important;
            text-shadow: none !important;
        }
        
        /* Tinh chỉnh ô nhập liệu */
        input {
            color: #FFFFFF !important;
            background-color: #0B0E14 !important;
            border: 1px solid #2A3142 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Thanh sidebar cấu hình thuật toán
st.sidebar.markdown('<h2 style="color:#EAB308; font-weight:bold;">🛠️ CẤU HÌNH RANDOM FOREST</h2>', unsafe_allow_html=True)
selected_ticker = st.sidebar.selectbox("Chọn mã cổ phiếu S&P 500", tickers, index=tickers.index('AAL') if 'AAL' in tickers else 0)

ticker_full_data = df[df['Name'] == selected_ticker].sort_values('date')
min_date = ticker_full_data['date'].min().date()
max_date = ticker_full_data['date'].max().date()

st.sidebar.markdown('<h3 style="color:#94A3B8; font-weight:600; margin-top:15px;">📅 LỌC KHOẢNG THỜI GIAN</h3>', unsafe_allow_html=True)
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

# Khu vực Header chính giữa
st.markdown(f'<h1 style="text-align:center; color:#FFFFFF; font-weight:800; margin-bottom:5px; text-shadow: 2px 2px 4px #000000;">📈 HỆ THỐNG DỰ BÁO GIÁ CỔ PHIẾU S&P 500</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:16px; color:#94A3B8; margin-bottom:25px;">Mã phân tích: <strong style="color:#EAB308;">{selected_ticker}</strong> | Tiến trình giai đoạn: <strong style="color:#00E676;">{start_date}</strong> đến <strong style="color:#00E676;">{end_date}</strong></p>', unsafe_allow_html=True)

if rf_model is None:
    st.error("⚠️ Không đủ dữ liệu trong khoảng thời gian đã chọn để huấn luyện mô hình. Vui lòng mở rộng khoảng thời gian ở thanh bên (Sidebar).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Dự báo & Kết quả", "📊 Trực quan hóa dữ liệu", "📋 Dữ liệu mẫu (Top 10)"])

# --- TAB 1: DỰ BÁO VÀ CHỈ BÁO MÀU SẮC RỰC RỠ ---
with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown('<div style="background-color:#141A26; padding:15px; border-radius:6px; border-left: 5px solid #EAB308;">'
                    '<h3 style="margin-top:0; color:#EAB308; font-size:16px; font-weight:bold;">📥 NHẬP THÔNG SỐ ĐẦU VÀO (PHIÊN T-1)</h3>'
                    '</div><br>', unsafe_allow_html=True)
        
        latest_row = filtered_df.iloc[-1]
        open_val = st.number_input("Giá mở cửa phiên trước (open)", value=float(latest_row['open']), format="%.2f")
        high_val = st.number_input("Giá cao nhất phiên trước (high)", value=float(latest_row['high']), format="%.2f")
        low_val = st.number_input("Giá thấp nhất phiên trước (low)", value=float(latest_row['low']), format="%.2f")
        close_val = st.number_input("Giá đóng cửa phiên trước (close)", value=float(latest_row['close']), format="%.2f")
        volume_val = st.number_input("Khối lượng giao dịch phiên trước (volume)", value=int(latest_row['volume']), step=1000)
        
        predict_clicked = st.button("🚀 Thực hiện tính toán dự báo giá", use_container_width=True)
        
    with col_output:
        st.markdown('<div style="background-color:#141A26; padding:15px; border-radius:6px; border-left: 5px solid #00E676;">'
                    '<h3 style="margin-top:0; color:#00E676; font-size:16px; font-weight:bold;">📤 KẾT QUẢ ĐẦU RA TỪ MÔ HÌNH DỰ BÁO</h3>'
                    '</div><br>', unsafe_allow_html=True)
        
        if predict_clicked:
            input_features = np.array([[open_val, high_val, low_val, close_val, volume_val]])
            prediction = rf_model.predict(input_features)[0]
            price_diff = prediction - close_val
            pct_diff = (price_diff / close_val) * 100
            
            # ĐỘC QUYỀN LOGIC MÀU SẮC: Xanh tăng rực - Đỏ giảm đậm chất sàn HOSE
            if price_diff > 0:
                box_bg = "rgba(0, 230, 118, 0.15)"
                border_c = "#00E676"
                text_c = "#00E676"
                status_text = f"▲ Tăng {price_diff:+.2f} ({pct_diff:+.2f}%)"
            elif price_diff < 0:
                box_bg = "rgba(255, 23, 68, 0.15)"
                border_c = "#FF1744"
                text_c = "#FF1744"
                status_text = f"▼ Giảm {price_diff:+.2f} ({pct_diff:+.2f}%)"
            else:
                box_bg = "rgba(254, 240, 138, 0.1)"
                border_c = "#EAB308"
                text_c = "#EAB308"
                status_text = "■ Không đổi (Bằng giá tham chiếu)"

            st.markdown(f"""
                <div style="background-color: {box_bg}; padding: 25px; border-radius: 8px; border: 2px solid {border_c}; text-align: center;">
                    <p style="color: #CBD5E1; font-size: 13px; margin-bottom: 5px; font-weight: 600; letter-spacing: 0.5px;">XU HƯỚNG GIÁ ĐÓNG CỬA PHIÊN TIẾP THEO</p>
                    <h1 style="color: {text_c} !important; font-size: 46px !important; font-weight: 800; margin: 0; padding: 5px 0;">${prediction:.2f}</h1>
                    <p style="color: {text_c} !important; font-size: 18px; margin: 5px 0 0 0; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                        {status_text}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("Hệ thống đang chờ. Hãy bấm nút 'Thực hiện tính toán dự báo giá' bên trái.")
            st.markdown("---")
            
        st.markdown('<p style="font-weight:600; color:#94A3B8; margin-bottom:12px;">Độ tin cậy thuật toán (Dữ liệu kiểm thử hiện tại):</p>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score", f"{metrics['R2']:.4f}")
        m_col2.metric("MAE (Sai số)", f"{metrics['MAE']:.4f}")
        m_col3.metric("MSE", f"{metrics['MSE']:.4f}")

# --- TAB 2: ĐỒ THỊ LÀM SÁNG ĐƯỜNG LƯỚI GRIDLINE ---
with tab2:
    st.markdown('<h3 style="color:#00E676; margin-bottom:20px; font-size:18px; font-weight:bold;">📊 ĐỒ THỊ PHÂN TÍCH KỸ THUẬT VÀ THUẬT TOÁN</h3>', unsafe_allow_html=True)
    
    # --- ĐỂ KHÔNG BỊ NUỐT ĐƯỜNG CHỈ: Sửa template thành dark và làm sáng gridcolor ---
    chart_layout_config = dict(
        template="plotly_dark",
        paper_bgcolor='rgba(15, 23, 42, 0.6)', 
        plot_bgcolor='#0F172A',
        font=dict(color='#E2E8F0'),
        # Tăng cường độ tương phản đường chỉ bằng cách đổi màu grid thành xám sáng #334155
        xaxis=dict(gridcolor='#334155', linecolor='#475569', title_font=dict(color='#94A3B8')),
        yaxis=dict(gridcolor='#334155', linecolor='#475569', title_font=dict(color='#94A3B8'))
    )

    # Biểu đồ 1: Biến động giá
    fig1 = object_graph.Figure()
    fig1.add_trace(object_graph.Scatter(x=filtered_df['date'], y=filtered_df['close'], name='Giá đóng cửa (Close)', line=dict(color='#00E676', width=2)))
    fig1.add_trace(object_graph.Scatter(x=filtered_df['date'], y=filtered_df['open'], name='Giá mở cửa (Open)', line=dict(color='#EAB308', width=1.5, dash='dot')))
    fig1.update_layout(title=f"Biểu đồ 1: Lịch sử Biến động Giá cổ phiếu {selected_ticker}", hovermode="x unified", **chart_layout_config)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Biểu đồ 2: Đặc trưng quan trọng (Phối màu: Trần xanh lam, tăng xanh lá, sàn tím)
    fig2 = px.bar(
        importance_df, x='Importance', y='Feature', orientation='h',
        title="Biểu đồ 2: Mức độ quan trọng của các đặc trưng đầu vào",
        labels={'Importance': 'Độ quan trọng', 'Feature': 'Đặc trưng'},
        color='Importance', color_continuous_scale=['#9c27b0', '#FF1744', '#EAB308', '#00E676', '#00e5ff']
    )
    fig2.update_layout(coloraxis_showscale=False, **chart_layout_config)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Biểu đồ 3: So sánh Thực tế vs Dự đoán (Đường dự đoán nét đứt rõ nét trên nền lưới sáng)
    fig3 = object_graph.Figure()
    display_length = min(100, len(y_test))
    y_test_plot = y_test.values[-display_length:]
    y_pred_plot = y_pred[-display_length:]
    x_axis = np.arange(len(y_test_plot))
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_test_plot, mode='lines', name='Thực tế', line=dict(color='#00E676', width=2.5)))
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_pred_plot, mode='lines', name='Dự đoán', line=dict(color='#FF1744', width=2, dash='dash')))
    fig3.update_layout(title=f"Biểu đồ 3: So sánh Giá trị Thực tế vs Dự đoán ({display_length} phiên cuối)", **chart_layout_config)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Biểu đồ 4: Khối lượng giao dịch tương quan
    fig4 = px.scatter(
        filtered_df, x='volume', y='close', color='high',
        title="Biểu đồ 4: Mối tương quan giữa Khối lượng giao dịch và Giá đóng cửa",
        labels={'volume': 'Khối lượng khớp', 'close': 'Giá khớp', 'high': 'Giá cao nhất'},
        color_continuous_scale='Bluered'
    )
    fig4.update_layout(**chart_layout_config)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Biểu đồ 5: Biên độ dao động hình cột màu sắc Tím Huế đặc trưng sàn
    filtered_df['Price_Range'] = filtered_df['high'] - filtered_df['low']
    fig5 = px.histogram(
        filtered_df, x='Price_Range', 
        title="Biểu đồ 5: Phân phối Biên độ dao động giá trong ngày (High - Low)",
        labels={'Price_Range': 'Biên độ dao động (USD)'},
        color_discrete_sequence=['#9c27b0'], nbins=50
    )
    fig5.update_layout(yaxis_title="Tần suất xuất hiện", **chart_layout_config)
    st.plotly_chart(fig5, use_container_width=True)

# --- TAB 3: DỮ LIỆU SÁNG RÕ KHÔNG BỊ KHUẤT CHỮ ---
with tab3:
    st.markdown(f'<h3 style="color:#EAB308; font-weight:bold;">📋 10 DÒNG DỮ LIỆU ĐẦU TIÊN CỦA KHOẢNG THỜI GIAN ĐÃ CHỌN ({selected_ticker})</h3>', unsafe_allow_html=True)
    # Hiển thị trực tiếp Dataframe rõ ràng nhờ có tùy biến CSS cục bộ bảng ở phần trên đầu file
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    st.markdown('<h3 style="color:#EAB308; margin-top:30px; font-weight:bold;">📊 Thống kê mô tả tổng quan trong khoảng thời gian này</h3>', unsafe_allow_html=True)
    st.dataframe(filtered_df.describe(), use_container_width=True)
