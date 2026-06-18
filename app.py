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

# --- CSS TỐI ƯU ĐỘ TƯƠNG PHẢN CHUẨN SSI IBOARD ---
st.markdown("""
    <style>
        /* Toàn bộ app và chữ cơ bản */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
            color: #F1F5F9 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Sidebar layout */
        section[data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #334155;
        }
        section[data-testid="stSidebar"] * {
            color: #CBD5E1 !important;
        }

        /* Các widget nhập liệu */
        div[data-testid="stBlock"] {
            background-color: rgba(15, 23, 42, 0.85) !important;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #334155;
        }
        
        /* Menu Điều hướng Tabs phong cách HOSE / VN30 */
        button[data-baseweb="tab"] {
            color: #94A3B8 !important;
            background-color: #0F172A !important;
            border-radius: 6px 6px 0 0;
            margin-right: 6px;
            padding: 10px 20px !important;
            border: 1px solid #334155 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #1E293B !important;
            color: #22C55E !important; 
            font-weight: bold !important;
            border-bottom: 3px solid #22C55E !important;
        }
        
        /* SỬA LỖI ĐEN QUÁ KHÔNG THẤY DỮ LIỆU: Tối ưu Dataframe */
        .stDataFrame div {
            background-color: #0F172A !important;
        }
        /* Giữ màu chữ trong bảng rõ nét, có phân cấp độ tương phản */
        .stDataFrame data-table, .stDataFrame td, .stDataFrame th {
            color: #F8FAFC !important;
        }
        
        /* Ô input số */
        input {
            color: #FFFFFF !important;
            background-color: #1E293B !important;
            border: 1px solid #475569 !important;
        }
        
        /* Định dạng Metric tùy chỉnh cho màu tăng giảm rực rỡ */
        div[data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 700 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Thanh cấu hình bên trái - Sử dụng màu Vàng tham chiếu và Xanh biển làm điểm nhấn
st.sidebar.markdown('<h2 style="color:#EAB308; font-weight:bold;">🛠️ CẤU HÌNH RANDOM FOREST</h2>', unsafe_allow_html=True)
selected_ticker = st.sidebar.selectbox("Chọn mã cổ phiếu S&P 500", tickers, index=tickers.index('AAL') if 'AAL' in tickers else 0)

ticker_full_data = df[df['Name'] == selected_ticker].sort_values('date')
min_date = ticker_full_data['date'].min().date()
max_date = ticker_full_data['date'].max().date()

st.sidebar.markdown('<h3 style="color:#38BDF8; font-weight:600; margin-top:15px;">📅 LỌC KHOẢNG THỜI GIAN</h3>', unsafe_allow_html=True)
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

# Tiêu đề chính ứng dụng
st.markdown(f'<h1 style="text-align:center; color:#FFFFFF; font-weight:800; margin-bottom:10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.6);">📈 HỆ THỐNG DỰ BÁO GIÁ CỔ PHIẾU S&P 500</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:16px; color:#CBD5E1; margin-bottom:25px;">Phân tích mã cổ phiếu: <strong style="color:#EAB308; background-color:rgba(0,0,0,0.4); padding:2px 6px; border-radius:4px;">{selected_ticker}</strong> | Giai đoạn: <strong style="color:#22C55E;">{start_date}</strong> đến <strong style="color:#22C55E;">{end_date}</strong></p>', unsafe_allow_html=True)

if rf_model is None:
    st.error("⚠️ Không đủ dữ liệu trong khoảng thời gian đã chọn để huấn luyện mô hình. Vui lòng mở rộng khoảng thời gian ở thanh bên (Sidebar).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Dự báo & Kết quả", "📊 Trực quan hóa dữ liệu", "📋 Dữ liệu mẫu & Thống kê"])

# --- TAB 1: DỰ BÁO & KẾT QUẢ ---
with tab1:
    st.markdown("""
        <style>
            .stApp {
                background-image: linear-gradient(rgba(15, 23, 42, 0.75), rgba(15, 23, 42, 0.75)), 
                                  url('https://images.pexels.com/photos/30915372/pexels-photo-30915372.jpeg');
                background-size: cover; background-position: center; background-attachment: fixed;
            }
        </style>
    """, unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown('<div style="background-color:#1E293B; padding:15px; border-radius:6px; border-left: 5px solid #38BDF8;">'
                    '<h3 style="margin-top:0; color:#38BDF8; font-size:18px; font-weight:bold;">📥 NHẬP DỮ LIỆU ĐẦU VÀO (PHIÊN T-1)</h3>'
                    '</div><br>', unsafe_allow_html=True)
        
        latest_row = filtered_df.iloc[-1]
        open_val = st.number_input("Giá mở cửa phiên trước (open)", value=float(latest_row['open']), format="%.2f")
        high_val = st.number_input("Giá cao nhất phiên trước (high)", value=float(latest_row['high']), format="%.2f")
        low_val = st.number_input("Giá thấp nhất phiên trước (low)", value=float(latest_row['low']), format="%.2f")
        close_val = st.number_input("Giá đóng cửa phiên trước (close)", value=float(latest_row['close']), format="%.2f")
        volume_val = st.number_input("Khối lượng giao dịch phiên trước (volume)", value=int(latest_row['volume']), step=1000)
        
        predict_clicked = st.button("🚀 Bắt đầu dự đoán giá", use_container_width=True)
        
    with col_output:
        st.markdown('<div style="background-color:#1E293B; padding:15px; border-radius:6px; border-left: 5px solid #A855F7;">'
                    '<h3 style="margin-top:0; color:#A855F7; font-size:18px; font-weight:bold;">📤 KẾT QUẢ DỰ BÁO CHỈ SỐ VÀ ĐÁNH GIÁ</h3>'
                    '</div><br>', unsafe_allow_html=True)
        
        if predict_clicked:
            input_features = np.array([[open_val, high_val, low_val, close_val, volume_val]])
            prediction = rf_model.predict(input_features)[0]
            diff = prediction - close_val
            
            # BIẾN ĐỔI MÀU SẮC DỰ BÁO LINH HOẠT THEO THỊ TRƯỜNG TĂNG/GIẢM
            if diff > 0:
                metric_color = "#22C55E" # Xanh lá rực khi tăng
                arrow = "▲"
            elif diff < 0:
                metric_color = "#EF4444" # Đỏ rực khi giảm
                arrow = "▼"
            else:
                metric_color = "#EAB308" # Vàng khi đứng giá
                arrow = "■"
                
            st.markdown(f"""
                <div style="background-color: #0F172A; padding: 25px; border-radius: 8px; border: 1px solid #334155; text-align: center;">
                    <p style="color: #94A3B8; font-size: 14px; margin-bottom: 5px; font-weight: 600;">GIÁ ĐÓNG CỬA DỰ BÁO TIẾP THEO ({selected_ticker})</p>
                    <h1 style="color: {metric_color} !important; font-size: 45px !important; font-weight: 800; margin: 0;">${prediction:.2f}</h1>
                    <p style="color: {metric_color} !important; font-size: 16px; margin-top: 8px; font-weight: 700;">
                        {arrow} {diff:+.2f} ({ (diff/close_val)*100 :+.2f}%) so với phiên trước
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("Vui lòng nhấn nút 'Bắt đầu dự đoán giá' ở cột bên trái để xem kết quả dự báo.")
            st.markdown("---")
            
        st.markdown('<p style="font-weight:600; color:#CBD5E1; margin-bottom:12px;">Độ chính xác thuật toán trong khoảng thời gian này:</p>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score (Độ khớp)", f"{metrics['R2']:.4f}")
        m_col2.metric("MAE (Sai số tuyệt đối)", f"{metrics['MAE']:.4f}")
        m_col3.metric("MSE (Sai số bình phương)", f"{metrics['MSE']:.4f}")

# --- TAB 2: ĐỒ THỊ TRỰC QUAN HÓA (ĐÃ LÀM SÁNG LƯỚI KHÔNG BỊ NUỐT ĐƯỜNG CHỈ) ---
with tab2:
    st.markdown("""
        <style>
            .stApp {
                background-image: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.8)), 
                                  url('https://images.pexels.com/photos/6772076/pexels-photo-6772076.jpeg');
                background-size: cover; background-position: center; background-attachment: fixed;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h3 style="color:#22C55E; margin-bottom:20px; font-size:20px; font-weight:bold;">📊 BIỂU ĐỒ THEO HỆ MÀU BẢNG GIÁ CHỨNG KHOÁN mở rộng</h3>', unsafe_allow_html=True)
    
    # --- LÀM SÁNG NỀN VÀ ĐƯỜNG LƯỚI (GRIDLINES) ĐỂ KHÔNG BỊ CHE KHUẤT ĐƯỜNG CHỈ ---
    iboard_chart_theme = dict(
        paper_bgcolor='#1E293B',      # Đổi từ màu đen đặc sang màu xám Slate đậm, tăng độ tương phản rõ rệt
        plot_bgcolor='#0F172A',       # Nền vẽ đồ thị tối vừa phải
        font=dict(color='#CBD5E1', size=11),      
        title_font=dict(color='#FFFFFF', size=14, family='Segoe UI'),
        hoverlabel=dict(bgcolor='#1E293B', font_size=12, font_color='#FFFFFF'),
        # Nâng gridcolor lên màu sáng hơn (#334155) giúp nhìn rõ tọa độ và các đường chỉ nét đứt
        xaxis=dict(gridcolor='#334155', title_font=dict(color='#94A3B8'), tickfont=dict(color='#CBD5E1'), linecolor='#475569'),
        yaxis=dict(gridcolor='#334155', title_font=dict(color='#94A3B8'), tickfont=dict(color='#CBD5E1'), linecolor='#475569')
    )

    # Biểu đồ 1: Đường biến động giá phối hợp màu Xanh lá (Tăng) & Vàng (Tham chiếu)
    fig1 = object_graph.Figure()
    fig1.add_trace(object_graph.Scatter(x=filtered_df['date'], y=filtered_df['close'], name='Giá đóng cửa (Close)', line=dict(color='#22C55E', width=2.5)))
    fig1.add_trace(object_graph.Scatter(x=filtered_df['date'], y=filtered_df['open'], name='Giá mở cửa (Open)', line=dict(color='#EAB308', width=1.5, dash='dot'))) 
    fig1.update_layout(title=f"Biểu đồ 1: Lịch sử Biến động Giá cổ phiếu {selected_ticker}", hovermode="x unified", **iboard_chart_theme)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Biểu đồ 2: Cột độ quan trọng đặc trưng - Kết hợp hệ 5 màu: Tím (Sàn) -> Đỏ -> Vàng -> Xanh lá -> Xanh dương (Trần)
    fig2 = px.bar(
        importance_df, x='Importance', y='Feature', orientation='h',
        title="Biểu đồ 2: Mức độ quan trọng của các đặc trưng đầu vào",
        labels={'Importance': 'Độ quan trọng', 'Feature': 'Đặc trưng'},
        color='Importance', color_continuous_scale=['#A855F7', '#EF4444', '#EAB308', '#22C55E', '#38BDF8'] 
    )
    fig2.update_layout(coloraxis_showscale=False, **iboard_chart_theme)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Biểu đồ 3: So sánh Thực tế (Xanh lá tăng điểm) vs Dự đoán (Đỏ giảm điểm) - Đường nét đứt rõ ràng trên nền mới
    fig3 = object_graph.Figure()
    display_length = min(100, len(y_test))
    y_test_plot = y_test.values[-display_length:]
    y_pred_plot = y_pred[-display_length:]
    x_axis = np.arange(len(y_test_plot))
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_test_plot, mode='lines', name='Thực tế (Xanh tăng)', line=dict(color='#22C55E', width=2.5))) 
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_pred_plot, mode='lines', name='Dự đoán (Đỏ giảm)', line=dict(color='#EF4444', width=2, dash='dash'))) 
    fig3.update_layout(title=f"Biểu đồ 3: So sánh Giá trị Thực tế vs Dự đoán ({display_length} phiên cuối)", **iboard_chart_theme)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Biểu đồ 4: Khối lượng giao dịch - Sử dụng phổ màu Xanh dương (Trần) và Tím (Sàn) tạo điểm nhấn hài hòa
    fig4 = px.scatter(
        filtered_df, x='volume', y='close', color='high',
        title="Biểu đồ 4: Mối tương quan giữa Khối lượng giao dịch và Giá đóng cửa",
        labels={'volume': 'Khối lượng Khớp', 'close': 'Giá khớp', 'high': 'Giá cao nhất'},
        color_continuous_scale=['#A855F7', '#38BDF8']
    )
    fig4.update_layout(**iboard_chart_theme)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Biểu đồ 5: Biên độ dao động sử dụng dải màu Tím sàn đặc trưng iBoard
    filtered_df['Price_Range'] = filtered_df['high'] - filtered_df['low']
    fig5 = px.histogram(
        filtered_df, x='Price_Range', 
        title="Biểu đồ 5: Phân phối Biên độ dao động giá trong ngày (High - Low)",
        labels={'Price_Range': 'Biên độ dao động (USD)'},
        color_discrete_sequence=['#A855F7'], nbins=50 
    )
    fig5.update_layout(yaxis_title="Tần suất xuất hiện", **iboard_chart_theme)
    st.plotly_chart(fig5, use_container_width=True)

# --- TAB 3: DỮ LIỆU MẪU ĐÃ ĐƯỢC XỬ LÝ ĐỘ TƯƠNG PHẢN (ĐỌC RÕ 100%) ---
with tab3:
    st.markdown("""
        <style>
            .stApp {
                background-image: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), 
                                  url('https://cdn.vietnambiz.vn/2019/11/21/947c8cad98ec71b228fd-15743030298821491913597.jpg');
                background-size: cover; background-position: center; background-attachment: fixed;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<h3 style="color:#EAB308; font-weight:bold;">📋 10 DÒNG DỮ LIỆU ĐẦU TIÊN CỦA KHOẢNG THỜI GIAN ĐÃ CHỌN ({selected_ticker})</h3>', unsafe_allow_html=True)
    # Trình bày dữ liệu thô dạng DataFrame gốc với nền text rõ nét không bị CSS cũ ép màu đè mất chữ
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    st.markdown('<h3 style="color:#38BDF8; margin-top:30px; font-weight:bold;">📊 Thống kê mô tả tổng quan trong khoảng thời gian này</h3>', unsafe_allow_html=True)
    st.dataframe(filtered_df.describe(), use_container_width=True)
