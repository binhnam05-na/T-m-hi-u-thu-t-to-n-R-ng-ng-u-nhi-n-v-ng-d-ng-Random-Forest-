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
def train_rf_model(df, ticker, n_estimators, max_depth, min_samples_split):
    stock_df = df[df['Name'] == ticker].copy()
    if len(stock_df) < 10:
        return None, None, None, None, None
    stock_df['open_lag1'] = stock_df['open'].shift(1)
    stock_df['high_lag1'] = stock_df['high'].shift(1)
    stock_df['low_lag1'] = stock_df['low'].shift(1)
    stock_df['close_lag1'] = stock_df['close'].shift(1)
    stock_df['volume_lag1'] = stock_df['volume'].shift(1)
    stock_df = stock_df.dropna()
    
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
    
    return rf, metrics, importance_df, y_test, y_pred

try:
    df = load_and_preprocess_data('all_stocks_5yr.csv')
    tickers = sorted(df['Name'].unique().tolist())
except Exception:
    st.error("Không tìm thấy tệp 'all_stocks_5yr.csv'. Vui lòng đặt tệp dữ liệu cùng cấp với mã nguồn ứng dụng.")
    st.stop()

st.sidebar.markdown('<h2 style="color:#1E3A8A; font-weight:bold;">🛠️ CẤU HÌNH RANDOM FOREST</h2>', unsafe_allow_html=True)
selected_ticker = st.sidebar.selectbox("Chọn mã cổ phiếu S&P 500", tickers, index=tickers.index('AAL') if 'AAL' in tickers else 0)

n_estimators = st.sidebar.slider("Số lượng cây quyết định (n_estimators)", min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider("Độ sâu tối đa của cây (max_depth)", min_value=3, max_value=30, value=15, step=1)
min_samples_split = st.sidebar.slider("Mẫu tối thiểu tách nút (min_samples_split)", min_value=2, max_value=10, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown('<h3 style="color:#4B5563; font-weight:600;">Thông số Mô hình gốc</h3>', unsafe_allow_html=True)
st.sidebar.info("Mô hình sử dụng cơ chế Trễ 1 Phiên (Lag-1) trên các đặc trưng Open, High, Low, Close, Volume để thực hiện dự báo mức giá Close hiện tại.")

rf_model, metrics, importance_df, y_test, y_pred = train_rf_model(df, selected_ticker, n_estimators, max_depth, min_samples_split)

st.markdown(f'<h1 style="text-align:center; color:#0F172A; font-weight:800; margin-bottom:20px;">📈 HỆ THỐNG DỰ BÁO GIÁ CỔ PHIẾU S&P 500</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:18px; color:#475569;">Phân tích và dự báo mã cổ phiếu: <strong style="color:#2563EB;">{selected_ticker}</strong> bằng mô hình học máy Học máy Random Forest Regression</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Dự báo & Kết quả", "📊 Trực quan hóa dữ liệu", "📋 Dữ liệu mẫu (Top 10)"])

with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown('<div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border-left: 5px solid #2563EB;">'
                    '<h3 style="margin-top:0; color:#1E293B;">📥 NHẬP DỮ LIỆU ĐẦU VÀO (PHIÊN T-1)</h3>'
                    '</div>', unsafe_allow_html=True)
        
        stock_data = df[df['Name'] == selected_ticker].sort_values('date')
        latest_row = stock_data.iloc[-1]
        
        open_val = st.number_input("Giá mở cửa phiên trước (open)", value=float(latest_row['open']), format="%.2f")
        high_val = st.number_input("Giá cao nhất phiên trước (high)", value=float(latest_row['high']), format="%.2f")
        low_val = st.number_input("Giá thấp nhất phiên trước (low)", value=float(latest_row['low']), format="%.2f")
        close_val = st.number_input("Giá đóng cửa phiên trước (close)", value=float(latest_row['close']), format="%.2f")
        volume_val = st.number_input("Khối lượng giao dịch phiên trước (volume)", value=int(latest_row['volume']), step=1000)
        
        predict_clicked = st.button("🚀 Bắt đầu dự đoán giá", use_container_width=True)
        
    with col_output:
        st.markdown('<div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border-left: 5px solid #10B981;">'
                    '<h3 style="margin-top:0; color:#1E293B;">📤 THÔNG SỐ ĐẦU RA & ĐÁNH GIÁ MÔ HÌNH</h3>'
                    '</div>', unsafe_allow_html=True)
        
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
            
        st.markdown('<p style="font-weight:600; color:#475569; margin-bottom:5px;">Chỉ số đánh giá độ chính xác (Trên tập kiểm thử):</p>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score (Độ chuẩn xác)", f"{metrics['R2']:.4f}")
        m_col2.metric("MAE (Sai số tuyệt đối)", f"{metrics['MAE']:.4f}")
        m_col3.metric("MSE (Sai số bình phương)", f"{metrics['MSE']:.4f}")

with tab2:
    st.markdown('<h3 style="color:#1E293B; margin-bottom:20px;">📉 HỆ THỐNG 5 BIỂU ĐỒ TRỰC QUAN HÓA CAO CẤP</h3>', unsafe_allow_html=True)
    
    ticker_df = df[df['Name'] == selected_ticker].sort_values('date').copy()
    
    fig1 = object_graph.Figure()
    fig1.add_trace(object_graph.Scatter(x=ticker_df['date'], y=ticker_df['close'], name='Giá đóng cửa (Close)', line=dict(color='#2563EB', width=2)))
    fig1.add_trace(object_graph.Scatter(x=ticker_df['date'], y=ticker_df['open'], name='Giá mở cửa (Open)', line=dict(color='#94A3B8', width=1.5, dash='dash')))
    fig1.update_layout(
        title=f"Biểu đồ 1: Lịch sử Biến động Giá cổ phiếu {selected_ticker} (Line Style)",
        xaxis_title="Thời gian", yaxis_title="Mức giá (USD)",
        template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.bar(
        importance_df, x='Importance', y='Feature', orientation='h',
        title="Biểu đồ 2: Mức độ quan trọng của các đặc trưng đầu vào (Bar Style)",
        labels={'Importance': 'Độ quan trọng', 'Feature': 'Đặc trưng'},
        color='Importance', color_continuous_scale='Viridis'
    )
    fig2.update_layout(template='ggplot2', coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    fig3 = object_graph.Figure()
    y_test_plot = y_test.values[-100:]
    y_pred_plot = y_pred[-100:]
    x_axis = np.arange(len(y_test_plot))
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_test_plot, mode='lines+markers', name='Giá trị thực tế', line=dict(color='#059669', width=2)))
    fig3.add_trace(object_graph.Scatter(x=x_axis, y=y_pred_plot, mode='lines+markers', name='Giá trị dự đoán', line=dict(color='#DC2626', width=2, dash='dot')))
    fig3.update_layout(
        title="Biểu đồ 3: So sánh Giá trị Thực tế vs Dự đoán (100 phiên cuối cùng - Mixed Style)",
        xaxis_title="Các phiên kiểm thử cuối", yaxis_title="Giá cổ phiếu (USD)",
        template="seaborn"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    fig4 = px.scatter(
        ticker_df, x='volume', y='close', color='high',
        title="Biểu đồ 4: Mối tương quan giữa Khối lượng giao dịch và Giá đóng cửa (Scatter Style)",
        labels={'volume': 'Khối lượng giao dịch', 'close': 'Giá đóng cửa', 'high': 'Giá cao nhất'},
        color_continuous_scale='Plasma'
    )
    fig4.update_layout(template='plotly_dark')
    st.plotly_chart(fig4, use_container_width=True)
    
    ticker_df['Price_Range'] = ticker_df['high'] - ticker_df['low']
    fig5 = px.histogram(
        ticker_df, x='Price_Range', 
        title="Biểu đồ 5: Phân phối Biên độ dao động giá trong ngày (High - Low) (Histogram Style)",
        labels={'Price_Range': 'Biên độ dao động (USD)'},
        color_discrete_sequence=['#8B5CF6'], nbins=50
    )
    fig5.update_layout(template='simple_white', yaxis_title="Tần suất xuất hiện")
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.markdown(f'<h3 style="color:#1E293B;">📋 10 DÒNG DỮ LIỆU ĐẦU TIÊN CỦA TỆP CSV ({selected_ticker})</h3>', unsafe_allow_html=True)
    st.dataframe(df[df['Name'] == selected_ticker].head(10), use_container_width=True)
    
    st.markdown('<h3 style="color:#1E293B; margin-top:30px;">📊 Thống kê mô tả tổng quan của mã cổ phiếu</h3>', unsafe_allow_html=True)
    st.dataframe(df[df['Name'] == selected_ticker].describe(), use_container_width=True)
