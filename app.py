import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HouseInsight | Dự đoán giá nhà", page_icon="🏡", layout="wide")

st.markdown("""
<style>
.stApp {background:#f5f7fb;}
.hero {padding:24px;border-radius:18px;background:linear-gradient(120deg,#0f766e,#2563eb);color:white;margin-bottom:18px}
.hero h1 {color:white;margin:0}
.hero p {color:#eef7ff;margin:8px 0 0}
div[data-testid="stMetric"] {background:white;padding:14px;border-radius:14px;border:1px solid #e6ebf2}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_data(n=100, seed=42, noise=200):
    rng = np.random.default_rng(seed)
    area = rng.integers(40, 201, n)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    price = 500 + area * 20 + bedrooms * 100 + bathrooms * 80 + rng.normal(0, noise, n)
    return pd.DataFrame({
        "Diện tích (m²)": area,
        "Phòng ngủ": bedrooms,
        "Phòng tắm": bathrooms,
        "Giá nhà (triệu đồng)": np.maximum(price, 100)
    })

def train_model(df):
    features = ["Diện tích (m²)", "Phòng ngủ", "Phòng tắm"]
    X = df[features]
    y = df["Giá nhà (triệu đồng)"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R²": r2_score(y_test, predictions),
    }
    comparison = pd.DataFrame({
        "Giá thực tế": y_test.to_numpy(),
        "Giá dự đoán": predictions,
    })
    return model, metrics, comparison, features

st.sidebar.title("🏡 HouseInsight")
st.sidebar.caption("Phân tích và dự đoán giá nhà")
page = st.sidebar.radio("Điều hướng", [
    "Tổng quan", "Dự đoán giá nhà", "Phân cụm K-Means", "Đánh giá mô hình"
])
st.sidebar.divider()
st.sidebar.subheader("Thiết lập dữ liệu mẫu")
n_samples = st.sidebar.slider("Số căn nhà", 50, 500, 100, 25)
noise = st.sidebar.slider("Mức nhiễu dữ liệu", 50, 600, 200, 50)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)

df = create_data(n_samples, int(seed), noise)
model, metrics, comparison, features = train_model(df)

st.markdown("""
<div class="hero">
<h1>🏡 HouseInsight</h1>
<p>Khám phá dữ liệu nhà ở · Dự đoán bằng Linear Regression · Phân cụm bằng K-Means</p>
</div>
""", unsafe_allow_html=True)

if page == "Tổng quan":
    st.subheader("📊 Tổng quan dữ liệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số căn nhà", f"{len(df):,}")
    c2.metric("Giá trung bình", f"{df['Giá nhà (triệu đồng)'].mean():,.0f} triệu")
    c3.metric("Diện tích trung bình", f"{df['Diện tích (m²)'].mean():.1f} m²")
    c4.metric("R² mô hình", f"{metrics['R²']:.3f}")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(df, x="Diện tích (m²)", y="Giá nhà (triệu đồng)",
                         color="Phòng ngủ", size="Phòng tắm",
                         title="Diện tích và giá nhà", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(df, x="Giá nhà (triệu đồng)", nbins=18,
                           title="Phân bố giá nhà", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dữ liệu mẫu")
    st.dataframe(df.head(10).round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Tải dữ liệu CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="du_lieu_nha_o.csv",
        mime="text/csv"
    )
    st.info("Đây là dữ liệu mô phỏng phục vụ học tập, không phải giá nhà thị trường thực tế.")

elif page == "Dự đoán giá nhà":
    st.subheader("🧮 Dự đoán giá một căn nhà")
    left, right = st.columns([1, 1])
    with left:
        area = st.slider("Diện tích (m²)", 20, 400, 100)
        bedrooms = st.select_slider("Số phòng ngủ", options=[1, 2, 3, 4, 5, 6], value=3)
        bathrooms = st.select_slider("Số phòng tắm", options=[1, 2, 3, 4], value=2)
        predict_clicked = st.button("✨ Dự đoán giá nhà", type="primary", use_container_width=True)
    with right:
        st.markdown("#### Thông tin căn nhà")
        st.metric("Diện tích", f"{area} m²")
        st.metric("Phòng ngủ / phòng tắm", f"{bedrooms} / {bathrooms}")
        if predict_clicked:
            new_house = pd.DataFrame({
                "Diện tích (m²)": [area],
                "Phòng ngủ": [bedrooms],
                "Phòng tắm": [bathrooms]
            })
            estimate = float(model.predict(new_house)[0])
            st.success("Đã dự đoán xong")
            st.metric("Giá nhà ước tính", f"{estimate:,.1f} triệu đồng")
            st.caption(f"Tương đương khoảng {estimate / 1000:,.3f} tỷ đồng.")
            st.caption("Kết quả dựa trên dữ liệu mô phỏng, chỉ dùng cho mục đích học tập.")

    st.subheader("Hệ số mô hình")
    st.dataframe(pd.DataFrame({
        "Đặc trưng": features,
        "Hệ số hồi quy": model.coef_
    }).round(3), use_container_width=True, hide_index=True)
    st.write(f"Hệ số chặn: **{model.intercept_:,.2f} triệu đồng**")

elif page == "Phân cụm K-Means":
    st.subheader("🧩 Phân cụm nhà ở")
    k = st.slider("Số cụm K", 2, 6, 3)
    cluster_cols = ["Diện tích (m²)", "Phòng ngủ", "Phòng tắm", "Giá nhà (triệu đồng)"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cluster_cols])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clustered = df.copy()
    clustered["Cụm"] = km.fit_predict(scaled).astype(str)

    fig = px.scatter(clustered, x="Diện tích (m²)", y="Giá nhà (triệu đồng)",
                     color="Cụm", hover_data=["Phòng ngủ", "Phòng tắm"],
                     title=f"Phân cụm nhà ở (K={k})", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    summary = clustered.groupby("Cụm")[cluster_cols].mean().round(2)
    summary["Số căn nhà"] = clustered.groupby("Cụm").size()
    st.dataframe(summary, use_container_width=True)
    st.download_button(
        "⬇️ Tải dữ liệu phân cụm CSV",
        data=clustered.to_csv(index=False).encode("utf-8-sig"),
        file_name="du_lieu_phan_cum.csv",
        mime="text/csv"
    )
    st.caption("Các nhãn cụm là số hiệu nhóm; cần xem đặc trưng trung bình để diễn giải.")

else:
    st.subheader("📈 Đánh giá mô hình hồi quy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{metrics['MAE']:.2f} triệu")
    c2.metric("MSE", f"{metrics['MSE']:.2f}")
    c3.metric("RMSE", f"{metrics['RMSE']:.2f} triệu")
    c4.metric("R²", f"{metrics['R²']:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison["Giá thực tế"], y=comparison["Giá dự đoán"],
        mode="markers", name="Giá dự đoán"
    ))
    low = float(min(comparison["Giá thực tế"].min(), comparison["Giá dự đoán"].min()))
    high = float(max(comparison["Giá thực tế"].max(), comparison["Giá dự đoán"].max()))
    fig.add_trace(go.Scatter(
        x=[low, high], y=[low, high], mode="lines",
        name="Dự đoán hoàn hảo", line=dict(dash="dash")
    ))
    fig.update_layout(
        xaxis_title="Giá thực tế (triệu đồng)",
        yaxis_title="Giá dự đoán (triệu đồng)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

    st.subheader("Nhận xét")
    if metrics["R²"] >= 0.8:
        st.success("Mô hình giải thích tốt biến động giá trong bộ dữ liệu mô phỏng.")
    elif metrics["R²"] >= 0.5:
        st.info("Mô hình có khả năng giải thích tương đối; có thể cải thiện bằng dữ liệu tốt hơn.")
    else:
        st.warning("Khả năng giải thích còn hạn chế trên tập kiểm tra.")
    st.write("MAE/RMSE càng thấp thường càng tốt; R² càng gần 1 thường cho thấy mô hình giải thích dữ liệu tốt hơn.")

st.divider()
st.caption("HouseInsight · Bài thực hành học máy · Dữ liệu mô phỏng")
