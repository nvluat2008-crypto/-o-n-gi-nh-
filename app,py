        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name="du_lieu_nha_o.csv",
        mime="text/csv"
    )
    st.info("Dữ liệu trong website là dữ liệu giả lập phục vụ học tập, không đại diện cho giá nhà thực tế.")

elif page == "Dự đoán giá nhà":
    st.markdown("### 🧮 Dự đoán giá một căn nhà")
    st.write("Điều chỉnh các thông số bên dưới, sau đó nhấn **Dự đoán giá nhà**.")
    left, right = st.columns([1, 1.3])
    with left:
        area = st.slider("Diện tích (m²)", 20, 400, 100)
        bedrooms = st.select_slider("Số phòng ngủ", options=[1, 2, 3, 4, 5, 6], value=3)
        bathrooms = st.select_slider("Số phòng tắm", options=[1, 2, 3, 4], value=2)
        predict_clicked = st.button("✨ Dự đoán giá nhà", type="primary", use_container_width=True)
    with right:
        st.markdown("#### Thông tin căn nhà")
        st.metric("Diện tích", f"{area} m²")
        st.metric("Phòng ngủ / Phòng tắm", f"{bedrooms} / {bathrooms}")
        if predict_clicked:
            input_df = pd.DataFrame({
                "Diện tích (m²)": [area],
                "Phòng ngủ": [bedrooms],
                "Phòng tắm": [bathrooms]
            })
            estimate = float(model.predict(input_df)[0])
            st.success("Đã hoàn tất dự đoán")
            st.markdown("#### Giá nhà ước tính")
            st.metric("Giá dự đoán", f"{estimate:,.1f} triệu đồng")
            st.caption(f"Tương đương khoảng {estimate / 1000:,.3f} tỷ đồng.")
            st.caption("Ước tính từ mô hình học trên dữ liệu giả lập, chỉ dùng cho mục đích học tập.")

    st.divider()
    st.markdown("### 📐 Công thức mô hình")
    coefs = pd.DataFrame({"Đặc trưng": features, "Hệ số hồi quy": model.coef_})
    st.dataframe(coefs.round(3), use_container_width=True, hide_index=True)
    st.write(f"Hệ số chặn (intercept): **{model.intercept_:,.2f}** triệu đồng.")

elif page == "Phân cụm K-Means":
    st.markdown("### 🧩 Phân nhóm nhà ở bằng K-Means")
    k = st.slider("Số cụm K", min_value=2, max_value=6, value=3)
    cluster_features = ["Diện tích (m²)", "Phòng ngủ", "Phòng tắm", "Giá nhà (triệu đồng)"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[cluster_features])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clustered = data.copy()
    clustered["Cụm"] = km.fit_predict(scaled).astype(str)

    fig = px.scatter(
        clustered, x="Diện tích (m²)", y="Giá nhà (triệu đồng)",
        color="Cụm", hover_data=["Phòng ngủ", "Phòng tắm"],
        title=f"Phân cụm nhà ở với K = {k}",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(template="plotly_white", height=480)
    st.plotly_chart(fig, use_container_width=True)

    summary = clustered.groupby("Cụm")[cluster_features].mean().round(2)
    summary["Số căn nhà"] = clustered.groupby("Cụm").size()
    st.markdown("#### Đặc trưng trung bình của từng cụm")
    st.dataframe(summary, use_container_width=True)
    st.download_button(
        "⬇️ Tải dữ liệu có nhãn cụm",
        data=clustered.to_csv(index=False).encode("utf-8-sig"),
        file_name="du_lieu_phan_cum_kmeans.csv",
        mime="text/csv"
    )
    st.caption("K-Means là phương pháp phân cụm không giám sát. Nhãn cụm 0, 1, 2... không tự động đồng nghĩa với thấp, trung bình hay cao cấp; cần xem đặc trưng trung bình để diễn giải.")

else:
    st.markdown("### 🧪 Đánh giá mô hình hồi quy tuyến tính")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['MAE']:.2f} trđ")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f} trđ")
    m3.metric("MSE", f"{metrics['MSE']:.2f}")
    m4.metric("R²", f"{metrics['R²']:.3f}")

    st.markdown("#### Giá thực tế và giá dự đoán")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=compare["Giá thực tế"], y=compare["Giá dự đoán"],
        mode="markers", name="Dự đoán",
        marker=dict(size=9, opacity=0.8)
    ))
    low = min(compare.min())
    high = max(compare.max())
    fig.add_trace(go.Scatter(
        x=[low, high], y=[low, high], mode="lines",
        name="Dự đoán hoàn hảo", line=dict(dash="dash", color="#ef4444")
    ))
    fig.update_layout(
        xaxis_title="Giá thực tế (triệu đồng)",
        yaxis_title="Giá dự đoán (triệu đồng)",
        template="plotly_white", height=430
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Bảng so sánh trên tập kiểm tra")
    st.dataframe(compare.round(2), use_container_width=True, hide_index=True)
    st.markdown("#### Nhận xét tự động")
    if metrics["R²"] >= 0.8:
        st.success("Mô hình giải thích khá tốt biến động giá trong bộ dữ liệu giả lập.")
    elif metrics["R²"] >= 0.5:
        st.info("Mô hình có khả năng giải thích ở mức tương đối; có thể cải thiện bằng dữ liệu và đặc trưng tốt hơn.")
    else:
        st.warning("Khả năng giải thích của mô hình còn hạn chế trên tập kiểm tra.")
    st.write(
        "MAE và RMSE càng thấp thường cho thấy dự đoán càng gần giá trị thực tế. "
        "R² càng gần 1 thường cho thấy mô hình giải thích tốt hơn biến động của dữ liệu. "
        "Các kết quả ở đây chỉ phản ánh dữ liệu mô phỏng."
    )

st.divider()
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:.85rem'>"
    "HouseInsight · Bài thực hành học máy · Linear Regression + K-Means"
    "</div>",
    unsafe_allow_html=True
)
