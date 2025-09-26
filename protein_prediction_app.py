import streamlit as st

# ✅ 确保 set_page_config() 是第一条 Streamlit 语句
st.set_page_config(layout="wide", page_title="蛋白分离预测平台", page_icon="🧪")

import numpy as np
import joblib
import os

# 设定模型路径
MODEL_PATH = os.path.join("0921", "ET_optimized.pkl")


# ✅ 加载模型（只加载一次，提高性能）
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"❌ Model file not found. Please check the path: {MODEL_PATH}")
        st.stop()

adaboost_model = load_model()

# ✅ 设置页面标题
st.title("🧪 Protein Separation Prediction Platform - Based on Extratrees Machine Learning Model")
st.write("""
This platform allows input of experimental parameters for protein separation and provides predictions. 
Enter the parameters on the left and click **"🚀 Predict"** to obtain results based on the model.
""")

# ✅ 侧边栏输入区域
st.sidebar.header("🔧 Input Experimental Parameters")
st.sidebar.write("Please adjust the parameters below (units: mL, h, mM, etc.):")

# ✅ 允许用户调整的实验特征值（基于数据文件的特征范围）
X_1 = st.sidebar.number_input("Volume of KH550 (mL)", value=1.0)
X_2 = st.sidebar.number_input("Temperature (°C)", value=60.0)
X_3 = st.sidebar.number_input("Reaction time (h)", value=6.0)
X_4 = st.sidebar.number_input("Protein incubation time (h)", value=2.0)
X_5 = st.sidebar.number_input("Imidazole concentration (mM)", value=500.0)
X_6 = st.sidebar.number_input("Volume of imidazole (µL)", value=500.0)

# ✅ 预测按钮
predict_button = st.sidebar.button("🚀 Predict")

# ✅ 主页面用于展示预测结果
if predict_button:
    st.header("📈 Prediction Result")
    try:
        # 组装输入特征
        input_array = np.array([X_1, X_2, X_3, X_4, X_5, X_6]).reshape(1, -1)

        # 使用模型进行预测
        prediction = adaboost_model.predict(input_array)[0]

        # ✅ 显示预测结果
        st.success(f"🔍 预测结果：{prediction:.2f}")
    except Exception as e:
        st.error(f"⚠️ 预测时发生错误：{e}")

# ✅ 页脚信息
st.markdown("---")
st.header("📌 Instructions")
st.write("""
- **Adjust experimental parameters** and observe changes in predictions.
- This model is trained using the AdaBoost algorithm and is designed for predicting protein separation experiment outcomes.
- Ensure input parameters are within a reasonable range for accurate predictions.
""")



