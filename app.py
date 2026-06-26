import streamlit as st
import pickle
import numpy as np


st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")


st.markdown("""
<style>
    /* Button ko thoda modern banana */
    div.stButton > button:first-child {
        background-color: #00C853; /* Green color */
        color: white;
        font-weight: bold;
        font-size: 20px;
        padding: 10px 24px;
        border-radius: 10px;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #00E676; /* Hover pe glow effect */
        color: white;
        transform: scale(1.02);
    }

    /* Result Box ka styling */
    .price-box {
        background-color: #1E1E1E;
        border-left: 8px solid #00C853;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        text-align: center;
        margin-top: 20px;
    }
    .price-text {
        color: #00C853;
        font-size: 45px;
        font-weight: 900;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. MODEL LOADING (Fast Load ke liye Cache use kiya hai) ---
@st.cache_resource
def load_models():
    pipe = pickle.load(open("pipe.pkl", "rb"))
    df = pickle.load(open("df.pkl", "rb"))
    return pipe, df


pipe, df = load_models()


st.markdown("<h1 style='text-align: center; color: #1E88E5;'>💻Laptop Price Predictor</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Enter your dream laptop specifications below and let AI estimate the price!</p>",
    unsafe_allow_html=True)
st.divider()




col1, col2, col3 = st.columns(3)
with col1:
    company = st.selectbox('🏢 Brand', df['Company'].unique())
with col2:
    type = st.selectbox('🖥️ Type', df['TypeName'].unique())
with col3:
    os = st.selectbox('⚙️ Operating System', df['Os'].unique())


col4, col5, col6 = st.columns(3)
with col4:
    cpu = st.selectbox('🧠 CPU Brand', df['Cpu Brand'].unique())
with col5:
    ram = st.selectbox('⚡ RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
with col6:
    gpu = st.selectbox('🎮 GPU Brand', df['Gpu Brand'].unique())

st.markdown("### 📏 Physical & Display Specs")
# Row 3
col7, col8, col9 = st.columns(3)
with col7:
    weight = st.number_input('⚖️ Weight of Laptop (in kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
with col8:
    screen_size = st.slider('📐 Screen Size (in inches)', 10.0, 18.0, 15.6)
with col9:
    resolution = st.selectbox('📺 Screen Resolution',
                              ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                               '2560x1440', '2304x1440'])


col10, col11 = st.columns(2)
with col10:
    touchscreen = st.radio('👆 Touchscreen', ['No', 'Yes'], horizontal=True)
with col11:
    ips = st.radio('🎨 IPS Display', ['No', 'Yes'], horizontal=True)

st.markdown("### 💾 Storage Specs")

col12, col13 = st.columns(2)
with col12:
    hdd = st.selectbox('💽 HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
with col13:
    ssd = st.selectbox('🚀 SSD (in GB)', [0, 8, 128, 256, 512, 1024])

st.write("")  # Spacing
st.write("")


if st.button('🎯 Predict Price'):
    with st.spinner("AI is calculating the best price... ⏳"):
        # Processing inputs
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        # Predict
        predicted_price = int(np.exp(pipe.predict(query)[0]))

        # Yahan se st.balloons() hata diya gaya hai!

        # Cool Output Box (Professional Look)
        st.markdown(f"""
        <div class="price-box">
            <h3 style="color: white; margin: 0;">Predicted Price</h3>
            <p class="price-text">₹ {predicted_price:,}</p>
        </div>
        """, unsafe_allow_html=True)
