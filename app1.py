import streamlit as st
import pandas as pd
import joblib
import base64

# ---------- BACKGROUND ----------
def add_bg_from_local(image_file):
    import base64

    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;              /* KEY FIX */
            background-position: center;         /* CENTER IMAGE */
            background-repeat: no-repeat;        /* NO REPEAT */
            background-attachment: fixed;        /* FULL SCREEN */
            min-height: 100vh;                   /* FULL HEIGHT */
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.jpg")


# ---------- LOAD MODEL ----------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load('heart_columns.pkl')

# ---------- TITLE ----------
st.markdown("<h1 style='text-align: center;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details below</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------- INPUT LAYOUT ----------
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

# ---------- BUTTON ----------
if st.button("🔍 Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("---")

    st.markdown("## 🩺 Prediction Result")

if prediction == 1:
    st.markdown(
        """
        <div style="
            background-color: rgba(255, 0, 0, 0.15);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid red;
            font-size: 22px;
            font-weight: bold;
            color: red;
        ">
            ⚠️ High Risk of Heart Disease
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="
            background-color: rgba(0, 200, 0, 0.15);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid green;
            font-size: 22px;
            font-weight: bold;
            color: green;
        ">
            ✅ Low Risk of Heart Disease
        </div>
        """,
        unsafe_allow_html=True
    )