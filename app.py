import streamlit as st
import joblib
import numpy as np
import pandas as pd

# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="ML_Iris Dataset Dashboard",
    page_icon="🌸",
    layout="wide"
)

# CUSTOM DESIGN
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #808080;
}

[data-testid="stSidebar"] {
    background-color: #14213d;
}

[data-testid="stSidebar"] * {
    color: yellow !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button {
    width: 100% !important;
    justify-content: flex-start !important;
    color: #172554 !important;
    text-align: left !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button p {
    color: #172554 !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button span {
    color: #172554 !important;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #14213d;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

.result-card {
    background-color: #e8f5ee;
    padding: 22px;
    border-radius: 12px;
    border-left: 5px solid #16a34a;
    color: #14532d;
}
</style>
""", unsafe_allow_html=True)

# LOAD TRAINED MODEL
# -------------------------------
model = joblib.load("iris_model.pkl")
encoder = joblib.load("iris_encoder.pkl")

# SIDEBAR NAVIGATION
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

st.sidebar.title("Machine Learning with Iris Dataset")
st.sidebar.caption("Iris Flower Classification Project")

st.sidebar.divider()

if st.sidebar.button("🏠 Dashboard", use_container_width=True):
    st.session_state.page = "Dashboard"

if st.sidebar.button("ℹ️ About the Model", use_container_width=True):
    st.session_state.page = "About the Model"

st.sidebar.divider()
st.sidebar.caption("Built with Python and Streamlit")


# MAIN DASHBOARD
# -------------------------------
if st.session_state.page == "Dashboard":

    st.markdown(
        '<div class="main-title">Iris Flower Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p style="color:black">Use machine learning to identify Iris flower species from their physical measurements.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # TOP METRICS
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Flower Species", "3")
    col2.metric("Model Accuracy", "93.33%")
    col3.metric("Test Samples", "30")

    st.divider()

    # INPUT SECTION
    # -------------------------------
    st.subheader("🌿 Enter Flower Measurements")

    st.write(
        "Provide the four measurements in centimetres "
        "to identify the flower species."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.number_input(
                "Sepal Length (cm)",
                min_value=0.0,
                value=5.1,
                step=0.1
            )

            sepal_width = st.number_input(
                "Sepal Width (cm)",
                min_value=0.0,
                value=3.5,
                step=0.1
            )

        with col2:
            petal_length = st.number_input(
                "Petal Length (cm)",
                min_value=0.0,
                value=1.4,
                step=0.1
            )

            petal_width = st.number_input(
                "Petal Width (cm)",
                min_value=0.0,
                value=0.2,
                step=0.1
            )

        predict_button = st.form_submit_button(
            "🔍 Predict Flower Species",
            use_container_width=True
        )

    # PREDICTION
    # -------------------------------
    if predict_button:

        input_data = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        prediction = model.predict(input_data)

        predicted_species = encoder.inverse_transform(
            prediction
        )[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = max(probabilities) * 100

        st.divider()

        st.subheader("📊 Prediction Results")

        result_col, confidence_col = st.columns(2)

        with result_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>Predicted Flower Species</h3>
                    <h1>{predicted_species.capitalize()}</h1>
                    <p>Classification completed successfully.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with confidence_col:
            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(float(confidence / 100))

        st.write("")

        st.subheader("Species Prediction Probabilities")

        probability_data = pd.DataFrame({
            "Species": [
                name.capitalize() for name in encoder.classes_
            ],
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_data.set_index("Species")
        )

        st.caption(
            "Prediction confidence is the highest class probability "
            "reported by the trained Random Forest model."
        )


# ABOUT THE MODEL
# -------------------------------
else:

    st.title("About the Model")

    st.write(
        "This application uses Machine Learning "
        "to classify Iris flowers into three species."
    )

    st.subheader("Algorithm Used")
    st.write("Random Forest Classifier")

    st.subheader("Dataset Information")
    st.write("Dataset: Iris Flower Dataset")
    st.write("Number of species: 3")
    st.write("Training samples: 119")
    st.write("Testing samples: 30")

    st.subheader("Model Performance")
    st.metric("Test Accuracy", "93.33%")

    st.write(
        "The model uses four input features: "
        "sepal length, sepal width, petal length "
        "and petal width."
    )

    st.write(
        "A Random Forest combines predictions "
        "from multiple decision trees to classify "
        "the flower species."
    )


# FOOTER
# -------------------------------
st.divider()

st.caption(
    "Machine Learning Assignment | Iris Flower Classification "
    "| Built with Python, Scikit-learn and Streamlit"
)