import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: #c62828;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.patient-card {
    background-color: #f8f9fa;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
}

.result-yes {
    background-color: #ffebee;
    border: 2px solid #ef5350;
    color: #b71c1c;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.result-no {
    background-color: #e8f5e9;
    border: 2px solid #66bb6a;
    color: #1b5e20;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "heart_disease_ann_model.keras"
    )


# =========================================================
# LOAD PREPROCESSOR
# =========================================================

@st.cache_resource
def load_preprocessor():

    return joblib.load(
        "heart_disease_preprocessor.pkl"
    )


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "heart_disease.csv"
    )


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    model = load_model()
    preprocessor = load_preprocessor()
    df = load_dataset()

except Exception as e:

    st.error("❌ Required model files could not be loaded.")

    st.code(str(e))

    st.stop()


# =========================================================
# TARGET COLUMN
# =========================================================

TARGET_COLUMN = "Heart_ stroke"


if TARGET_COLUMN not in df.columns:

    st.error(
        f"Target column '{TARGET_COLUMN}' was not found in the dataset."
    )

    st.stop()


# =========================================================
# FEATURE COLUMNS
# =========================================================

feature_columns = [
    column
    for column in df.columns
    if column != TARGET_COLUMN
]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Artificial Neural Network Based Prediction System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INPUT
# =========================================================

st.markdown(
    '<div class="section-title">👤 Enter Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="patient-card">',
    unsafe_allow_html=True
)


input_data = {}

columns = st.columns(2)


for index, feature in enumerate(feature_columns):

    with columns[index % 2]:

        # -----------------------------------------------
        # NUMERICAL FEATURES
        # -----------------------------------------------

        if pd.api.types.is_numeric_dtype(df[feature]):

            min_value = float(
                df[feature].min()
            )

            max_value = float(
                df[feature].max()
            )

            median_value = float(
                df[feature].median()
            )

            # Avoid invalid min/max configuration
            if min_value == max_value:

                input_data[feature] = st.number_input(
                    feature,
                    value=median_value
                )

            else:

                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=median_value
                )

        # -----------------------------------------------
        # CATEGORICAL FEATURES
        # -----------------------------------------------

        else:

            options = (
                df[feature]
                .dropna()
                .unique()
                .tolist()
            )

            if len(options) > 0:

                input_data[feature] = st.selectbox(
                    feature,
                    options
                )

            else:

                input_data[feature] = st.text_input(
                    feature
                )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("")

predict_button = st.button(
    "🔮 Predict Heart Disease",
    use_container_width=True,
    type="primary"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -----------------------------------------------
        # CREATE INPUT DATAFRAME
        # -----------------------------------------------

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )


        # -----------------------------------------------
        # PREPROCESS INPUT
        # -----------------------------------------------

        processed_input = preprocessor.transform(
            input_df
        )


        # -----------------------------------------------
        # ANN PREDICTION
        # -----------------------------------------------

        prediction_output = model.predict(
            processed_input,
            verbose=0
        )


        probability = float(
            np.asarray(
                prediction_output
            ).ravel()[0]
        )


        # Make sure probability stays between 0 and 1

        probability = max(
            0.0,
            min(
                1.0,
                probability
            )
        )


        # -----------------------------------------------
        # CLASSIFICATION
        # -----------------------------------------------

        if probability >= 0.5:

            prediction = "YES"

            confidence = probability

            st.markdown(
                f"""
                <div class="result-yes">

                ⚠️ Heart Disease Prediction: YES

                <br><br>

                Prediction Probability:
                {probability:.2%}

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            prediction = "NO"

            confidence = 1 - probability

            st.markdown(
                f"""
                <div class="result-no">

                ✅ Heart Disease Prediction: NO

                <br><br>

                Prediction Probability:
                {confidence:.2%}

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # RESULT DETAILS
        # =================================================

        st.markdown("### 📊 Prediction Details")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Prediction",
                prediction
            )


        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )


        with col3:

            st.metric(
                "ANN Output",
                f"{probability:.4f}"
            )


        # =================================================
        # PROBABILITY CHART
        # =================================================

        st.markdown("### 📈 Prediction Probability")


        no_probability = 1 - probability
        yes_probability = probability


        fig, ax = plt.subplots(
            figsize=(8, 4)
        )


        classes = [
            "No Heart Disease",
            "Heart Disease"
        ]


        probabilities = [
            no_probability,
            yes_probability
        ]


        bars = ax.bar(
            classes,
            probabilities
        )


        ax.set_ylim(
            0,
            1
        )


        ax.set_ylabel(
            "Probability"
        )


        ax.set_title(
            "ANN Prediction Probability"
        )


        for bar, value in zip(
            bars,
            probabilities
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2%}",
                ha="center",
                fontweight="bold"
            )


        plt.tight_layout()


        st.pyplot(
            fig
        )


        # =================================================
        # PATIENT INPUT SUMMARY
        # =================================================

        with st.expander(
            "👤 View Entered Patient Information"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction could not be generated."
        )

        st.code(
            str(e)
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "❤️ Heart Disease Classification using Artificial Neural Network"
)