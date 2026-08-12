# ================================================================
# TEXT EMOTION DETECTION
# Multi-Kernel CNN - Streamlit Application
# ================================================================

import os
import re
import pickle
import html

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Text Emotion Detection",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================================================
# APPLICATION CONFIGURATION
# ================================================================

MODEL_DIR = "final_model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "multi_kernel_cnn_final.keras"
)

TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

EMOTION_LABELS_PATH = os.path.join(
    MODEL_DIR,
    "emotion_labels.pkl"
)

MAX_SEQUENCE_LENGTH = 40

EMOTIONS = [
    "anger",
    "fear",
    "joy",
    "love",
    "sadness",
    "surprise"
]


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .prediction-label {
        font-size: 30px;
        font-weight: 700;
        text-transform: capitalize;
    }

    .confidence-label {
        font-size: 18px;
        margin-top: 5px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 14px;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================================================================
# TEXT PREPROCESSING
# ================================================================

def preprocess_text(text):
    """
    Apply the same basic text-cleaning approach used
    during model development.

    The model was trained using cleaned/lemmatized text,
    so the inference pipeline performs equivalent cleaning.
    """

    # Convert to string
    text = str(text)

    # Convert HTML entities
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Keep alphabetic characters and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ================================================================
# LOAD MODEL AND ARTIFACTS
# ================================================================

@st.cache_resource(show_spinner=False)
def load_model_artifacts():

    # Check required files
    required_files = [
        MODEL_PATH,
        TOKENIZER_PATH,
        LABEL_ENCODER_PATH,
        EMOTION_LABELS_PATH
    ]

    missing_files = [
        file
        for file in required_files
        if not os.path.exists(file)
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following required model files are missing:\n\n"
            + "\n".join(missing_files)
        )

    # Load trained model
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # Load tokenizer
    with open(
        TOKENIZER_PATH,
        "rb"
    ) as file:
        tokenizer = pickle.load(file)

    # Load label encoder
    with open(
        LABEL_ENCODER_PATH,
        "rb"
    ) as file:
        label_encoder = pickle.load(file)

    # Load emotion labels
    with open(
        EMOTION_LABELS_PATH,
        "rb"
    ) as file:
        emotion_labels = pickle.load(file)

    return (
        model,
        tokenizer,
        label_encoder,
        emotion_labels
    )


# ================================================================
# PREDICTION FUNCTION
# ================================================================

def predict_emotion(
    text,
    model,
    tokenizer,
    label_encoder,
    emotion_labels
):

    # Clean text
    cleaned_text = preprocess_text(text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    # Model prediction
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Predicted class index
    predicted_index = int(
        np.argmax(probabilities)
    )

    # Decode label
    try:
        predicted_label = label_encoder.inverse_transform(
            [predicted_index]
        )[0]
    except Exception:
        predicted_label = emotion_labels[predicted_index]

    # Confidence
    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_label,
        confidence,
        probabilities,
        cleaned_text
    )


# ================================================================
# LOAD ARTIFACTS
# ================================================================

try:

    (
        model,
        tokenizer,
        label_encoder,
        emotion_labels
    ) = load_model_artifacts()

    model_loaded = True

except Exception as error:

    model_loaded = False
    model = None
    tokenizer = None
    label_encoder = None
    emotion_labels = None

    st.error(
        "Unable to load the trained model and supporting files."
    )

    with st.expander(
        "Technical details"
    ):
        st.code(
            str(error)
        )

    st.stop()


# ================================================================
# HEADER
# ================================================================

st.markdown(
    '<div class="main-title">💬 Text Emotion Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Detect the emotion expressed in a piece of text using a
    trained Multi-Kernel Convolutional Neural Network.
    </div>
    """,
    unsafe_allow_html=True
)


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.markdown(
        """
        **Model:** Multi-Kernel CNN

        **Embedding Dimension:** 100

        **Convolution Filters:** 64

        **Kernel Sizes:** 2, 3, 4

        **Dense Units:** 128

        **Dropout:** 0.30

        **Maximum Sequence Length:** 40

        **Number of Classes:** 6
        """
    )

    st.divider()

    st.subheader("📊 Final Test Performance")

    st.metric(
        "Test Accuracy",
        "90.15%"
    )

    st.metric(
        "Weighted F1-Score",
        "90.45%"
    )

    st.divider()

    st.subheader("🎯 Supported Emotions")

    for emotion in EMOTIONS:
        st.write(
            f"• {emotion.capitalize()}"
        )

    st.divider()

    st.caption(
        "Developed as part of an NLP Deep Learning "
        "Emotion Detection project."
    )


# ================================================================
# MAIN INPUT SECTION
# ================================================================

st.markdown(
    '<div class="section-title">Enter Text</div>',
    unsafe_allow_html=True
)

default_text = (
    "I am extremely happy and excited about the wonderful news."
)

user_text = st.text_area(
    "Enter a sentence or paragraph:",
    value="",
    height=150,
    placeholder=(
        "Example: I am feeling really happy today "
        "because everything went well."
    ),
    label_visibility="collapsed"
)


# ================================================================
# EXAMPLE TEXTS
# ================================================================

st.markdown("**Try an example:**")

example_columns = st.columns(3)

examples = [
    (
        "😊 Joy",
        "I am extremely happy and excited about the wonderful news."
    ),
    (
        "😢 Sadness",
        "I feel completely heartbroken and lonely today."
    ),
    (
        "😡 Anger",
        "I am really angry about what happened."
    ),
    (
        "😨 Fear",
        "I am scared and worried about what might happen next."
    ),
    (
        "❤️ Love",
        "I feel deeply connected to the person I love."
    ),
    (
        "😮 Surprise",
        "Wow, I cannot believe this happened!"
    )
]

for index, (
    button_label,
    example_text
) in enumerate(examples):

    column = example_columns[
        index % 3
    ]

    with column:

        if st.button(
            button_label,
            key=f"example_{index}",
            use_container_width=True
        ):
            st.session_state["example_text"] = example_text


# Apply selected example
if "example_text" in st.session_state:

    user_text = st.session_state[
        "example_text"
    ]

    st.text_area(
        "Selected example:",
        value=user_text,
        height=100,
        key="selected_example",
        disabled=True
    )


# ================================================================
# PREDICTION BUTTONS
# ================================================================

col1, col2 = st.columns(
    [1, 1]
)

with col1:

    predict_button = st.button(
        "🔍 Predict Emotion",
        type="primary",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


if clear_button:

    st.session_state.pop(
        "example_text",
        None
    )

    st.rerun()


# ================================================================
# PREDICTION
# ================================================================

if predict_button:

    # Use example text if selected
    prediction_text = st.session_state.get(
        "example_text",
        user_text
    )

    if not prediction_text or not prediction_text.strip():

        st.warning(
            "Please enter some text before clicking "
            "'Predict Emotion'."
        )

    else:

        with st.spinner(
            "Analyzing the text..."
        ):

            try:

                (
                    predicted_label,
                    confidence,
                    probabilities,
                    cleaned_text
                ) = predict_emotion(
                    prediction_text,
                    model,
                    tokenizer,
                    label_encoder,
                    emotion_labels
                )

            except Exception as error:

                st.error(
                    "An error occurred while making the prediction."
                )

                with st.expander(
                    "Technical details"
                ):
                    st.code(
                        str(error)
                    )

                st.stop()

        # ========================================================
        # PREDICTION RESULT
        # ========================================================

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(
            [1, 1]
        )

        with result_col1:

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="font-size:16px;">
                        Predicted Emotion
                    </div>
                    <div class="prediction-label">
                        {predicted_label}
                    </div>
                    <div class="confidence-label">
                        Confidence: <strong>{confidence:.2%}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

            st.progress(
                confidence
            )

        # ========================================================
        # PROBABILITY DISTRIBUTION
        # ========================================================

        st.markdown(
            '<div class="section-title">'
            'Emotion Probability Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame(
            {
                "Emotion": [
                    str(label).capitalize()
                    for label in emotion_labels
                ],
                "Probability": probabilities
            }
        )

        probability_df[
            "Probability"
        ] = probability_df[
            "Probability"
        ].astype(float)

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        ).reset_index(
            drop=True
        )

        display_df = probability_df.copy()

        display_df[
            "Probability"
        ] = display_df[
            "Probability"
        ].map(
            lambda value: f"{value:.2%}"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Bar chart
        chart_df = probability_df.set_index(
            "Emotion"
        )

        st.bar_chart(
            chart_df[
                "Probability"
            ],
            use_container_width=True
        )

        # ========================================================
        # PREPROCESSING INFORMATION
        # ========================================================

        with st.expander(
            "🔎 View Text Processing Details"
        ):

            st.write(
                "**Original Text:**"
            )

            st.write(
                prediction_text
            )

            st.write(
                "**Cleaned Text:**"
            )

            st.write(
                cleaned_text
            )

            st.write(
                f"**Token Sequence Length:** "
                f"{len(tokenizer.texts_to_sequences([cleaned_text])[0])}"
            )

        # ========================================================
        # INTERPRETATION
        # ========================================================

        st.markdown(
            '<div class="section-title">Interpretation</div>',
            unsafe_allow_html=True
        )

        second_highest = (
            probability_df.iloc[1]
        )

        st.info(
            f"The model predicts **"
            f"{predicted_label}** with a confidence of "
            f"**{confidence:.2%}**. "
            f"The next most likely emotion is "
            f"**{second_highest['Emotion']}** "
            f"with a probability of "
            f"**{second_highest['Probability']:.2%}**."
        )


# ================================================================
# PROJECT INFORMATION
# ================================================================

st.divider()

st.markdown(
    '<div class="section-title">📘 About the Project</div>',
    unsafe_allow_html=True
)

about_col1, about_col2 = st.columns(
    [2, 1]
)

with about_col1:

    st.markdown(
        """
        This application uses a **Multi-Kernel Convolutional Neural
        Network (CNN)** to classify text into one of six emotions:

        **Anger, Fear, Joy, Love, Sadness, and Surprise.**

        The model uses a learned word embedding followed by multiple
        1D convolutional layers with kernel sizes **2, 3, and 4**.
        These different kernels allow the model to capture patterns
        of different lengths within the text.

        The extracted features are combined and passed through a
        dense layer before producing the final emotion prediction.
        """
    )

with about_col2:

    st.markdown(
        """
        **Final Model**

        • Multi-Kernel CNN  
        • 64 filters  
        • Kernel sizes: 2, 3, 4  
        • Dense layer: 128 units  
        • Dropout: 0.30  
        • Sequence length: 40  
        • Six emotion classes  

        **Final Test Accuracy**

        ### 90.15%

        **Weighted F1-Score**

        ### 90.45%
        """
    )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "Text Emotion Detection | Multi-Kernel CNN | "
    "Deep Learning NLP Project"
)