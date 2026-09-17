import streamlit as st
import numpy as np
from PIL import Image
import ocr_utils


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="✍️",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    digit_model = ocr_utils.load_digit_model()
    text_model = ocr_utils.load_crnn_model()

    return digit_model, text_model


try:

    digit_model, text_model = load_models()

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("Handwritten OCR")

st.write(
    "Digit Model: **MNIST TensorFlow/Keras** | "
    "Text Engine: **Trained CRNN Model (`model.h5`)**"
)


# ============================================================
# TABS
# ============================================================

digit_tab, text_tab = st.tabs(
    [
        "🔢 Digit Recognition",
        "📝 Text Recognition"
    ]
)


# ============================================================
# DIGIT RECOGNITION
# ============================================================

with digit_tab:

    st.header("🔢 Digit Recognition")

    st.write(
        "Upload an image containing one handwritten digit "
        "from 0 to 9."
    )

    digit_file = st.file_uploader(
        "📷 Upload handwritten digit",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        key="digit_upload"
    )

    if digit_file is not None:

        try:

            digit_image = Image.open(
                digit_file
            ).convert("L")

            st.image(
                digit_image,
                caption="Uploaded Digit",
                width=250
            )

            recognize_digit = st.button(
                "🔍 Recognize Digit",
                key="recognize_digit_button"
            )

            if recognize_digit:

                image_array = np.array(
                    digit_image
                )

                try:

                    # IMPORTANT:
                    # predict_digit accepts image only.
                    digit_result = ocr_utils.predict_digit(
                        image_array
                    )

                    # Handle dictionary result
                    if isinstance(
                        digit_result,
                        dict
                    ):

                        prediction = digit_result.get(
                            "prediction",
                            ""
                        )

                        confidence = digit_result.get(
                            "confidence",
                            None
                        )

                    else:

                        prediction = digit_result
                        confidence = None


                    st.success(
                        f"### Predicted Digit: **{prediction}**"
                    )


                    if confidence is not None:

                        st.write(
                            f"Confidence: "
                            f"**{confidence * 100:.2f}%**"
                        )

                except Exception as e:

                    st.error(
                        f"Digit recognition error: {e}"
                    )

        except Exception as e:

            st.error(
                f"Could not open digit image: {e}"
            )


# ============================================================
# HANDWRITTEN TEXT RECOGNITION
# ============================================================

with text_tab:

    st.header(
        "📝 Handwritten Text Recognition"
    )

    st.write(
        "Upload an image containing a handwritten "
        "word or short sentence."
    )

    st.info(
        "Text engine: **Trained CRNN Model (`model.h5`)**"
    )


    text_file = st.file_uploader(
        "📷 Upload handwritten text",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        key="text_upload"
    )


    if text_file is not None:

        try:

            text_image = Image.open(
                text_file
            ).convert("RGB")


            # Use width instead of use_container_width
            # for Streamlit compatibility.
            st.image(
                text_image,
                caption="Uploaded Handwritten Text",
                width=700
            )


            recognize_text = st.button(
                "🔍 Recognize Handwritten Text",
                key="recognize_text_button"
            )


            if recognize_text:

                image_array = np.array(
                    text_image
                )


                with st.spinner(
                    "Reading handwritten text..."
                ):

                    try:

                        text_result = (
                            ocr_utils.predict_text(
                                image_array
                            )
                        )


                        # predict_text returns:
                        # {
                        #     "prediction": "...",
                        #     "engine": "..."
                        # }

                        if isinstance(
                            text_result,
                            dict
                        ):

                            recognized_text = (
                                text_result.get(
                                    "prediction",
                                    ""
                                )
                            )

                            engine = (
                                text_result.get(
                                    "engine",
                                    "Trained CRNN Model"
                                )
                            )

                        else:

                            recognized_text = (
                                str(text_result)
                            )

                            engine = (
                                "Trained CRNN Model"
                            )


                        recognized_text = (
                            recognized_text.strip()
                        )


                        if recognized_text:

                            st.success(
                                "### Recognized Text"
                            )

                            st.text_area(
                                "OCR Result",
                                recognized_text,
                                height=150
                            )

                            st.caption(
                                f"Engine: {engine}"
                            )

                        else:

                            st.warning(
                                "No handwritten text "
                                "was recognized."
                            )


                    except Exception as e:

                        st.error(
                            "Text recognition error: "
                            f"{e}"
                        )


        except Exception as e:

            st.error(
                f"Could not open text image: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Handwritten OCR | "
    "MNIST Digit Recognition + "
    "Trained CRNN Handwritten Text Recognition"
)
