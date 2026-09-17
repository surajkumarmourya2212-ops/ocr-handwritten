import os
import numpy as np
import cv2
import tensorflow as tf

# MLTU imports for the trained handwritten text model
from mltu.inferenceModel import OnnxInferenceModel
from mltu.tensorflow.model_utils import ctc_decoder
from mltu.utils.text_utils import ctc_decoder as text_ctc_decoder
from mltu.dataProvider import DataProvider
from mltu.preprocessors import ImageResizer
from mltu.transformers import LabelIndexer, LabelPadding

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# DO NOT CHANGE DIGIT MODEL
DIGIT_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "digit_model.keras"
)

# TRAINED HANDWRITTEN TEXT MODEL
CRNN_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "model.h5"
)

_digit_model = None
_crnn_model = None


# --------------------------------------------------
# EXACT VOCABULARY USED DURING TRAINING
# --------------------------------------------------

VOCAB = (
    'sXE-A5!krH+SMh93F?Pg,KYm)R/QzN:ne;VpjxuGO*TyLJl"'
    'vit0WB#w4Zq6aob1D.7d8UCc(&f2\'I'
)


# --------------------------------------------------
# DIGIT MODEL
# --------------------------------------------------

def load_digit_model():
    global _digit_model

    if _digit_model is not None:
        return _digit_model

    if not os.path.exists(DIGIT_MODEL_PATH):
        raise FileNotFoundError(
            f"Digit model not found: {DIGIT_MODEL_PATH}"
        )

    _digit_model = tf.keras.models.load_model(
        DIGIT_MODEL_PATH
    )

    return _digit_model


# --------------------------------------------------
# CRNN MODEL
# --------------------------------------------------

def load_crnn_model():
    global _crnn_model

    if _crnn_model is not None:
        return _crnn_model

    if not os.path.exists(CRNN_MODEL_PATH):
        raise FileNotFoundError(
            f"Handwritten text model not found: {CRNN_MODEL_PATH}"
        )

    try:
        _crnn_model = tf.keras.models.load_model(
            CRNN_MODEL_PATH,
            compile=False
        )

        return _crnn_model

    except Exception as e:
        raise RuntimeError(
            f"Could not load handwritten text model: {e}"
        )


# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

def _to_gray(image):

    if len(image.shape) == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    return image.copy()


# ==================================================
# DIGIT RECOGNITION
# ==================================================

def is_valid_digit_image(image):

    gray = _to_gray(image)

    gray = cv2.resize(
        gray,
        (200, 200)
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) > 30
    ]

    if len(contours) == 0 or len(contours) > 5:
        return False

    largest = max(
        contours,
        key=cv2.contourArea
    )

    largest_area = cv2.contourArea(
        largest
    )

    total_area = sum(
        cv2.contourArea(c)
        for c in contours
    )

    if total_area == 0:
        return False

    if largest_area / total_area < 0.60:
        return False

    x, y, w, h = cv2.boundingRect(
        largest
    )

    image_area = (
        binary.shape[0] *
        binary.shape[1]
    )

    size_ratio = (
        w * h
    ) / image_area

    if size_ratio < 0.005 or size_ratio > 0.75:
        return False

    aspect_ratio = w / float(h)

    return (
        0.15 <= aspect_ratio <= 5.0
    )


def preprocess_digit_image(image):

    gray = _to_gray(image)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    useful = [
        c for c in contours
        if cv2.contourArea(c) > 10
    ]

    if useful:

        largest = max(
            useful,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest
        )

        padding = int(
            max(w, h) * 0.25
        )

        x1 = max(
            0,
            x - padding
        )

        y1 = max(
            0,
            y - padding
        )

        x2 = min(
            binary.shape[1],
            x + w + padding
        )

        y2 = min(
            binary.shape[0],
            y + h + padding
        )

        binary = binary[
            y1:y2,
            x1:x2
        ]

    h, w = binary.shape

    size = max(h, w)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    y_offset = (
        size - h
    ) // 2

    x_offset = (
        size - w
    ) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = binary

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    resized = (
        resized.astype("float32")
        / 255.0
    )

    model = load_digit_model()

    input_shape = model.input_shape

    if len(input_shape) == 4:

        return resized.reshape(
            1,
            28,
            28,
            1
        )

    elif len(input_shape) == 3:

        return resized.reshape(
            1,
            28,
            28
        )

    elif len(input_shape) == 2:

        return resized.reshape(
            1,
            784
        )

    else:

        raise ValueError(
            f"Unsupported model input shape: {input_shape}"
        )


def predict_digit(image):

    if not is_valid_digit_image(image):

        raise ValueError(
            "Invalid image. Please upload one clear "
            "handwritten digit (0-9)."
        )

    model = load_digit_model()

    processed = preprocess_digit_image(
        image
    )

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    confidence = float(
        np.max(probabilities)
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "engine": "MNIST TensorFlow/Keras Model"
    }


# ==================================================
# HANDWRITTEN TEXT RECOGNITION
# ==================================================

def preprocess_text_image(image):

    """
    Same basic image preparation used for
    the trained MLTU sentence recognition model.
    """

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    elif image.shape[-1] == 4:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2RGB
        )

    image = image.astype(
        np.uint8
    )

    # Training used height = 96
    target_height = 96

    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError(
            "Invalid image dimensions."
        )

    scale = (
        target_height / float(h)
    )

    target_width = max(
        1,
        int(w * scale)
    )

    resized = cv2.resize(
        image,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return resized


def predict_text_crnn(image):

    model = load_crnn_model()

    processed = preprocess_text_image(
        image
    )

    # Model was trained with 96px height.
    # Width is padded to the model's expected width.
    model_input_shape = model.input_shape

    expected_height = int(
        model_input_shape[1]
    )

    expected_width = int(
        model_input_shape[2]
    )

    if processed.shape[0] != expected_height:

        processed = cv2.resize(
            processed,
            (
                processed.shape[1],
                expected_height
            ),
            interpolation=cv2.INTER_AREA
        )

    current_width = processed.shape[1]

    if current_width > expected_width:

        processed = cv2.resize(
            processed,
            (
                expected_width,
                expected_height
            ),
            interpolation=cv2.INTER_AREA
        )

    elif current_width < expected_width:

        canvas = np.ones(
            (
                expected_height,
                expected_width,
                3
            ),
            dtype=np.uint8
        ) * 255

        canvas[
            :,
            :current_width
        ] = processed

        processed = canvas

    batch = np.expand_dims(
        processed,
        axis=0
    )

    predictions = model.predict(
        batch,
        verbose=0
    )

    # CTC decoding
    input_length = np.ones(
        predictions.shape[0]
    ) * predictions.shape[1]

    decoded, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_length,
        greedy=True
    )

    decoded = decoded[0].numpy()[0]

    result = ""

    for index in decoded:

        index = int(index)

        if (
            0 <= index < len(VOCAB)
        ):

            result += VOCAB[index]

    return result.strip()


def predict_text(image):

    """
    ONLY the trained CRNN model is used.

    Tesseract is intentionally NOT used as a fallback.
    """

    try:

        prediction = predict_text_crnn(
            image
        )

        return {
            "prediction": prediction,
            "engine": "Trained CRNN Model (model.h5)"
        }

    except Exception as e:

        print(
            "CRNN prediction error:",
            e
        )

        raise RuntimeError(
            f"Handwritten text model prediction failed: {e}"
        )
