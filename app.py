import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Photo Editor", layout="wide")

st.title("📸 Photo Editor using OpenCV & Streamlit")

# -----------------------------------
# Functions
# -----------------------------------

def adjust_brightness_contrast(img, brightness=0, contrast=1):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def apply_warm_filter(img):
    warm = img.copy()

    # Increase Red
    warm[:, :, 2] = cv2.add(warm[:, :, 2], 30)

    # Slight decrease Blue
    warm[:, :, 0] = cv2.subtract(warm[:, :, 0], 10)

    return warm


def sharpen_image(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(img, -1, kernel)


def portrait_blur(img):
    h, w = img.shape[:2]

    center_x = w // 2
    center_y = h // 2

    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.circle(
        mask,
        (center_x, center_y),
        min(h, w)//4,
        255,
        -1
    )

    blurred = cv2.GaussianBlur(img, (31, 31), 0)

    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    result = img.copy()

    for c in range(3):
        result[:, :, c] = (
            img[:, :, c] * (mask / 255.0)
            + blurred[:, :, c] * (1 - mask / 255.0)
        )

    return result.astype(np.uint8)


def sketch_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted = 255 - gray

    blur = cv2.GaussianBlur(inverted, (21, 21), 0)

    inverted_blur = 255 - blur

    sketch = cv2.divide(gray, inverted_blur, scale=256.0)

    return sketch


def cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    color = cv2.bilateralFilter(img, 9, 300, 300)

    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon


def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    return edges


# -----------------------------------
# Upload Image
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    edited = image.copy()

    st.sidebar.header("Image Controls")

    # Resize
    resize_percent = st.sidebar.slider(
        "Resize (%)",
        20,
        200,
        100
    )

    width = int(
        edited.shape[1] * resize_percent / 100
    )

    height = int(
        edited.shape[0] * resize_percent / 100
    )

    edited = cv2.resize(
        edited,
        (width, height)
    )

    # Brightness
    brightness = st.sidebar.slider(
        "Brightness",
        -100,
        100,
        0
    )

    # Contrast
    contrast = st.sidebar.slider(
        "Contrast",
        0.5,
        3.0,
        1.0
    )

    edited = adjust_brightness_contrast(
        edited,
        brightness,
        contrast
    )

    # Grayscale
    grayscale = st.sidebar.checkbox(
        "Convert to Grayscale"
    )

    if grayscale:
        edited = cv2.cvtColor(
            edited,
            cv2.COLOR_BGR2GRAY
        )

    # Blur
    blur = st.sidebar.slider(
        "Blur",
        0,
        25,
        0
    )

    if blur > 0:

        k = blur

        if k % 2 == 0:
            k += 1

        if len(edited.shape) == 2:
            edited = cv2.GaussianBlur(
                edited,
                (k, k),
                0
            )
        else:
            edited = cv2.GaussianBlur(
                edited,
                (k, k),
                0
            )

    # Warm Filter
    if st.sidebar.button("Apply Warm Filter"):

        if len(edited.shape) == 3:
            edited = apply_warm_filter(
                edited
            )

    # Portrait Blur
    if st.sidebar.button(
            "Portrait Background Blur"
    ):

        if len(edited.shape) == 3:
            edited = portrait_blur(
                edited
            )

    # Sharpen
    if st.sidebar.button(
            "Sharpen Image"
    ):

        if len(edited.shape) == 3:
            edited = sharpen_image(
                edited
            )

    # Rotation
    rotate = st.sidebar.slider(
        "Rotate",
        0,
        360,
        0
    )

    if rotate != 0:

        h, w = edited.shape[:2]

        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            rotate,
            1.0
        )

        edited = cv2.warpAffine(
            edited,
            matrix,
            (w, h)
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Extra Features")

    if st.sidebar.button("Edge Detection"):

        edited = edge_detection(
            edited if len(edited.shape) == 3
            else cv2.cvtColor(
                edited,
                cv2.COLOR_GRAY2BGR
            )
        )

    if st.sidebar.button("Sketch Effect"):

        if len(edited.shape) == 3:
            edited = sketch_effect(
                edited
            )

    if st.sidebar.button("Cartoon Effect"):

        if len(edited.shape) == 3:
            edited = cartoon_effect(
                edited
            )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

    with col2:

        st.subheader("Edited")

        if len(edited.shape) == 2:
            st.image(
                edited,
                use_container_width=True
            )
        else:
            st.image(
                cv2.cvtColor(
                    edited,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

    # Download
    if len(edited.shape) == 2:

        pil_img = Image.fromarray(
            edited
        )

    else:

        pil_img = Image.fromarray(
            cv2.cvtColor(
                edited,
                cv2.COLOR_BGR2RGB
            )
        )

    buf = io.BytesIO()

    pil_img.save(
        buf,
        format="PNG"
    )

    byte_im = buf.getvalue()

    st.download_button(
        label="⬇ Download Edited Image",
        data=byte_im,
        file_name="edited_image.png",
        mime="image/png"
    )