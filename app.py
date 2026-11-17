from pathlib import Path
import streamlit as st
import tensorflow as tf

IMG = 224
HERE = Path(__file__).parent
SAMPLES = HERE / "app_samples"

st.set_page_config(page_title="Chest X-ray Pneumonia Classifier", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(HERE / "final_model.keras")

def preprocess(image_bytes):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG, IMG))
    return tf.expand_dims(img, 0)

st.title("Chest X-ray Pneumonia Classifier")
st.caption("Class mini project. Fine-tuned MobileNetV2 trained on the Kaggle chest X-ray "
           "dataset. Educational demo only. Not a medical device.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
        help="Lower: catches more pneumonia but gives more false alarms. "
             "Higher: fewer false alarms but more missed cases.")
    st.caption("Tip: 0.09 is the threshold chosen on the validation set. It catches "
               "more pneumonia but gives more false alarms.")
    st.markdown("**Results on the official test set (threshold 0.5)**")
    st.markdown("- Accuracy: 92.5%\n- Recall: 95.9%\n- Specificity: 86.8%\n- ROC-AUC: 0.976")

uploaded = st.file_uploader("Upload a chest X-ray (JPEG or PNG)", type=["jpg", "jpeg", "png"])
sample_files = sorted(SAMPLES.glob("*.jp*g")) if SAMPLES.exists() else []
choice = st.selectbox("...or choose a sample image",
                      ["(none)"] + [p.name for p in sample_files],
                      disabled=uploaded is not None)

data, true_label, source = None, None, None
if uploaded is not None:
    data, source = uploaded.getvalue(), uploaded.name
elif choice != "(none)":
    data, source = (SAMPLES / choice).read_bytes(), choice
    true_label = choice.split("_")[0]

if data is not None:
    model = load_model()
    prob = float(model.predict(preprocess(data), verbose=0)[0][0])
    st.caption(f"Source: {source}")
    left, right = st.columns(2)
    left.image(data, caption="Input image")
    with right:
        st.subheader("Prediction")
        if prob >= threshold:
            st.error("Pneumonia likely")
        else:
            st.success("Normal likely")
        st.metric("Pneumonia probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Decision threshold: {threshold:.2f}")
        if true_label:
            st.caption(f"True label of this sample: {true_label}")

with st.expander("About this model and its limitations"):
    st.markdown(
        "- Trained on one public dataset of mostly paediatric chest X-rays.\n"
        "- On the official test set about 13% of healthy images are flagged as pneumonia.\n"
        "- Image size alone predicts the class fairly well in this dataset, so the model "
        "may partly use such cues.\n"
        "- Not validated on other hospitals or adults. Never use it for real diagnosis."
    )
