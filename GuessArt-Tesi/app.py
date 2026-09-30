import streamlit as st
import os
import json
import numpy as np
import tensorflow as tf
import keras
import cv2
from PIL import Image
from vit_keras import vit, utils, visualize
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess


st.set_page_config(
    page_title="GuessArt",
    page_icon="🎨",
    layout="wide"
)



st.markdown(
"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 10%, #2b1055 0%, #12071f 45%, #06040d 100%);
    color: #f4f1fb;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.2rem 1rem 1.4rem 1rem;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    background: linear-gradient(90deg, #ffd76a, #ff8fb1, #9d7dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero p {
    color: #cbbdf0;
    font-size: 1.05rem;
    letter-spacing: 0.03em;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 1.5px dashed #9d7dff66;
    border-radius: 18px;
    padding: 1rem;
}

[data-testid="stFileUploader"] label p {
    color: #f4f1fb !important;
    font-weight: 500;
    font-size: 1rem;
}

section[data-testid="stFileUploaderDropzone"] {
    background: rgba(18, 7, 31, 0.55) !important;
    border: 1.5px dashed rgba(157, 125, 255, 0.55) !important;
    border-radius: 14px !important;
}

section[data-testid="stFileUploaderDropzone"] svg {
    fill: #cbbdf0 !important;
}

div[data-testid="stFileUploaderDropzoneInstructions"] span,
div[data-testid="stFileUploaderDropzoneInstructions"] small,
div[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #e6ddff !important;
}

section[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, #ffd76a, #ff8fb1) !important;
    color: #12071f !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    padding: 0.4rem 1.1rem !important;
}

[data-testid="stFileUploaderFile"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 4px 8px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #f4f1fb !important;
}

[data-testid="stFileUploaderFileName"] {
    color: #f4f1fb !important;
}

/* Model cards */
.model-card {
    background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    backdrop-filter: blur(6px);
}

.model-card h4 {
    margin: 0 0 6px 0;
    font-size: 1.15rem;
    color: #ffd76a;
}

.style-pill {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    background: linear-gradient(90deg, #9d7dff, #ff8fb1);
    color: #12071f;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 6px 0 10px 0;
}

.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
}

.conf-bar-fill {
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ffd76a, #ff8fb1);
}

.conf-label {
    font-size: 0.85rem;
    color: #cbbdf0;
    margin-top: 4px;
}

.winner-banner {
    text-align: center;
    background: linear-gradient(160deg, rgba(255,215,106,0.16), rgba(157,125,255,0.14));
    border: 1px solid rgba(255,215,106,0.4);
    border-radius: 18px;
    padding: 20px 18px;
    margin: 0 0 1.1rem 0;
    box-shadow: 0 8px 22px rgba(0,0,0,0.3);
}

.winner-tag {
    margin: 0;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #cbbdf0;
}

.winner-banner h3 {
    margin: 4px 0 8px 0;
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    color: #ffd76a;
}

.winner-sub {
    margin: 0;
    color: #e6ddff;
    font-size: 0.92rem;
}

.waiting-panel {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 22px;
    color: #cbbdf0;
    text-align: center;
    font-size: 0.95rem;
}

.mini-result {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 8px;
    font-size: 0.9rem;
}

.mini-style {
    color: #ffd76a;
    font-weight: 500;
}

.mini-conf {
    color: #cbbdf0;
    font-variant-numeric: tabular-nums;
}

.section-gap {
    margin-top: 1.8rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 0.4rem;
}

/* Buttons */
div.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.06);
    color: #f4f1fb;
    font-weight: 500;
    padding: 0.4rem 1.1rem;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: #ffd76a;
    color: #ffd76a;
    background: rgba(255,215,106,0.08);
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #ffd76a, #ff8fb1);
    color: #12071f;
    border: none;
    font-weight: 700;
}

hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""",
unsafe_allow_html=True
)



RESNET_DIR = (
    "/kaggle/input/models/"
    "emanueladagostino/resnet50/"
    "keras/default/1"
)

DENSENET_DIR = (
    "/kaggle/input/models/"
    "emanueladagostino/dense121/"
    "keras/default/1"
)

VIT_DIR = (
    "/kaggle/input/models/"
    "emanueladagostino/vitkeras/"
    "keras/default/1"
)


class_names = [
    "Abstract_Expressionism",
    "Baroque",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Impressionism",
    "Pop_Art",
    "Romanticism",
    "Symbolism",
    "Ukiyo_e"
]

IMG_SIZE = 224


@st.cache_resource
def load_model_folder(model_dir, preprocess_name):

    if preprocess_name == "resnet":
        preprocess = resnet_preprocess
    elif preprocess_name == "densenet":
        preprocess = densenet_preprocess
    elif preprocess_name == "vit":
        preprocess = vit.preprocess_inputs

    with open(f"{model_dir}/config.json") as f:
        full_config = json.load(f)

    model = keras.saving.deserialize_keras_object(
        full_config,
        custom_objects={
            "preprocess_input": preprocess,
            "preprocess_inputs": preprocess
        },
        safe_mode=False
    )

    model.load_weights(f"{model_dir}/model.weights.h5")

    return model


@st.cache_resource
def load_all_models():
    model_resnet = load_model_folder(RESNET_DIR, "resnet")
    model_dense = load_model_folder(DENSENET_DIR, "densenet")
    model_vit = load_model_folder(VIT_DIR, "vit")
    return model_resnet, model_dense, model_vit


with st.spinner("Caricamento modelli..."):
    model_resnet, model_dense, model_vit = load_all_models()


def prepare_image(image):
   
    img = image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32")

    raw_batch = np.expand_dims(img_array, axis=0)

    resnet_batch = resnet_preprocess(raw_batch.copy())
    dense_batch = densenet_preprocess(raw_batch.copy())
    vit_batch = vit.preprocess_inputs(raw_batch.copy())

    return img_array, raw_batch, resnet_batch, dense_batch, vit_batch


def find_model_layers(model):
    backbone = None
    gap = None
    dropout = None
    dense_hidden = None
    output = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer.name
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap = layer.name
        elif isinstance(layer, tf.keras.layers.Dropout):
            dropout = layer.name
        elif isinstance(layer, tf.keras.layers.Dense):
            if dense_hidden is None:
                dense_hidden = layer.name
            else:
                output = layer.name

    return backbone, gap, dropout, dense_hidden, output


resnet_layers = find_model_layers(model_resnet)
dense_layers = find_model_layers(model_dense)


def make_gradcam(model, backbone_name, last_conv_layer_name, gap_name,
                  dropout_name, dense_name, output_name, image):

    backbone = model.get_layer(backbone_name)

    grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output]
    )

    with tf.GradientTape() as tape:
        conv_output, features = grad_model(image)
        x = model.get_layer(gap_name)(features)
        x = model.get_layer(dropout_name)(x, training=False)
        x = model.get_layer(dense_name)(x)
        prediction = model.get_layer(output_name)(x)
        class_index = tf.argmax(prediction[0])
        loss = prediction[:, class_index]

    gradients = tape.gradient(loss, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_gradients[..., None]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value != 0:
        heatmap /= max_value

    return heatmap.numpy()


def apply_gradcam(original, heatmap):
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(original.astype("uint8"), 0.6, heatmap, 0.4, 0)
    return result


def make_attention_map(model, image):
    
    from vit_keras import layers as vit_layers

    vit_base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            vit_base = layer
            break

    if vit_base is None:
        raise Exception("Backbone ViT non trovato")

    size = vit_base.input_shape[1]
    grid_size = int(np.sqrt(vit_base.layers[5].output[0].shape[-2] - 1))

    X = vit.preprocess_inputs(cv2.resize(image, (size, size)))[np.newaxis, :]

    outputs = [
        l.output[1] for l in vit_base.layers
        if isinstance(l, vit_layers.TransformerBlock)
    ]

    attention_extractor = tf.keras.models.Model(
        inputs=vit_base.inputs,
        outputs=outputs
    )

    weights = np.array(attention_extractor.predict(X, verbose=0))

    num_layers = weights.shape[0]
    num_heads = weights.shape[2]

    reshaped = weights.reshape(
        (num_layers, num_heads, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )

    reshaped = reshaped.mean(axis=1)

    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = mask / mask.max()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    return apply_gradcam(image, mask)


def run_predictions(raw_batch):
    pred_resnet = model_resnet.predict(raw_batch, verbose=0)[0]
    pred_dense = model_dense.predict(raw_batch, verbose=0)[0]
    pred_vit = model_vit.predict(raw_batch, verbose=0)[0]

    results = {
        "ResNet50V2": {
            "style": class_names[int(np.argmax(pred_resnet))],
            "conf": float(np.max(pred_resnet)),
            "icon": "🟠"
        },
        "DenseNet121": {
            "style": class_names[int(np.argmax(pred_dense))],
            "conf": float(np.max(pred_dense)),
            "icon": "🟣"
        },
        "Vision Transformer": {
            "style": class_names[int(np.argmax(pred_vit))],
            "conf": float(np.max(pred_vit)),
            "icon": "🔵"
        }
    }
    return results


def compute_xai(model_key, original, resnet_batch, dense_batch, vit_batch):
    if model_key == "ResNet50V2":
        backbone_r, gap_r, drop_r, dense_r, out_r = resnet_layers
        heatmap_r = make_gradcam(
            model_resnet, backbone_r, "post_relu",
            gap_r, drop_r, dense_r, out_r, resnet_batch
        )
        return apply_gradcam(original, heatmap_r)

    elif model_key == "DenseNet121":
        backbone_d, gap_d, drop_d, dense_d, out_d = dense_layers
        heatmap_d = make_gradcam(
            model_dense, backbone_d, "conv5_block16_concat",
            gap_d, drop_d, dense_d, out_d, dense_batch
        )
        return apply_gradcam(original, heatmap_d)

    elif model_key == "Vision Transformer":
        
        return make_attention_map(model_vit, original.astype("uint8"))


if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "results" not in st.session_state:
    st.session_state.results = None
if "prepared" not in st.session_state:
    st.session_state.prepared = None
if "xai_visible" not in st.session_state:
    st.session_state.xai_visible = {}
if "xai_cache" not in st.session_state:
    st.session_state.xai_cache = {}
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def reset_session():
    st.session_state.uploaded_bytes = None
    st.session_state.results = None
    st.session_state.prepared = None
    st.session_state.xai_visible = {}
    st.session_state.xai_cache = {}
    st.session_state.uploader_key += 1
    st.rerun()



st.markdown(
"""
<div class="hero">
    <h1>🎨GuessArt🎨</h1>
    <p>✨Classificazione degli stili artistici✨</p>
</div>
""",
unsafe_allow_html=True
)

top_left, top_right = st.columns([5, 1])
with top_right:
    if st.button("Clear🗑️", use_container_width=True):
        reset_session()

uploaded_file = st.file_uploader(
    "Carica un dipinto",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None and st.session_state.uploaded_bytes != uploaded_file.getvalue():
    st.session_state.uploaded_bytes = uploaded_file.getvalue()
    st.session_state.results = None
    st.session_state.prepared = None
    st.session_state.xai_visible = {}
    st.session_state.xai_cache = {}

if st.session_state.uploaded_bytes is not None:

    image = Image.open(__import__("io").BytesIO(st.session_state.uploaded_bytes))

    img_col, result_col = st.columns([1, 1], gap="large")

    with img_col:
        st.image(image, caption="Immagine caricata", use_container_width=True)

    with result_col:
        if st.session_state.results is None:
            st.write("")
            st.markdown(
                '<div class="waiting-panel">L\'opera è pronta: avvia l\'analisi per '
                'scoprire lo stile artistico.</div>',
                unsafe_allow_html=True
            )
            st.write("")
            if st.button("Analizza l'opera🔍 ", type="primary", use_container_width=True):
                with st.spinner("Analisi in corso..."):
                    original, raw_batch, resnet_batch, dense_batch, vit_batch = prepare_image(image)
                    st.session_state.prepared = {
                        "original": original,
                        "raw_batch": raw_batch,
                        "resnet_batch": resnet_batch,
                        "dense_batch": dense_batch,
                        "vit_batch": vit_batch
                    }
                    st.session_state.results = run_predictions(raw_batch)
                    st.session_state.xai_visible = {}
                    st.session_state.xai_cache = {}
                    st.rerun()
        else:
            results = st.session_state.results
            winner_key = max(results, key=lambda k: results[k]["conf"])
            winner = results[winner_key]

            st.markdown(
            f"""
            <div class="winner-banner">
                <p class="winner-tag">Stile riconosciuto🏆</p>
                <h3>{winner['style'].replace('_',' ')}</h3>
                <p class="winner-sub">Modello più sicuro: <b>{winner_key}</b> ({winner['conf']*100:.2f}%)</p>
            </div>
            """,
            unsafe_allow_html=True
            )

            for model_key, r in results.items():
                st.markdown(
                f"""
                <div class="mini-result">
                    <span>{r['icon']} {model_key}</span>
                    <span class="mini-style">{r['style'].replace('_',' ')}</span>
                    <span class="mini-conf">{r['conf']*100:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True
                )

            if st.button("Rianalizza🔄 ", use_container_width=True):
                st.session_state.results = None
                st.session_state.prepared = None
                st.session_state.xai_visible = {}
                st.session_state.xai_cache = {}
                st.rerun()

    if st.session_state.results is not None:

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.subheader("Dettagli modelli🔍")

        cols = st.columns(3)

        for col, (model_key, r) in zip(cols, results.items()):
            with col:
                st.markdown(
                f"""
                <div class="model-card">
                    <h4>{r['icon']} {model_key}</h4>
                    <div class="style-pill">{r['style'].replace('_',' ')}</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{r['conf']*100:.1f}%;"></div>
                    </div>
                    <div class="conf-label">Confidenza: {r['conf']*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
                )

                if st.button(f"Spiegabilità🧠", key=f"xai_btn_{model_key}", use_container_width=True):
                    st.session_state.xai_visible[model_key] = not st.session_state.xai_visible.get(model_key, False)

                if st.session_state.xai_visible.get(model_key, False):
                    with st.spinner("Generazione mappa XAI..."):
                        if model_key not in st.session_state.xai_cache:
                            p = st.session_state.prepared
                            xai_img = compute_xai(
                                model_key,
                                p["original"],
                                p["resnet_batch"],
                                p["dense_batch"],
                                p["vit_batch"]
                            )
                            st.session_state.xai_cache[model_key] = xai_img

                        caption = "Grad-CAM" if model_key != "Vision Transformer" else "Attention Map"
                        st.image(
                            st.session_state.xai_cache[model_key],
                            caption=f"{caption} — {model_key}",
                            use_container_width=True
                        )
else:
    st.info("Carica un'immagine per iniziare l'analisi.")