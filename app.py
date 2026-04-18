import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Dropout, MaxPool2D, BatchNormalization, concatenate
from PIL import Image
import os

# --- Custom Model Components (must match training/inference) ---
def dice_bce_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_loss = 1.0 - (2.0 * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.reshape(y_true, [-1, 1]), tf.reshape(y_pred, [-1, 1])))
    return dice_loss + bce

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin)
    return (2.0 * intersection + smooth) / (union + smooth)

class FixedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        return super().update_state(y_true, y_pred_binary, sample_weight)

class EncoderLayerBlock(Layer):
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = Conv2D(filters, 3, padding='same', activation='selu', kernel_initializer='he_normal')
        self.bn1 = BatchNormalization()
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, 3, padding='same', activation='selu', kernel_initializer='he_normal')
        self.bn2 = BatchNormalization()
        self.pool = MaxPool2D(pool_size=(2, 2))
    def call(self, X, training=False):
        x = self.bn1(self.c1(X), training=training)
        x = self.drop(x, training=training)
        x = self.bn2(self.c2(x), training=training)
        return (self.pool(x), x) if self.pooling else x
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'rate': self.rate, 'pooling': self.pooling})
        return config

class DecoderLayerBlock(Layer):
    def __init__(self, filters, rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.cT = Conv2DTranspose(filters, 3, strides=2, padding='same')
        self.next = EncoderLayerBlock(filters, rate, pooling=False)
    def call(self, X, training=False):
        X, skip = X
        return self.next(concatenate([self.cT(X), skip]), training=training)
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'rate': self.rate})
        return config

# --- Streamlit Logic ---

st.set_page_config(page_title="Human Body Matting", layout="wide")

@st.cache_resource
def load_matting_model():
    custom_objects = {
        'dice_bce_loss': dice_bce_loss,
        'dice_coefficient': dice_coefficient,
        'FixedMeanIoU': FixedMeanIoU,
        'EncoderLayerBlock': EncoderLayerBlock,
        'DecoderLayerBlock': DecoderLayerBlock
    }
    return load_model('UNet-Matting-Improved.h5', custom_objects=custom_objects)

st.title("✂️ Human Body Matting & Background Replacement")
st.markdown("Upload an image to extract the person and optionally replace the background.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.3, 0.05)
    st.info("Lower threshold = more inclusive mask.")

col1, col2 = st.columns(2)

with col1:
    source_file = st.file_uploader("Choose Source Image", type=['jpg', 'jpeg', 'png'])
with col2:
    bg_file = st.file_uploader("Choose Background (Optional)", type=['jpg', 'jpeg', 'png'])

if source_file is not None:
    # Load Model
    model = load_matting_model()
    
    # Process Source
    image = Image.open(source_file).convert("RGB")
    original_np = np.array(image)
    h, w = original_np.shape[:2]
    
    # Resize for inference
    IMAGE_SIZE = 224
    img_resized = cv2.resize(original_np, (IMAGE_SIZE, IMAGE_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    pred_mask = model.predict(img_input, verbose=0)[0]
    alpha = cv2.resize(pred_mask, (w, h))
    alpha = np.expand_dims(alpha, axis=-1)
    binary_mask = (alpha > threshold).astype('float32')
    
    # Results containers
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    
    with res_col2:
        st.subheader("Segmented (Black BG)")
        black_bg = np.zeros_like(original_np, dtype=np.float32)
        segmented = (original_np.astype('float32') * binary_mask).astype(np.uint8)
        st.image(segmented, use_container_width=True)
        
    with res_col3:
        if bg_file is not None:
            st.subheader("New Background")
            new_bg = Image.open(bg_file).convert("RGB")
            new_bg_np = np.array(new_bg)
            new_bg_np = cv2.resize(new_bg_np, (w, h))
            
            replaced = (original_np.astype('float32') * binary_mask + 
                        new_bg_np.astype('float32') * (1 - binary_mask))
            replaced = np.clip(replaced, 0, 255).astype(np.uint8)
            st.image(replaced, use_container_width=True)
        else:
            st.subheader("Mask Visualization")
            st.image(binary_mask, use_container_width=True)

    # Download options
    st.divider()
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        res_img = Image.fromarray(segmented)
        import io
        buf = io.BytesIO()
        res_img.save(buf, format="PNG")
        st.download_button("Download Segmented Image", data=buf.getvalue(), file_name="segmented.png", mime="image/png")
    
    if bg_file is not None:
        with dl_col2:
            res_bg_img = Image.fromarray(replaced)
            buf_bg = io.BytesIO()
            res_bg_img.save(buf_bg, format="PNG")
            st.download_button("Download Background Replaced Image", data=buf_bg.getvalue(), file_name="replaced.png", mime="image/png")

else:
    st.info("Please upload a source image to begin.")
