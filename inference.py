import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Dropout, MaxPool2D, BatchNormalization, concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define custom loss and metrics to load the model properly
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

# Custom Layers used in the architecture
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

# Dictionary of custom objects required by load_model
custom_objects = {
    'dice_bce_loss': dice_bce_loss,
    'dice_coefficient': dice_coefficient,
    'FixedMeanIoU': FixedMeanIoU,
    'EncoderLayerBlock': EncoderLayerBlock,
    'DecoderLayerBlock': DecoderLayerBlock
}

def load_matting_model(model_path='UNet-Matting-Improved.h5'):
    print(f'Loading model from {model_path}...')
    model = load_model(model_path, custom_objects=custom_objects)
    return model

def segment_image(model, image_path, bg_path=None, output_segmented="output_segmented.png", output_replaced="output_replaced.png", threshold=0.5):
    """ 
    Segments a given image. 
    Saves:
    1. Segmented subject with a black background.
    2. Subject with a new background if bg_path is provided.
    """
    IMAGE_SIZE = 224
    
    # Load and preprocess image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img.shape[:2]
    
    # Resize and normalize for model
    img_resized = cv2.resize(original_img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Predict the matte/mask
    pred_mask = model.predict(img_input, verbose=0)[0]
    
    # Resize mask back to original image size
    alpha = cv2.resize(pred_mask, (w, h))
    alpha = np.expand_dims(alpha, axis=-1)
    
    # Thresholding for binary matte
    binary_mask = (alpha > threshold).astype('float32')
    
    # 1. Background Removal (Black background)
    black_bg = np.zeros_like(original_img_rgb, dtype=np.float32)
    segmented = original_img_rgb.astype('float32') * binary_mask + black_bg * (1 - binary_mask)
    segmented = segmented.astype(np.uint8)
    cv2.imwrite(output_segmented, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved to {output_segmented}")

    # 2. Background Replacement
    if bg_path and os.path.exists(bg_path):
        new_bg = cv2.imread(bg_path)
        new_bg = cv2.cvtColor(new_bg, cv2.COLOR_BGR2RGB)
        new_bg = cv2.resize(new_bg, (w, h))
        
        replaced = original_img_rgb.astype('float32') * binary_mask + new_bg.astype('float32') * (1 - binary_mask)
        replaced = np.clip(replaced, 0, 255).astype(np.uint8)
        cv2.imwrite(output_replaced, cv2.cvtColor(replaced, cv2.COLOR_RGB2BGR))
        print(f"Background-changed image saved to {output_replaced}")
        return segmented, replaced
        
    return segmented, None

if __name__ == "__main__":
    import os
    test_image_path = "./image.png"
    background_path = "./background.png"
    
    if os.path.exists(test_image_path):
        model = load_matting_model('UNet-Matting-Improved.h5')
        segment_image(
            model, 
            test_image_path, 
            bg_path=background_path,
            output_segmented="segmented_black.png",
            output_replaced="segmented_new_bg.png",
            threshold=0.3
        )
    else:
        print(f"Please provide a valid image path instead of '{test_image_path}'.")
