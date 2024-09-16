from tensorflow import keras
from PIL import Image
import numpy as np
import tensorflow as tf

content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

vgg = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet")
vgg.trainable = False

style_outputs = [vgg.get_layer(name).output for name in style_layers]
content_outputs = [vgg.get_layer(name).output for name in content_layers]
model_outputs = style_outputs + content_outputs

model = keras.models.Model(vgg.input, model_outputs)
print(model.summary())

img = Image.open('/images/img.jpg')
img_style = Image.open('/images/img_style.jpg')

x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))
x_img_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))