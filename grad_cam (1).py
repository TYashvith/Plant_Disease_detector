import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, pred_index=None):

    # Find base model safely
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Base model not found")

    last_conv_layer = base_model.get_layer("Conv_1")

    # Feature extractor
    conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # Classifier head
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    passed = False
    for layer in model.layers:
        if layer.name == base_model.name:
            passed = True
            continue
        if passed:
            x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:

        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # ✅ SAFETY FIX
    if grads is None:
        raise RuntimeError("Gradients are None — graph disconnected")

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


def explain_prediction(pil_image, model, idx_to_class):

    pil_image = pil_image.convert("RGB")
    img = pil_image.resize((224,224))

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch, verbose=0)[0]

    pred_idx = np.argmax(preds)

    label = idx_to_class[pred_idx]
    confidence = float(preds[pred_idx])

    top3_idx = preds.argsort()[-3:][::-1]

    top3 = [(idx_to_class[i], float(preds[i])) for i in top3_idx]

    heatmap = make_gradcam_heatmap(img_batch, model, pred_idx)

    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    heatmap = cv2.resize(heatmap, (224,224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return label, confidence, top3, overlay