from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import base64
import cv2
from PIL import Image, ImageFilter

try:
    with open("www/logo.png", "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    encoded_logo = ""
except Exception:
    encoded_logo = ""

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, -1)
    y_true_filtered = tf.boolean_mask(y_true, mask)
    y_pred_filtered = tf.boolean_mask(y_pred, mask)
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true_filtered, y_pred_filtered)
    )

def is_valid_image_array(img_array, expected_height, expected_width):
    return (isinstance(img_array, np.ndarray) and
            img_array.shape == (1, expected_height, expected_width, 3) and
            img_array.size > 0)

base_path = os.path.dirname(__file__)
clahe_dl_path = os.path.join(base_path, "models", "dlcnn_MS_CLAHE.keras")
clahe_mc_path = os.path.join(base_path, "models", "multiclass_MS_CLAHE.keras")

# LLM ASSIST START
def load_mc_model_safely(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    raw_model = tf.keras.models.load_model(path)
    try:
        seq_model = raw_model.get_layer("sequential")
        return seq_model
    except ValueError:
        inputs = tf.keras.Input(shape=(50, 50, 3), name="wrapped_input")
        outputs = raw_model(inputs)
        rebuilt_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return rebuilt_model

def load_and_build_model(path, input_shape, custom_objects=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)
    dummy_input = tf.zeros((1, *input_shape), dtype=tf.float32)
    try:
        _ = model(dummy_input)
    except Exception:
        raise
    try:
        _ = model.input
    except (ValueError, AttributeError):
        inputs = tf.keras.Input(shape=input_shape, name="custom_input")
        outputs = model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

dlmodel = None
mcmodel = None
try:
    dlmodel = load_and_build_model(
        clahe_dl_path, (100, 100, 3),
        custom_objects={"masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy}
    )
except FileNotFoundError:
    pass
except Exception:
    pass
try:
    mcmodel = load_mc_model_safely(clahe_mc_path)
    dummy_input_mc = tf.zeros((1, 50, 50, 3), dtype=tf.float32)
    _ = mcmodel(dummy_input_mc)
except FileNotFoundError:
    pass
except Exception:
    pass

# LLM ASSIST END

models = {
    "Double Layer": dlmodel,
    "Multiclass": mcmodel,
}
class_labels = {
    "Double Layer": ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "Multiclass": ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "binary": ["Immune", "Tumour"]
}

demo_image_map = {
    "": "None",
    "Clear DCIS 2": "demo/CLEAR_DCIS_2_cell_15_100.png",
    "BLURRY Stromal": "demo/BLURRY_Stromal_cell_26_10.png",
    "Immune": "demo/IMMUNE_2772_100.png",
    "Tumour": "demo/TUMOR_cell_10043_100.png",
}

def encode_image(file_path):
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

last_prediction = {}
prediction_log = []

def mc_predict_class(file_path, model_key):
    if models.get(model_key) is None:
        return "Model not loaded", 0.0
    model = models[model_key]
    labels = class_labels[model_key]
    try:
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(50, 50))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if not is_valid_image_array(img_array, 50, 50):
            return "Invalid Image Data Format", 0.0

        multiclass_pred = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(multiclass_pred[0])
        confidence = float(np.max(multiclass_pred[0]))
        predicted_label = labels[predicted_index]
        return predicted_label, confidence
    except Exception:
        return "Prediction Error", 0.0


def dl_predict_class(file_path, model_key):
    if models.get(model_key) is None:
        return "Model not loaded", 0.0
    model = models[model_key]
    labels = class_labels[model_key]
    try:
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(100, 100))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if not is_valid_image_array(img_array, 100, 100):
            return "Invalid Image Data Format", 0.0

        binary_pred, multiclass_pred = model.predict(img_array, verbose=0)
        if binary_pred[0][0] > 0.5:
            predicted_index = np.argmax(multiclass_pred[0])
            predicted_label = labels[predicted_index]
            confidence = float(np.max(multiclass_pred[0]))
        else:
            predicted_label = "Immune"
            confidence = 1 - float(binary_pred[0][0])
        return predicted_label, confidence
    except Exception:
        return "Prediction Error", 0.0

# LLM ASSIST START
def get_gradcam_heatmap_dl(model, img_tensor, class_index, layer_name="conv2d_1", output_layer_name="multiclass_output"):
    if model is None:
        raise ValueError("DL Model not loaded for Grad-CAM generation.")
    _ = model(img_tensor)
    try:
        conv_layer = model.get_layer(layer_name)
        symbolic_output = model.get_layer(output_layer_name).output
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[conv_layer.output, symbolic_output]
        )
    except Exception:
        raise
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_tensor, tf.float32)
        conv_outputs, predictions = grad_model(inputs, training=False)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def get_gradcam_heatmap_mc(model, img_tensor, class_index, layer_name="conv2d_1", output_layer_name="dense_1"):
    if model is None:
        raise ValueError("MC Model not loaded for Grad-CAM generation.")
    _ = model(img_tensor)
    try:
        if "sequential" in [l.name for l in model.layers]:
            base_model_for_gradcam = model.get_layer("sequential")
        else:
            base_model_for_gradcam = model

        grad_inputs = tf.keras.Input(shape=base_model_for_gradcam.input_shape[1:], name="grad_input_tensor")
        x = grad_inputs
        conv_layer_output_symbolic = None
        output_layer_output_symbolic = None
        for layer in base_model_for_gradcam.layers:
            x = layer(x)
            if layer.name == layer_name:
                conv_layer_output_symbolic = x
            if layer.name == output_layer_name:
                output_layer_output_symbolic = x

        if conv_layer_output_symbolic is None or output_layer_output_symbolic is None:
             raise ValueError(f"Could not find target layers ('{layer_name}' or '{output_layer_name}') in base_model_for_gradcam.")

        grad_model = tf.keras.models.Model(
            inputs=grad_inputs,
            outputs=[conv_layer_output_symbolic, output_layer_output_symbolic]
        )
    except Exception:
        raise
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_gradcam_to_file(img_path, heatmap, cam_path, alpha=0.4):
    original_img_cv = cv2.imread(img_path)
    if original_img_cv is None:
        return

    target_display_size = (250, 250)

    heatmap_resized = cv2.resize(heatmap, target_display_size)
    original_img_resized = cv2.resize(original_img_cv, target_display_size)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, original_img_resized, 1 - alpha, 0)
    cv2.imwrite(cam_path, superimposed_img)


# LLM ASSIST END

def is_blurry(file_path, threshold=1000):
    try:
        img = Image.open(file_path).convert("L").filter(ImageFilter.FIND_EDGES)
        variance = np.var(np.asarray(img))
        return variance > threshold
    except FileNotFoundError:
        return False
    except Exception:
        return False
    

def app_ui(request):
    return ui.page_fluid(
        ui.tags.head(
            ui.tags.title("Interactive H&E Classifier"),
            ui.tags.link(
                rel="stylesheet",
                href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.1/dist/darkly/bootstrap.min.css",
                id="theme-css"
            ),
            ui.tags.link(rel="icon", type="image/png", href="/favicon.png"),
            ui.tags.style("""
                input[type="file"]::file-selector-button {
                    background-color: transparent !important;
                    color: white !important;
                    border: 2px solid white !important;
                    border-radius: 5px !important;
                    padding: 6px 12px !important;
                    font-weight: bold;
                    cursor: pointer;
                }
                input[type="file"]::file-selector-button:hover {
                    background-color: white !important;
                    color: black !important;
                }
                input[type="file"] {
                    background-color: #222 !important;
                    color: #fff !important;
                    border: 1px solid #666 !important;
                    border-radius: 4px !important;
                    padding: 6px;
                }
                /* Style for consistent image box size (now using aspect-ratio) */
                .image-display-container {
                    border: 0px solid #ccc;
                    overflow: hidden;
                    max-width: 100%;
                    margin-bottom: 10px;
                    text-align: center;
                    aspect-ratio: 1 / 1; /* Maintain a square aspect ratio */
                    height: auto; /* Allow height to adjust based on aspect ratio and width */
                    display: flex; /* Use flexbox to center content vertically */
                    align-items: center; /* Center content vertically */
                    justify-content: center; /* Center content horizontally */
                }
                .image-display-container p {
                    margin: 0; /* Remove default paragraph margin */
                    padding: 10px;
                }
            """),
            ui.tags.style("""
                #gradcam-spinner {
                    display: none;
                    text-align: center;
                    margin-top: 10px;
                }
                .spinner-border {
                    width: 3rem;
                    height: 3rem;
                    border-width: 0.4em;
                }
            """),
        ),
        ui.page_navbar(
            ui.nav_panel("", test_image_layout()),
            title=ui.output_ui("app_title"),
        ),
    )

def test_image_layout():
    return ui.layout_columns(
        ui.card(
            ui.h4("Upload & Session Management"),
            ui.tags.div(
                ui.tags.label("Upload H&E Image", style="color: white; font-weight: bold; display: block;"),
                ui.input_file(
                    "image_upload",
                    label=None,
                    accept=[".jpg", ".png", ".tif"],
                    multiple=False
                ),
                style="margin-bottom: 1rem;"
            ),
            ui.input_select(
                "demo_selector",
                "Or Select a Demo Image",
                choices=demo_image_map,
                selected=""
            ),
            ui.tags.div(
                ui.tags.label("Resume from previous session (.csv)", style="color: white; font-weight: bold; display: block; margin-top: 1rem;"),
                ui.input_file(
                    "log_upload",
                    label=None,
                    accept=[".csv"],
                    multiple=False
                ),
                    style="margin-bottom: 1rem;"
            ),
        ),
        ui.card(
            ui.h4("Image Preview & Grad-CAM"),
            ui.input_radio_buttons(
                "image_view_selector",
                "Select Image View:",
                choices={
                    "original": "Original Image",
                    "dl_gradcam": "Double Layer Grad-CAM",
                    "mc_gradcam": "Multiclass Grad-CAM"
                },
                selected="original",
                inline=True
            ),
            ui.output_ui("image_display"),
            ui.output_ui("blur_warning"),
            ui.hr(),
            ui.h4("Prediction Result"),
            ui.output_ui("prediction_result"),
            style="margin-bottom: 20px;"
        ),
        ui.card(
            ui.h4("User Feedback"),
            ui.input_radio_buttons(
                "prediction_agreement",
                "Do you agree with the model’s prediction?",
                choices=["Agree", "Disagree", "Unsure"],
                selected="Agree"
            ),
            ui.input_text_area("user_comment", "Add Comment"),
            ui.output_ui("correct_class_ui"),
            ui.input_action_button("submit_comment", "Log Prediction", class_="btn btn-primary"),
            ui.hr(),
            ui.h4("Prediction Log"),
            ui.output_ui("prediction_table"),
            ui.input_text("csv_filename", "Name your download file", placeholder="e.g., my_session_log"),
            ui.download_button("download_log", "Download Log", class_="btn btn-primary"),
            style="margin-bottom: 20px;"
        )
    )


def server(input, output, session):
    current_selected_image_path = reactive.Value(None)
    current_selected_image_name = reactive.Value(None)
    current_dl_gradcam_path = reactive.Value(None)
    current_mc_gradcam_path = reactive.Value(None)
    current_blur_status = reactive.Value(False)
    current_active_source = reactive.Value("none")

    @reactive.Effect
    def _set_initial_app_state():
        current_selected_image_path.set(None)
        current_selected_image_name.set(None)
        current_dl_gradcam_path.set(None)
        current_mc_gradcam_path.set(None)
        current_blur_status.set(False)
        current_active_source.set("none")

        session.send_input_message("image_upload", {"value": None})
        session.send_input_message("demo_selector", {"value": ""})
        session.send_input_message("image_view_selector", {"value": "original"})
        session.send_input_message("user_comment", {"value": ""})
        session.send_input_message("prediction_agreement", {"value": "Agree"})
        if hasattr(input, 'correct_class'):
            session.send_input_message("correct_class", {"value": "Invasive"})

    @reactive.Effect
    @reactive.event(input.image_upload)
    def handle_upload_event():
        uploaded_file = input.image_upload()
        
        if uploaded_file and uploaded_file[0]["datapath"] is not None:
            file_path = uploaded_file[0]["datapath"]
            file_name = uploaded_file[0]["name"]
            
            if current_active_source.get() != "upload":
                session.send_input_message("demo_selector", {"value": ""})
                current_active_source.set("upload")

            current_selected_image_path.set(file_path)
            current_selected_image_name.set(file_name)
            
        elif current_active_source.get() == "upload" and (uploaded_file is None or not uploaded_file):
            current_selected_image_path.set(None)
            current_selected_image_name.set(None)
            current_active_source.set("none")
            session.send_input_message("demo_selector", {"value": ""})

    @reactive.Effect
    @reactive.event(input.demo_selector)
    def handle_demo_selection_event():
        demo_selection = input.demo_selector()

        if demo_selection and demo_selection != "":
            if current_active_source.get() != "demo":
                session.send_input_message("image_upload", {"value": None})
                current_active_source.set("demo")
            
            demo_image_full_path = os.path.join(base_path, demo_image_map.get(demo_selection, ""))

            print(f"Base path is: {base_path}")
            print(f"Demo selection: {demo_selection}")
            print(f"Full path: {demo_image_full_path}")

            if os.path.exists(demo_image_full_path):
                current_selected_image_path.set(demo_image_full_path)
                current_selected_image_name.set(demo_selection)
            else:
                ui.notification_show(f"Demo image file not found: {demo_image_full_path}", type="warning", duration=5000)
                current_selected_image_path.set(None)
                current_selected_image_name.set(None)
                current_active_source.set("none")
                session.send_input_message("demo_selector", {"value": ""})
        
        elif current_active_source.get() == "demo" and (demo_selection is None or demo_selection == ""):
            current_selected_image_path.set(None)
            current_selected_image_name.set(None)
            current_active_source.set("none")
            session.send_input_message("image_upload", {"value": None})

    @reactive.Effect
    @reactive.event(input.image_upload, input.demo_selector)
    def ensure_cleared_if_nothing_selected():
        uploaded_file = input.image_upload()
        demo_selection = input.demo_selector()
        
        is_upload_empty = not (uploaded_file and uploaded_file[0]["datapath"])
        is_demo_empty = not (demo_selection and demo_selection != "")

        if is_upload_empty and is_demo_empty:
            if current_active_source.get() != "none":
                current_selected_image_path.set(None)
                current_selected_image_name.set(None)
                current_active_source.set("none")

    @reactive.Effect
    @reactive.event(current_selected_image_path)
    def process_and_generate_visualizations():

        file_path = current_selected_image_path.get()
        file_name = current_selected_image_name.get()

        current_dl_gradcam_path.set(None)
        current_mc_gradcam_path.set(None)
        current_blur_status.set(False)
        session.send_input_message("image_view_selector", {"value": "original"})

        if file_path is None:
            return

        current_blur_status.set(is_blurry(file_path))

        try:
            if dlmodel is not None:
                img_dl = tf.keras.preprocessing.image.load_img(file_path, target_size=(100, 100))
                img_array_dl = tf.keras.preprocessing.image.img_to_array(img_dl) / 255.0
                img_array_dl = np.expand_dims(img_array_dl, axis=0)

                binary_pred, multiclass_pred_dl = dlmodel.predict(img_array_dl, verbose=0)
                if binary_pred[0][0] > 0.5:
                    dl_index = np.argmax(multiclass_pred_dl[0])
                    img_tensor_dl = tf.convert_to_tensor(img_array_dl, dtype=tf.float32)
                    dl_heatmap = get_gradcam_heatmap_dl(dlmodel, img_tensor_dl, dl_index)
                    dl_path = os.path.join(base_path, "www", "gradcam_dl_current.jpg")
                    save_gradcam_to_file(file_path, dl_heatmap, cam_path=dl_path)
                    current_dl_gradcam_path.set(dl_path)
                else:
                    current_dl_gradcam_path.set(None)
            else:
                current_dl_gradcam_path.set(None)

            if mcmodel is not None:
                img_mc = tf.keras.preprocessing.image.load_img(file_path, target_size=(50, 50))
                img_array_mc = tf.keras.preprocessing.image.img_to_array(img_mc) / 255.0
                img_array_mc = np.expand_dims(img_array_mc, axis=0)
                img_tensor_mc = tf.convert_to_tensor(img_array_mc, dtype=tf.float32)

                mc_pred = mcmodel.predict(img_array_mc, verbose=0)
                mc_index = np.argmax(mc_pred[0])
                mc_heatmap = get_gradcam_heatmap_mc(mcmodel, img_tensor_mc, mc_index)
                mc_path = os.path.join(base_path, "www", "gradcam_mc_current.jpg")
                save_gradcam_to_file(file_path, mc_heatmap, cam_path=mc_path)
                current_mc_gradcam_path.set(mc_path)
            else:
                current_mc_gradcam_path.set(None)

        except Exception as e:
            ui.notification_show(f"Error generating visualizations: {e}", type="error", duration=5000)
            current_dl_gradcam_path.set(None)
            current_mc_gradcam_path.set(None)
    
    @output
    @render.ui
    def app_title():
        if encoded_logo:
            return ui.tags.div(
                ui.tags.img(
                    src=f"data:image/png;base64,{encoded_logo}",
                    height="90px",
                    style="margin-left: 10px;"
                )
            )
        else:
            return ui.h4("Interactive H&E Classifier", style="margin-left: 10px; margin-top: 15px;")

    @output
    @render.ui
    def image_display():
        selected_view = input.image_view_selector()
        display_path = None
        message_text = "Upload an H&E image for model prediction and visualisation."

        if current_selected_image_path.get() is None:
            return ui.HTML(f"""
                <div class='image-display-container'>
                    <p>{message_text}</p>
                </div>
            """)
        if selected_view == "original":
            display_path = current_selected_image_path.get()
            message_text = None
        elif selected_view == "dl_gradcam":
            display_path = current_dl_gradcam_path.get()
            if dlmodel is None:
                message_text = "Double Layer Model not loaded. Grad-CAM unavailable."
            elif display_path is None or not os.path.exists(display_path):
                message_text = "Double Layer Grad-CAM not available (predicted 'Immune' or generation error)."
            else:
                message_text = None
        elif selected_view == "mc_gradcam":
            display_path = current_mc_gradcam_path.get()
            if mcmodel is None:
                message_text = "Multiclass Model not loaded. Grad-CAM unavailable."
            elif display_path is None or not os.path.exists(display_path):
                 message_text = "Multiclass Grad-CAM not available (generation error)."
            else:
                message_text = None

        if display_path and message_text is None:
            try:
                encoded = encode_image(display_path)
                return ui.HTML(f"""
                    <div class='image-display-container'>
                        <img id='display-img' src='data:image/png;base64,{encoded}'
                            style='width: 100%; height: auto; display: block; margin: 0 auto; object-fit: contain;' draggable="false"/>
                    </div>
                """)
            except Exception:
                return ui.HTML(f"""
                    <div class='image-display-container'>
                        <p>Failed to load image for display.</p>
                    </div>
                """)
        else:
            return ui.HTML(f"""
                <div class='image-display-container'>
                    <p>{message_text}</p>
                </div>
            """)


    @output
    @render.ui
    def prediction_result():
        file_path = current_selected_image_path.get()
        file_name = current_selected_image_name.get()

        if file_path is None or not os.path.exists(file_path):
            return ""

        results = []
        for model_key in ["Double Layer", "Multiclass"]:
            if model_key == "Double Layer" and dlmodel is not None:
                dl_pred, dl_conf = dl_predict_class(file_path, model_key)
                results.append((model_key, dl_pred, dl_conf))
            elif model_key == "Multiclass" and mcmodel is not None:
                mc_pred, mc_conf = mc_predict_class(file_path, model_key)
                results.append((model_key, mc_pred, mc_conf))
            else:
                results.append((model_key, "Model Not Loaded", 0.0))

        if results:
            last_prediction.clear()
            last_prediction.update({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": file_name,
                "dl_prediction": results[0][1],
                "dl_confidence": results[0][2],
                "mc_prediction": results[1][1],
                "mc_confidence": results[1][2],
                "comment": "",
                "agreement": "",
                "suggested": ""
            })

            return ui.div(
                ui.layout_columns(
                    *[
                        ui.card(
                            ui.h5(f"{key}"),
                            ui.p(f"Prediction: {pred}"),
                            ui.p(f"Confidence: {conf:.2%}")
                        )
                        for key, pred, conf in results
                    ]
                ),
                ui.hr(),
            )
        else:
            return ""

    @output
    @render.ui
    def blur_warning():
        if current_blur_status.get():
            return ui.tags.div(
                ui.tags.i(class_="bi bi-exclamation-triangle-fill"),
                ui.tags.span(" This image appears blurry. Model performance may be affected.", style="color: orange; margin-left: 5px;")
            )
        return None

    @reactive.effect
    @reactive.event(input.submit_comment)
    def on_submit_comment():
        if input.submit_comment():
            comment = input.user_comment()
            agreement = input.prediction_agreement()
            suggested = input.correct_class() if agreement == "Disagree" and hasattr(input, 'correct_class') else ""

            if last_prediction and isinstance(last_prediction, dict) and last_prediction.get("filename"):
                entry = last_prediction.copy()
                entry.update({
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "comment": comment,
                    "agreement": agreement,
                    "suggested": suggested,
                })
                prediction_log.append(entry)

                session.send_input_message("user_comment", {"value": ""})
                session.send_input_message("prediction_agreement", {"value": "Agree"})
                if hasattr(input, 'correct_class') and input.correct_class() != "Invasive":
                    session.send_input_message("correct_class", {"value": "Invasive"})

                ui.notification_show("Comment saved with prediction", type="message", duration=3000)
            else:
                ui.notification_show("No prediction available to annotate. Please upload or select an image first.", type="error", duration=3000)

    @output
    @render.ui
    @reactive.event(input.submit_comment, input.log_upload)
    def prediction_table():
        if not prediction_log:
            return ui.p("No prediction log available. Submit a comment to record this prediction.")
        else:
            df = pd.DataFrame(prediction_log)
            required_cols_order = ["timestamp", "filename", "dl_prediction", "dl_confidence",
                                   "mc_prediction", "mc_confidence", "comment", "agreement", "suggested"]
            df = df.reindex(columns=required_cols_order, fill_value="")
            html_table = df.to_html(classes="table table-striped", index=False, border=0)
            return ui.HTML(f"<div class='table-responsive' style='max-height: 300px; overflow-y: auto;'>{html_table}</div>")


    @output
    @render.download(
        filename=lambda: (input.csv_filename().strip() or "prediction_log") + ".csv"
    )
    def download_log():
        def writer():
            df = pd.DataFrame(prediction_log)
            if df.empty:
                yield b"No prediction data available.\n"
            else:
                df_filtered = df[
                    ["timestamp", "filename", "dl_prediction", "dl_confidence",
                    "mc_prediction", "mc_confidence", "comment", "agreement", "suggested"]
                ]
                yield df_filtered.to_csv(index=False).encode("utf-8")
        return writer()

    @reactive.effect
    @reactive.event(input.log_upload)
    def upload_log():
        uploaded_file = input.log_upload()
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file[0]["datapath"])
                prediction_log.clear()
                for index, row in uploaded_df.iterrows():
                    prediction_log.append(row.to_dict())
                ui.notification_show(f"Successfully loaded {len(uploaded_df)} entries from {uploaded_file[0]['name']}", type="message", duration=3000)
            except Exception as e:
                ui.notification_show(f"Error loading log file: {e}", type="error", duration=5000)


    @output
    @render.ui
    def correct_class_ui():
        if input.prediction_agreement() == "Disagree":
            labels_to_offer = class_labels.get("Multiclass", [])
            if not labels_to_offer:
                labels_to_offer = ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive", "Immune", "Stromal"]

            return ui.input_select(
                "correct_class",
                "Select Correct Class:",
                choices=labels_to_offer,
                selected=labels_to_offer[0] if labels_to_offer else ""
            )
        return None

app = App(app_ui, server)