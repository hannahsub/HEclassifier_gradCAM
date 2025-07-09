from shiny import App, ui, render
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix
from shiny import reactive
import tensorflow as tf
import os
import base64

with open("www/logo.png", "rb") as f:
    encoded_logo = base64.b64encode(f.read()).decode()


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, -1)
    y_true_filtered = tf.boolean_mask(y_true, mask)
    y_pred_filtered = tf.boolean_mask(y_pred, mask)

    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true_filtered, y_pred_filtered)
    )


base_path = os.path.dirname(__file__)
clahe_path = os.path.join(base_path, "models", "dlcnn_MS_CLAHE.keras")
nonclahe_path = os.path.join(base_path, "models", "multiclass_MS_NONCLAHE.keras")

dlmodel = tf.keras.models.load_model(
    clahe_path,
    custom_objects={"masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy}
)
mcmodel = tf.keras.models.load_model(nonclahe_path)

models = {
    "Double Layer": dlmodel,
    "Multiclass": mcmodel,
}

class_labels = {
    "Double Layer": ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "dlcnn_nonclahe": ["Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"], 
    "Multiclass": ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "multiclass_nonclahe": ["Immune", "Invasive", "DCIS 1", "DCIS 2", "Prolif Invasive"],
    "binary": ["Immune", "Tumour"]
}

demo_image_map = {
    "Clear DCIS 2": "demo/CLEAR_DCIS_2_cell_15_100.png",
    "BLURRY Stromal": "demo/BLURRY_Stromal_cell_26_10.png",
    "Immune": "demo/IMMUNE_2772_100.png",
    "Tumour": "demo/TUMOR_cell_10043_100.png",
}

model_outputs = {
    "Double Layer": {
        "ytrue": "models/outputs/dlcnn_MS_CLAHE_ytrue.npy",
        "ypred": "models/outputs/dlcnn_MS_CLAHE_ypred.npy",
        "conf": "models/outputs/dlcnn_MS_CLAHE_conf.npy",
        "history": "models/history/dlcnn_MS_CLAHE_history.csv"
    },
    "Multiclass": {
        "ytrue": "models/outputs/multiclass_CLAHE_ytrue.npy",
        "ypred": "models/outputs/multiclass_CLAHE_ypred.npy",
        "conf": "models/outputs/multiclass_CLAHE_confidence_values.npy"
    }
}

y_true = np.load(model_outputs["Double Layer"]["ytrue"])
y_pred = np.load(model_outputs["Double Layer"]["ypred"])
history_df = pd.read_csv(model_outputs["Double Layer"]["history"])
confidences_dict = {
    key: np.load(paths["conf"]) for key, paths in model_outputs.items()
}

msacc = round((np.sum(y_true == y_pred) / len(y_pred)) * 100, 1)

def add_immune_padding(y_true, y_pred, immune_class_index=4, extra=0.9144 * 3000):
    pad = np.ones(round(extra)) * immune_class_index
    return np.concatenate((y_true, pad)), np.concatenate((y_pred, pad))

y_immunetrue, y_immunepred = add_immune_padding(y_true, y_pred)
msimmacc = round((np.sum(y_immunetrue == y_immunepred) / len(y_immunepred)) * 100, 1)

def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

last_prediction = {}
prediction_log = []

def mc_predict_class(file_path, model_key):
    print("Selected model key:", model_key)
    if model_key not in models:
        print("Model key not found in models dictionary")
        return "Invalid model selected", 0.0

    model = models[model_key]
    labels = class_labels[model_key]

    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    multiclass_pred = model.predict(img_array)
    
    predicted_index = np.argmax(multiclass_pred[0])
    confidence = float(np.max(multiclass_pred[0])) 

    predicted_label = labels[predicted_index]
    return predicted_label, confidence

def dl_predict_class(file_path, model_key):
    print("Selected model key:", model_key)
    if model_key not in models:
        print("Model key not found in models dictionary")
        return "Invalid model selected", 0.0

    model = models[model_key]
    labels = class_labels[model_key]

    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    binary_pred, multiclass_pred = model.predict(img_array)

    if binary_pred[0][0] > 0.5:
        predicted_index = np.argmax(multiclass_pred[0])
        predicted_label = labels[predicted_index]
        confidence = float(np.max(multiclass_pred[0]))
    else:
        predicted_label = "Immune"
        confidence = 1 - float(binary_pred[0][0])
        
    return predicted_label, confidence

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
            """),
        ),

        ui.page_navbar(
            ui.nav_panel("", test_image_layout()),
            title=ui.output_ui("app_title"),
        ),
    )

def test_image_layout():
    return ui.layout_sidebar(
        ui.sidebar(
            "Upload an H&E image for model prediction and visualization.",
        ),
        ui.layout_columns(
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
                    selected=None
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
                ui.h4("Image Preview"),
                ui.output_ui("image_preview"),
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
    )

from PIL import Image, ImageFilter

def is_blurry(file_path, threshold=1000):
    img = Image.open(file_path).convert("L").filter(ImageFilter.FIND_EDGES)
    variance = np.var(np.asarray(img))
    return variance > threshold

def server(input, output, session):

    @output
    @render.ui
    def clahe_conf_comparison():
        bins = input.bins()
        html_str = dual_confidence_plot(
            conf1=confidences_dict["multiclass_CLAHE"],
            conf2=confidences_dict["dlcnn_MS_CLAHE"],
            label1="CLAHE - 1 Layer",
            label2="CLAHE - 2 Layer",
            bins=bins
        )
        return ui.HTML(html_str)

    @output
    @render.ui
    def image_preview():
        demo_label = input.demo_selector()
        uploaded = input.image_upload()

        if demo_label:
            file_path = os.path.join(base_path, demo_image_map.get(demo_label))
        elif uploaded:
            file_path = uploaded[0]["datapath"]
        else:
            return ui.p("No image uploaded")

        try:
            encoded = encode_image(file_path)
            return ui.HTML(f"""
                <div style='border: 1px solid #ccc; overflow: hidden; max-width: 100%; margin-bottom: 10px;'>
                    <img id='panzoom-img' src='data:image/png;base64,{encoded}'
                        style='width: 100%; cursor: grab;' draggable="false"/>
                </div>
                <span class='badge bg-success' style='color: #000; font-weight: bold; background-color: #28a745; padding: 8px 16px; font-size: 1rem; display: inline-block;'>
                    Prediction complete
                </span>
            """)
        except Exception as e:
            return ui.p(f"Failed to load image: {e}")

    @output
    @render.ui
    def app_title():
        return ui.tags.div(
            ui.tags.img(
                src=f"data:image/png;base64,{encoded_logo}",
                height="90px",
                style="margin-left: 10px;"
     ))

    @output
    @render.ui
    def prediction_result():
        demo_label = input.demo_selector()
        uploaded = input.image_upload()

        if demo_label:
            file_path = os.path.join(base_path, demo_image_map.get(demo_label))
            file_name = demo_label
        elif uploaded:
            file_path = uploaded[0]["datapath"]
            file_name = uploaded[0]["name"]
        else:
            return ""

        results = []
        for model_key in ["Double Layer", "Multiclass"]:
            if model_key == "Double Layer":
                dl_pred, dl_conf = dl_predict_class(file_path, model_key)
                results.append((model_key, dl_pred, dl_conf))
            if model_key == "Multiclass":
                mc_pred, mc_conf = mc_predict_class(file_path, model_key)
                results.append((model_key, mc_pred, mc_conf))

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

    @reactive.Effect
    def clear_initial_demo():
        session.send_input_message("demo_selector", {"value": None})

    @reactive.effect
    @reactive.event(input.submit_comment)
    def on_submit_comment():
        if input.submit_comment():
            comment = input.user_comment()
            agreement = input.prediction_agreement()
            suggested = input.correct_class() if agreement == "Disagree" else ""
    
            if last_prediction and isinstance(last_prediction, dict):
                entry = last_prediction.copy()
                entry.update({
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "comment": comment,
                    "agreement": agreement,
                    "suggested": suggested,
                })
                prediction_log.append(entry)
                
                session.send_input_message("user_comment", {"value": ""})
                session.send_input_message("region_of_interest", {"value": ""})
                session.send_input_message("prediction_agreement", {"value": "Agree"})
    
                ui.notification_show("Comment saved with prediction", type="message", duration=3000)
            else:
                ui.notification_show("No prediction available to annotate", type="error", duration=3000)

    @output
    @render.ui
    @reactive.event(input.submit_comment)
    def prediction_table():
        if not prediction_log:
            return ui.p("No prediction log available. Submit a comment to record this prediction.")
        else:
            df = pd.DataFrame(prediction_log)
            df.columns = ["timestamp", "filename", "dl_prediction", "dl_confidence", "mc_prediction", "mc_confidence","comment", "agreement", "suggested"]
            html_table = df.to_html(classes="table table-striped", index=False, border=0)
            return ui.HTML(f"<div class='table-responsive'>{html_table}</div>")

    def write_log_csv(out_file):
        df = pd.DataFrame(prediction_log)
        if df.empty:
            out_file.write("No prediction data available.")
        else:
            df = df[
                ["timestamp", "filename", "dl_prediction", "dl_confidence",
                "mc_prediction", "mc_confidence", "comment", "agreement", "suggested"]
            ]
            df.to_csv(out_file, index=False)

        script = """
            const nav = document.querySelector('.navbar');
            if (nav) {
                nav.classList.toggle('navbar-dark', arguments[0]);
                nav.classList.toggle('navbar-light', !arguments[0]);
                nav.style.backgroundColor = arguments[0] ? '#222' : '#f8f9fa';
            }
            const title = document.querySelector('.navbar-brand');
            if (title) {
                title.style.color = arguments[0] ? '#fff' : '#000';
            }
        """

    @reactive.Effect
    @reactive.event(input.image_upload)
    def reset_demo_on_upload():
        session.send_input_message("demo_selector", {"value": None})
        

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
        
    @output
    @render.ui
    def correct_class_ui():
        if input.prediction_agreement() == "Disagree":
            return ui.input_select(
                "correct_class",
                "Suggested Correct Class",
                choices=[
                    "Immune",
                    "Invasive",
                    "DCIS 1",
                    "DCIS 2",
                    "Prolif Invasive"
                ]
            )
        else:
            return ui.div()
    
    
    @output
    @render.ui
    def blur_warning():
        demo_label = input.demo_selector()
        uploaded = input.image_upload()

        if demo_label:
            file_path = os.path.join(base_path, demo_image_map.get(demo_label))
        elif uploaded:
            file_path = uploaded[0]["datapath"]
        else:
            return ""

        if is_blurry(file_path):
            return ui.tags.span("Warning! Poor image quality detected: This image may be difficult to classify", class_="badge bg-danger")
        else:
            return ""

    @reactive.Effect
    def load_uploaded_log():
        uploaded = input.log_upload()
        if uploaded:
            try:
                df = pd.read_csv(uploaded[0]["datapath"])
                required_cols = {"timestamp", "filename", "dl_prediction", "dl_confidence", 
                 "mc_prediction", "mc_confidence", "comment", "agreement", "suggested"}
                if not required_cols.issubset(df.columns):
                    ui.notification_show("Invalid session log format", type="error")
                    return
                prediction_log.clear()
                prediction_log.extend(df.to_dict(orient="records"))
                ui.notification_show("Session log restored.", type="message", duration=3000)
            except Exception as e:
                ui.notification_show(f"Error loading session log: {e}", type="error")
            

app = App(app_ui, server)