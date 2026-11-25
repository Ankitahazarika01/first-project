import json
import os
import cv2
import numpy as np

# Use lightweight tflite_runtime on edge devices (Raspberry Pi) or full TensorFlow on laptops
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# 1. Local Offline Knowledge Base (Disease/Pest -> Symptoms & Treatment Advisory)
OFFLINE_ADVISORY_DB = {
    "Potato___Early_blight": {
        "crop": "Potato",
        "condition": "Early Blight (Alternaria solani)",
        "severity": "Moderate",
        "organic_remedy": "Remove lower infected leaves; spray neem oil (3 ml/litre of water) every 7 days.",
        "chemical_remedy": "Apply Mancozeb 75% WP (2 g/litre) or Chlorothalonil as per local agriculture guidelines.",
        "prevention": "Use certified disease-free tubers, practice crop rotation, and avoid overhead irrigation."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "condition": "Late Blight (Phytophthora infestans)",
        "severity": "High (Urgent Action Required)",
        "organic_remedy": "Destroy severely infected plants immediately; apply copper-based bio-fungicide.",
        "chemical_remedy": "Spray Cymoxanil 8% + Mancozeb 64% WP (2.5 g/litre) before rain or high humidity.",
        "prevention": "Ensure proper field drainage and avoid excessive nitrogen fertilizer."
    },
    "Rice___Leaf_blast": {
        "crop": "Rice (Paddy)",
        "condition": "Rice Leaf Blast (Magnaporthe oryzae)",
        "severity": "High",
        "organic_remedy": "Spray Pseudomonas fluorescens (10 g/litre) at early symptom appearance.",
        "chemical_remedy": "Apply Tricyclazole 75% WP (0.6 g/litre) or Isoprothiolane 40% EC.",
        "prevention": "Avoid excess nitrogen top-dressing and maintain proper plant spacing."
    },
    "Tomato___Leaf_Miner_Pest": {
        "crop": "Tomato",
        "condition": "Serpentine Leaf Miner Pest (Liriomyza trifolii)",
        "severity": "Moderate",
        "organic_remedy": "Install yellow sticky traps (15/acre) and spray Azadirachtin 1% (2 ml/litre).",
        "chemical_remedy": "Use Spinosad 45% SC (0.3 ml/litre) if infestation exceeds 20% leaves.",
        "prevention": "Remove and destroy mined leaves early; avoid broad-spectrum insecticides that kill natural predators."
    },
    "Healthy_Crop": {
        "crop": "General Crop",
        "condition": "Healthy Leaf (No Disease or Pest Detected)",
        "severity": "None",
        "organic_remedy": "No treatment needed. Continue regular monitoring.",
        "chemical_remedy": "None required.",
        "prevention": "Maintain balanced NPK nutrition and regular field scouting."
    }
}

CLASS_NAMES = list(OFFLINE_ADVISORY_DB.keys())


class OfflineCropDoctor:
    def __init__(self, model_path="crop_disease_mobilenetv2.tflite", img_size=(224, 224)):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Offline model '{model_path}' not found. Run 'train_model.py' first to generate it."
            )
        self.img_size = img_size
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image_bgr):
        """Resize, convert BGR->RGB, and normalize image for MobileNetV2."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size)
        img_normalized = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
        return img_normalized

    def predict(self, image_bgr):
        """Run offline TFLite inference and fetch local advisory."""
        input_tensor = self.preprocess_image(image_bgr)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        predictions = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx]) * 100.0

        label = CLASS_NAMES[top_idx]
        advisory = OFFLINE_ADVISORY_DB.get(label, {})

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "advisory": advisory
        }


def run_webcam_scanner(model_path="crop_disease_mobilenetv2.tflite"):
    """Capture leaf image from webcam and diagnose offline."""
    doctor = OfflineCropDoctor(model_path=model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("=== OFFLINE CROP DISEASE & PEST SCANNER ===")
    print("Press [SPACE] to capture and diagnose leaf | Press [Q] to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw center target box for leaf placement
        h, w, _ = frame.shape
        box_size = min(h, w) // 2
        x1, y1 = (w - box_size) // 2, (h - box_size) // 2
        x2, y2 = x1 + box_size, y1 + box_size

        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, "Place leaf inside box & press SPACE", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Offline Crop Disease & Pest Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            leaf_crop = frame[y1:y2, x1:x2]
            result = doctor.predict(leaf_crop)
            adv = result["advisory"]

            print("\n" + "=" * 50)
            print(f"DIAGNOSIS   : {adv.get('condition')} ({result['confidence']}%)")
            print(f"CROP        : {adv.get('crop')}")
            print(f"SEVERITY    : {adv.get('severity')}")
            print(f"ORGANIC     : {adv.get('organic_remedy')}")
            print(f"CHEMICAL    : {adv.get('chemical_remedy')}")
            print(f"PREVENTION  : {adv.get('prevention')}")
            print("=" * 50 + "\n")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example 1: Diagnose from a saved leaf photo offline
    # doctor = OfflineCropDoctor("crop_disease_mobilenetv2.tflite")
    # img = cv2.imread("sample_leaf.jpg")
    # print(json.dumps(doctor.predict(img), indent=2))

    # Example 2: Run live webcam scanner offline
    run_webcam_scanner()
