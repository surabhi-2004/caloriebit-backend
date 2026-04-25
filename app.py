from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
import traceback
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    SamModel,
    SamProcessor,
    DPTForDepthEstimation,
    DPTImageProcessor,
)

app = Flask(__name__)
CORS(app)

# ── Nutrition knowledge base (per 100g) ──────────────────────────────────────
NUTRITION_DB = {
    "apple":      {"calories": 52,  "protein": 0.3, "carbs": 14.0, "fat": 0.2, "density": 0.85},
    "banana":     {"calories": 89,  "protein": 1.1, "carbs": 23.0, "fat": 0.3, "density": 0.96},
    "orange":     {"calories": 47,  "protein": 0.9, "carbs": 12.0, "fat": 0.1, "density": 0.87},
    "mango":      {"calories": 60,  "protein": 0.8, "carbs": 15.0, "fat": 0.4, "density": 0.90},
    "grapes":     {"calories": 69,  "protein": 0.7, "carbs": 18.0, "fat": 0.2, "density": 1.06},
    "strawberry": {"calories": 32,  "protein": 0.7, "carbs": 8.0,  "fat": 0.3, "density": 0.58},
    "watermelon": {"calories": 30,  "protein": 0.6, "carbs": 8.0,  "fat": 0.2, "density": 0.96},
    "pineapple":  {"calories": 50,  "protein": 0.5, "carbs": 13.0, "fat": 0.1, "density": 0.86},
    "kiwi":       {"calories": 61,  "protein": 1.1, "carbs": 15.0, "fat": 0.5, "density": 1.10},
    "pear":       {"calories": 57,  "protein": 0.4, "carbs": 15.0, "fat": 0.1, "density": 0.97},
    "peach":      {"calories": 39,  "protein": 0.9, "carbs": 10.0, "fat": 0.3, "density": 0.89},
    "lemon":      {"calories": 29,  "protein": 1.1, "carbs": 9.0,  "fat": 0.3, "density": 1.05},
    "coconut":    {"calories": 354, "protein": 3.3, "carbs": 15.0, "fat": 33.0,"density": 1.50},
    "blueberry":  {"calories": 57,  "protein": 0.7, "carbs": 14.0, "fat": 0.3, "density": 0.62},
    "papaya":     {"calories": 43,  "protein": 0.5, "carbs": 11.0, "fat": 0.3, "density": 0.91},
    "pomegranate":{"calories": 83,  "protein": 1.7, "carbs": 19.0, "fat": 1.2, "density": 1.01},
    "guava":      {"calories": 68,  "protein": 2.6, "carbs": 14.0, "fat": 1.0, "density": 1.00},
    "cherry":     {"calories": 50,  "protein": 1.0, "carbs": 12.0, "fat": 0.3, "density": 1.02},
    "fig":        {"calories": 74,  "protein": 0.8, "carbs": 19.0, "fat": 0.3, "density": 0.98},
    "lychee":     {"calories": 66,  "protein": 0.8, "carbs": 17.0, "fat": 0.4, "density": 1.00},
}

HEALTH_TAGS = {
    "apple":       ["High Fiber", "Vitamin C", "Antioxidants", "Low Calorie"],
    "banana":      ["High Potassium", "Natural Energy", "Vitamin B6", "Pre-workout"],
    "orange":      ["Vitamin C", "Immune Boost", "Low Fat", "Hydrating"],
    "mango":       ["Vitamin A", "Digestive Aid", "Tropical", "Rich in Folate"],
    "grapes":      ["Resveratrol", "Antioxidants", "Heart Health", "Natural Sugars"],
    "strawberry":  ["Vitamin C", "Low Calorie", "Antioxidants", "Skin Health"],
    "watermelon":  ["Hydrating", "Lycopene", "Low Calorie", "Summer Fruit"],
    "pineapple":   ["Bromelain", "Digestive Aid", "Vitamin C", "Anti-inflammatory"],
    "kiwi":        ["Vitamin C", "Vitamin K", "Digestive Health", "Immune Boost"],
    "pear":        ["High Fiber", "Low Calorie", "Vitamin C", "Gut Health"],
    "peach":       ["Vitamin A", "Low Calorie", "Skin Health", "Hydrating"],
    "lemon":       ["Vitamin C", "Detox", "Alkalizing", "Immune Boost"],
    "coconut":     ["Healthy Fats", "MCT Oil", "Electrolytes", "Keto Friendly"],
    "blueberry":   ["Antioxidants", "Brain Health", "Low GI", "Vitamin K"],
    "papaya":      ["Digestive Aid", "Vitamin C", "Papain Enzyme", "Skin Health"],
    "pomegranate": ["Antioxidants", "Heart Health", "Anti-inflammatory", "Vitamin C"],
    "guava":       ["Vitamin C", "High Fiber", "Immune Boost", "Low GI"],
    "cherry":      ["Melatonin", "Anti-inflammatory", "Antioxidants", "Sleep Aid"],
    "fig":         ["High Fiber", "Calcium", "Natural Sweetener", "Iron"],
    "lychee":      ["Vitamin C", "Copper", "Antioxidants", "Hydrating"],
}

FRUIT_EMOJIS = {
    "apple": "🍎", "banana": "🍌", "orange": "🍊", "mango": "🥭",
    "grapes": "🍇", "strawberry": "🍓", "watermelon": "🍉", "pineapple": "🍍",
    "kiwi": "🥝", "pear": "🍐", "peach": "🍑", "lemon": "🍋",
    "coconut": "🥥", "blueberry": "🫐", "papaya": "🧡", "pomegranate": "❤️",
    "guava": "💚", "cherry": "🍒", "fig": "🟣", "lychee": "🔴",
}

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading DETR model...")
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_model.eval()

print("Loading SAM model...")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_model.eval()

print("Loading DPT model...")
dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
dpt_model.eval()

print("All models loaded!")

# COCO fruit labels that DETR knows
FRUIT_LABELS = {
    "apple", "banana", "orange", "broccoli", "carrot",
    "sandwich", "pizza", "cake", "donut"
}
COCO_FRUITS = {"apple", "banana", "orange"}

def match_nutrition(label):
    label = label.lower()
    for key in NUTRITION_DB:
        if key in label or label in key:
            return key
    return None

def decode_image(data_url_or_b64):
    if "," in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(",")[1]
    img_bytes = base64.b64decode(data_url_or_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return image

# ── Pipeline ──────────────────────────────────────────────────────────────────

def step1_detect(image):
    """DETR object detection — find fruit bounding box."""
    inputs = detr_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = detr_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = detr_model.config.id2label[label.item()].lower()
        detections.append({
            "label": label_name,
            "score": round(score.item(), 3),
            "box": [round(x) for x in box.tolist()]
        })
    # prefer fruit detections
    fruit_detections = [d for d in detections if any(f in d["label"] for f in COCO_FRUITS)]
    if fruit_detections:
        return sorted(fruit_detections, key=lambda x: x["score"], reverse=True)[0]
    if detections:
        return sorted(detections, key=lambda x: x["score"], reverse=True)[0]
    # fallback: whole image
    w, h = image.size
    return {"label": "fruit", "score": 0.5, "box": [0, 0, w, h]}

def step2_segment(image, box):
    """SAM segmentation — get precise fruit mask."""
    inputs = sam_processor(
        images=image,
        input_boxes=[[box]],
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )
    mask = masks[0][0][0].numpy()
    return mask

def step3_depth(image):
    """DPT depth estimation — get per-pixel depth map."""
    inputs = dpt_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = dpt_model(**inputs)
        predicted_depth = outputs.predicted_depth
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().numpy()
    # normalize to 0-1
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

def step4_volume(mask, depth):
    """Combine mask + depth to estimate volume (in arbitrary units)."""
    fruit_depth = depth[mask > 0.5]
    if len(fruit_depth) == 0:
        return 0
    pixel_count = np.sum(mask > 0.5)
    avg_depth = np.mean(fruit_depth)
    # Volume ≈ Σ(depth × pixel_area)
    # We use a scaling factor to convert to cm³
    PIXEL_TO_CM = 0.05  # approximate: 1 pixel ≈ 0.05 cm at typical photo distance
    pixel_area_cm2 = PIXEL_TO_CM ** 2
    volume_cm3 = float(np.sum(fruit_depth[fruit_depth > 0.3]) * pixel_area_cm2 * 10)
    # clamp to realistic range (20–2000 cm³)
    volume_cm3 = max(20, min(volume_cm3, 2000))
    return round(volume_cm3, 2)

def step5_calories(fruit_key, volume_cm3):
    """Convert volume → mass → calories."""
    nutrition = NUTRITION_DB.get(fruit_key, NUTRITION_DB["apple"])
    density = nutrition["density"]  # g/cm³
    mass_g = volume_cm3 * density
    calories = (nutrition["calories"] * mass_g) / 100
    protein = (nutrition["protein"] * mass_g) / 100
    carbs = (nutrition["carbs"] * mass_g) / 100
    fat = (nutrition["fat"] * mass_g) / 100
    return {
        "mass_g": round(mass_g, 1),
        "calories": round(calories, 1),
        "protein": round(protein, 2),
        "carbs": round(carbs, 2),
        "fat": round(fat, 2),
    }

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "CalorieBit API is running"})

@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode image
        image = decode_image(data["image"])

        # Step 1: Detect
        detection = step1_detect(image)
        detected_label = detection["label"]
        box = detection["box"]

        # Step 2: Segment
        mask = step2_segment(image, box)

        # Step 3: Depth
        depth = step3_depth(image)

        # Step 4: Volume
        volume_cm3 = step4_volume(mask, depth)

        # Match to nutrition DB
        fruit_key = match_nutrition(detected_label) or "apple"

        # Step 5: Calories
        nutrition = step5_calories(fruit_key, volume_cm3)
        base_nutrition = NUTRITION_DB[fruit_key]

        result = {
            "fruit": fruit_key.capitalize(),
            "emoji": FRUIT_EMOJIS.get(fruit_key, "🍎"),
            "confidence": f"Detected with {int(detection['score']*100)}% confidence",
            "volume_cm3": volume_cm3,
            "mass_g": nutrition["mass_g"],
            "calories": nutrition["calories"],
            "calories_per_100g": base_nutrition["calories"],
            "protein": f"{nutrition['protein']}g",
            "carbs": f"{nutrition['carbs']}g",
            "fat": f"{nutrition['fat']}g",
            "portion": f"~{nutrition['mass_g']}g estimated portion",
            "health_tags": HEALTH_TAGS.get(fruit_key, ["Natural", "Healthy", "Fresh"]),
            "description": f"{fruit_key.capitalize()} is a nutritious fruit. "
                           f"Estimated volume: {volume_cm3} cm³, mass: {nutrition['mass_g']}g.",
            "pipeline": {
                "detection": detected_label,
                "box": box,
                "volume_cm3": volume_cm3,
                "mass_g": nutrition["mass_g"],
            }
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
