from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import traceback
import cv2

app = Flask(__name__)
CORS(app)

# ── Nutrition knowledge base (per 100g) ───────────────────────────────────────
NUTRITION_DB = {
    "apple":       {"calories": 52,  "protein": 0.3, "carbs": 14.0, "fat": 0.2,  "density": 0.85},
    "banana":      {"calories": 89,  "protein": 1.1, "carbs": 23.0, "fat": 0.3,  "density": 0.96},
    "orange":      {"calories": 47,  "protein": 0.9, "carbs": 12.0, "fat": 0.1,  "density": 0.87},
    "mango":       {"calories": 60,  "protein": 0.8, "carbs": 15.0, "fat": 0.4,  "density": 0.90},
    "grapes":      {"calories": 69,  "protein": 0.7, "carbs": 18.0, "fat": 0.2,  "density": 1.06},
    "strawberry":  {"calories": 32,  "protein": 0.7, "carbs": 8.0,  "fat": 0.3,  "density": 0.58},
    "watermelon":  {"calories": 30,  "protein": 0.6, "carbs": 8.0,  "fat": 0.2,  "density": 0.96},
    "pineapple":   {"calories": 50,  "protein": 0.5, "carbs": 13.0, "fat": 0.1,  "density": 0.86},
    "kiwi":        {"calories": 61,  "protein": 1.1, "carbs": 15.0, "fat": 0.5,  "density": 1.10},
    "pear":        {"calories": 57,  "protein": 0.4, "carbs": 15.0, "fat": 0.1,  "density": 0.97},
    "peach":       {"calories": 39,  "protein": 0.9, "carbs": 10.0, "fat": 0.3,  "density": 0.89},
    "lemon":       {"calories": 29,  "protein": 1.1, "carbs": 9.0,  "fat": 0.3,  "density": 1.05},
    "coconut":     {"calories": 354, "protein": 3.3, "carbs": 15.0, "fat": 33.0, "density": 1.50},
    "blueberry":   {"calories": 57,  "protein": 0.7, "carbs": 14.0, "fat": 0.3,  "density": 0.62},
    "papaya":      {"calories": 43,  "protein": 0.5, "carbs": 11.0, "fat": 0.3,  "density": 0.91},
    "pomegranate": {"calories": 83,  "protein": 1.7, "carbs": 19.0, "fat": 1.2,  "density": 1.01},
    "guava":       {"calories": 68,  "protein": 2.6, "carbs": 14.0, "fat": 1.0,  "density": 1.00},
    "cherry":      {"calories": 50,  "protein": 1.0, "carbs": 12.0, "fat": 0.3,  "density": 1.02},
    "fig":         {"calories": 74,  "protein": 0.8, "carbs": 19.0, "fat": 0.3,  "density": 0.98},
    "lychee":      {"calories": 66,  "protein": 0.8, "carbs": 17.0, "fat": 0.4,  "density": 1.00},
    "dragonfruit": {"calories": 60,  "protein": 1.2, "carbs": 13.0, "fat": 0.4,  "density": 0.90},
    "jackfruit":   {"calories": 95,  "protein": 1.7, "carbs": 23.0, "fat": 0.6,  "density": 0.95},
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
    "dragonfruit": ["Antioxidants", "Vitamin C", "Iron", "Probiotic"],
    "jackfruit":   ["High Fiber", "Vitamin B6", "Natural Sweetener", "Potassium"],
}

FRUIT_EMOJIS = {
    "apple": "🍎", "banana": "🍌", "orange": "🍊", "mango": "🥭",
    "grapes": "🍇", "strawberry": "🍓", "watermelon": "🍉", "pineapple": "🍍",
    "kiwi": "🥝", "pear": "🍐", "peach": "🍑", "lemon": "🍋",
    "coconut": "🥥", "blueberry": "🫐", "papaya": "🧡", "pomegranate": "❤️",
    "guava": "💚", "cherry": "🍒", "fig": "🟣", "lychee": "🔴",
    "dragonfruit": "🐉", "jackfruit": "🟡",
}

FRUIT_DESCRIPTIONS = {
    "apple": "Apples are rich in fiber and vitamin C, supporting digestive health and immunity. They contain powerful antioxidants linked to reduced disease risk.",
    "banana": "Bananas are an excellent source of potassium and natural energy. They support heart health and provide quick fuel for workouts.",
    "orange": "Oranges are packed with vitamin C and immune-boosting compounds. They are hydrating and support skin health.",
    "mango": "Mangoes are tropical superfruits rich in vitamin A and digestive enzymes. They support eye health and immunity.",
    "grapes": "Grapes contain resveratrol, a powerful antioxidant linked to heart health. They provide natural sugars for quick energy.",
    "strawberry": "Strawberries are low in calories and high in vitamin C and antioxidants. They support skin health and reduce inflammation.",
    "watermelon": "Watermelon is 92% water, making it extremely hydrating. It contains lycopene, linked to reduced heart disease risk.",
    "pineapple": "Pineapples contain bromelain, a digestive enzyme with anti-inflammatory properties. They are rich in vitamin C and manganese.",
    "kiwi": "Kiwis are nutrient-dense fruits with high vitamin C and K content. They support digestive health and immune function.",
    "pear": "Pears are high in fiber and gentle on digestion. They support gut health and provide steady energy release.",
    "peach": "Peaches are rich in vitamin A and low in calories. They support skin health and provide hydration.",
    "lemon": "Lemons are powerful detoxifiers rich in vitamin C. They support immunity and have alkalizing effects on the body.",
    "coconut": "Coconuts are rich in healthy medium-chain triglycerides (MCTs). They provide sustained energy and are popular in keto diets.",
    "blueberry": "Blueberries are among the most antioxidant-rich foods. They support brain health, memory, and have a low glycemic index.",
    "papaya": "Papayas contain papain, a powerful digestive enzyme. They are rich in vitamin C and support skin and digestive health.",
    "pomegranate": "Pomegranates are packed with punicalagins, powerful antioxidants. They support heart health and reduce inflammation.",
    "guava": "Guavas have one of the highest vitamin C contents of any fruit. They are high in fiber and support immune function.",
    "cherry": "Cherries contain melatonin and anti-inflammatory compounds. They may support sleep quality and reduce muscle soreness.",
    "fig": "Figs are high in fiber and natural sugars. They provide calcium and iron, supporting bone health and energy.",
    "lychee": "Lychees are rich in vitamin C and copper. They provide antioxidants and hydration.",
    "dragonfruit": "Dragon fruit is rich in antioxidants and prebiotics. It supports gut health and provides iron and vitamin C.",
    "jackfruit": "Jackfruit is high in fiber and B vitamins. It provides sustained energy and supports digestive health.",
}

# ── Colour-based fruit detection using HSV ranges ─────────────────────────────
# Each fruit has dominant HSV colour ranges
FRUIT_COLOR_PROFILES = {
    "apple":       [((0,50,50),(10,255,255)), ((160,50,50),(180,255,255))],  # red
    "banana":      [((20,80,100),(35,255,255))],   # yellow
    "orange":      [((10,80,80),(25,255,255))],    # orange
    "mango":       [((15,80,80),(35,255,255))],    # yellow-orange
    "grapes":      [((120,30,30),(160,255,180))],  # purple
    "strawberry":  [((0,80,80),(10,255,255)),((160,80,80),(180,255,255))],  # red
    "watermelon":  [((0,100,50),(10,255,255))],    # red flesh
    "pineapple":   [((20,60,100),(35,255,255))],   # yellow
    "kiwi":        [((30,40,40),(70,255,200))],    # green-brown
    "pear":        [((25,30,80),(80,180,255))],    # yellow-green
    "peach":       [((5,60,150),(20,200,255))],    # peach/pink
    "lemon":       [((25,100,150),(35,255,255))],  # bright yellow
    "blueberry":   [((100,50,30),(140,255,150))],  # blue-purple
    "papaya":      [((10,80,150),(25,255,255))],   # orange
    "pomegranate": [((0,80,80),(10,255,200))],     # dark red
    "cherry":      [((0,80,50),(10,255,180)),((160,80,50),(180,255,180))],  # dark red
    "lychee":      [((0,30,180),(15,120,255))],    # pinkish white
    "guava":       [((25,20,150),(80,100,255))],   # light green/yellow
    "coconut":     [((15,20,80),(35,80,200))],     # brown
    "fig":         [((120,20,30),(160,180,150))],  # purple-brown
    "dragonfruit": [((150,80,100),(180,255,255)),((0,80,100),(10,255,255))],  # pink/red
    "jackfruit":   [((20,50,80),(35,200,255))],    # yellow-green
}

def decode_image(data_url_or_b64):
    if "," in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(",")[1]
    img_bytes = base64.b64decode(data_url_or_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return image

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ── Step 1: Colour-based fruit detection ─────────────────────────────────────
def detect_fruit_by_color(image):
    img_cv = pil_to_cv2(image)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    best_fruit = None
    best_score = 0
    best_mask = None

    for fruit, ranges in FRUIT_COLOR_PROFILES.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        score = np.sum(combined_mask > 0)
        if score > best_score:
            best_score = score
            best_fruit = fruit
            best_mask = combined_mask

    total_pixels = hsv.shape[0] * hsv.shape[1]
    confidence = min(0.99, best_score / (total_pixels * 0.3))
    return best_fruit, best_mask, confidence

# ── Step 2: Contour-based segmentation ───────────────────────────────────────
def segment_fruit(image, color_mask):
    img_cv = pil_to_cv2(image)
    # Clean up mask
    kernel = np.ones((15, 15), np.uint8)
    cleaned = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    # Find largest contour
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # fallback: use full image center region
        h, w = img_cv.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (w//3, h//3), 0, 0, 360, 255, -1)
        return mask
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

# ── Step 3: Estimate depth from image brightness/gradient ────────────────────
def estimate_depth(image, mask):
    img_gray = np.array(image.convert("L"), dtype=np.float32)
    # Use Laplacian as proxy for depth detail
    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    lap = np.abs(lap)
    # Normalize
    lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
    # Brightest regions = closer to camera
    brightness = img_gray / 255.0
    depth_proxy = (brightness * 0.6 + lap * 0.4)
    return depth_proxy

# ── Step 4: Volume estimation ─────────────────────────────────────────────────
def estimate_volume(mask, depth_map, image_size):
    w, h = image_size
    fruit_pixels = np.sum(mask > 127)
    total_pixels = w * h
    fruit_ratio = fruit_pixels / total_pixels

    # Average depth in fruit region
    fruit_depth_values = depth_map[mask > 127]
    if len(fruit_depth_values) == 0:
        avg_depth = 0.5
    else:
        avg_depth = float(np.mean(fruit_depth_values))

    # Estimate real-world size assuming standard photo distance (~50cm)
    # 1 pixel ≈ 0.05cm at 50cm distance with typical phone camera
    PIXEL_TO_CM = 0.05
    pixel_area_cm2 = PIXEL_TO_CM ** 2
    projected_area_cm2 = fruit_pixels * pixel_area_cm2

    # Estimate depth extent from avg_depth (0-1 → 2-20cm)
    depth_extent_cm = 2 + avg_depth * 18

    # Volume of ellipsoid approximation: V = (4/3)π * a * b * c
    # where a,b = sqrt of projected area / π, c = depth/2
    import math
    if projected_area_cm2 > 0:
        ab = math.sqrt(projected_area_cm2 / math.pi)
        c = depth_extent_cm / 2
        volume_cm3 = (4/3) * math.pi * ab * ab * c
    else:
        volume_cm3 = 100  # fallback

    # Clamp to realistic range
    volume_cm3 = max(15, min(volume_cm3, 3000))
    return round(volume_cm3, 2), round(projected_area_cm2, 2)

# ── Step 5: Mass and calorie calculation ─────────────────────────────────────
def calculate_nutrition(fruit_key, volume_cm3):
    info = NUTRITION_DB.get(fruit_key, NUTRITION_DB["apple"])
    mass_g = volume_cm3 * info["density"]
    factor = mass_g / 100
    return {
        "mass_g":   round(mass_g, 1),
        "calories": round(info["calories"] * factor, 1),
        "protein":  round(info["protein"] * factor, 2),
        "carbs":    round(info["carbs"] * factor, 2),
        "fat":      round(info["fat"] * factor, 2),
        "calories_per_100g": info["calories"],
    }

# ── API Routes ────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "CalorieBit Lightweight API is running"})

@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image = decode_image(data["image"])

        # Step 1: Detect fruit by colour
        fruit_key, color_mask, confidence = detect_fruit_by_color(image)
        if fruit_key is None:
            fruit_key = "apple"
            confidence = 0.5

        # Step 2: Segment fruit
        seg_mask = segment_fruit(image, color_mask)

        # Step 3: Estimate depth
        depth_map = estimate_depth(image, seg_mask)

        # Step 4: Estimate volume
        volume_cm3, area_cm2 = estimate_volume(seg_mask, depth_map, image.size)

        # Step 5: Calculate nutrition
        nutrition = calculate_nutrition(fruit_key, volume_cm3)

        result = {
            "fruit":            fruit_key.capitalize(),
            "emoji":            FRUIT_EMOJIS.get(fruit_key, "🍎"),
            "confidence":       f"Detected with {int(confidence * 100)}% confidence",
            "volume_cm3":       volume_cm3,
            "area_cm2":         area_cm2,
            "mass_g":           nutrition["mass_g"],
            "calories":         nutrition["calories"],
            "calories_per_100g":nutrition["calories_per_100g"],
            "protein":          f"{nutrition['protein']}g",
            "carbs":            f"{nutrition['carbs']}g",
            "fat":              f"{nutrition['fat']}g",
            "portion":          f"~{nutrition['mass_g']}g estimated portion",
            "health_tags":      HEALTH_TAGS.get(fruit_key, ["Natural", "Healthy", "Fresh"]),
            "description":      FRUIT_DESCRIPTIONS.get(fruit_key, f"{fruit_key.capitalize()} is a nutritious fruit."),
            "pipeline": {
                "step1_detection":   fruit_key,
                "step2_segmentation": f"{int(np.sum(seg_mask > 127))} pixels",
                "step3_depth":       "brightness+gradient proxy",
                "step4_volume":      f"{volume_cm3} cm³",
                "step5_mass":        f"{nutrition['mass_g']}g",
            }
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
