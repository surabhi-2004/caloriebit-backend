from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import traceback
import cv2
import urllib.request
import json
import os
import math

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCq8U0DqrsiWhcFLn-VgE_CMbHqI3U43So")

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
    "apple": "Apples are rich in fiber and vitamin C, supporting digestive health and immunity.",
    "banana": "Bananas are an excellent source of potassium and natural energy.",
    "orange": "Oranges are packed with vitamin C and immune-boosting compounds.",
    "mango": "Mangoes are tropical superfruits rich in vitamin A and digestive enzymes.",
    "grapes": "Grapes contain resveratrol, a powerful antioxidant linked to heart health.",
    "strawberry": "Strawberries are low in calories and high in vitamin C and antioxidants.",
    "watermelon": "Watermelon is 92% water, making it extremely hydrating. Contains lycopene.",
    "pineapple": "Pineapples contain bromelain, a digestive enzyme with anti-inflammatory properties.",
    "kiwi": "Kiwis are nutrient-dense fruits with high vitamin C and K content.",
    "pear": "Pears are high in fiber and gentle on digestion.",
    "peach": "Peaches are rich in vitamin A and low in calories.",
    "lemon": "Lemons are powerful detoxifiers rich in vitamin C.",
    "coconut": "Coconuts are rich in healthy medium-chain triglycerides (MCTs).",
    "blueberry": "Blueberries are among the most antioxidant-rich foods available.",
    "papaya": "Papayas contain papain, a powerful digestive enzyme.",
    "pomegranate": "Pomegranates are packed with punicalagins, powerful antioxidants.",
    "guava": "Guavas have one of the highest vitamin C contents of any fruit.",
    "cherry": "Cherries contain melatonin and anti-inflammatory compounds.",
    "fig": "Figs are high in fiber and natural sugars providing calcium and iron.",
    "lychee": "Lychees are rich in vitamin C and copper providing antioxidants.",
    "dragonfruit": "Dragon fruit is rich in antioxidants and prebiotics supporting gut health.",
    "jackfruit": "Jackfruit is high in fiber and B vitamins providing sustained energy.",
}

def decode_image(data_url_or_b64):
    if "," in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(",")[1]
    missing_padding = len(data_url_or_b64) % 4
    if missing_padding:
        data_url_or_b64 += '=' * (4 - missing_padding)
    img_bytes = base64.b64decode(data_url_or_b64)
    image = Image.open(io.BytesIO(img_bytes))
    image = image.convert("RGB")
    return image

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def identify_fruit_gemini(image_b64, mime_type="image/jpeg"):
    prompt = 'Look at this image and identify the fruit. Reply with ONLY a JSON object like this: {"fruit": "apple", "confidence": 0.95}. Use lowercase fruit name. If not a fruit, use "unknown".'
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            {"text": prompt}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 100}
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    text = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    return data.get("fruit", "unknown").lower(), float(data.get("confidence", 0.8))

def segment_fruit(image):
    img_cv = pil_to_cv2(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((20, 20), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    else:
        h, w = img_cv.shape[:2]
        cv2.ellipse(mask, (w//2, h//2), (w//3, h//3), 0, 0, 360, 255, -1)
    return mask

def estimate_volume(mask, image_size):
    w, h = image_size
    fruit_pixels = int(np.sum(mask > 127))
    total_pixels = w * h
    fruit_ratio = fruit_pixels / total_pixels
    if fruit_ratio < 0.05:
        diameter_cm = 3.0
    elif fruit_ratio < 0.15:
        diameter_cm = 5.0
    elif fruit_ratio < 0.30:
        diameter_cm = 6.5
    elif fruit_ratio < 0.50:
        diameter_cm = 7.5
    else:
        diameter_cm = 8.5
    radius_cm = diameter_cm / 2.0
    volume_cm3 = (4.0/3.0) * math.pi * (radius_cm ** 3)
    volume_cm3 = max(15.0, min(volume_cm3, 1500.0))
    projected_area_cm2 = math.pi * (radius_cm ** 2)
    return round(volume_cm3, 2), round(projected_area_cm2, 2), round(fruit_ratio * 100, 1)

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

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "CalorieBit API running"})

@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        raw_b64 = data["image"]
        mime_type = "image/jpeg"
        if "," in raw_b64:
            header = raw_b64.split(",")[0]
            if "png" in header:
                mime_type = "image/png"
            elif "webp" in header:
                mime_type = "image/webp"

        image = decode_image(raw_b64)
        clean_b64 = raw_b64.split(",")[1] if "," in raw_b64 else raw_b64

        try:
            fruit_key, confidence = identify_fruit_gemini(clean_b64, mime_type)
        except Exception as e:
            print(f"Gemini error: {e}")
            fruit_key = "apple"
            confidence = 0.5

        if fruit_key not in NUTRITION_DB:
            fruit_key = "apple"

        mask = segment_fruit(image)
        volume_cm3, area_cm2, fruit_pct = estimate_volume(mask, image.size)
        nutrition = calculate_nutrition(fruit_key, volume_cm3)

        result = {
            "fruit":             fruit_key.capitalize(),
            "emoji":             FRUIT_EMOJIS.get(fruit_key, "🍎"),
            "confidence":        f"Detected with {int(confidence * 100)}% confidence",
            "volume_cm3":        volume_cm3,
            "area_cm2":          area_cm2,
            "mass_g":            nutrition["mass_g"],
            "calories":          nutrition["calories"],
            "calories_per_100g": nutrition["calories_per_100g"],
            "protein":           f"{nutrition['protein']}g",
            "carbs":             f"{nutrition['carbs']}g",
            "fat":               f"{nutrition['fat']}g",
            "portion":           f"~{nutrition['mass_g']}g estimated portion",
            "health_tags":       HEALTH_TAGS.get(fruit_key, ["Natural", "Healthy", "Fresh"]),
            "description":       FRUIT_DESCRIPTIONS.get(fruit_key, f"{fruit_key.capitalize()} is a nutritious fruit."),
            "pipeline": {
                "step1_gemini_id": fruit_key,
                "step2_fruit_pct": f"{fruit_pct}% of image",
                "step3_volume":    f"{volume_cm3} cm3",
                "step4_mass":      f"{nutrition['mass_g']}g",
            }
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
