# CalorieBit Backend API

Transformer-based fruit calorie estimation pipeline using:
- **DETR** (facebook/detr-resnet-50) — Object Detection
- **SAM** (facebook/sam-vit-base) — Instance Segmentation  
- **DPT** (Intel/dpt-large) — Depth Estimation
- **Volume formula** — Mass & Calorie Calculation

## Deploy to Render.com (Free)

1. Push this folder to a GitHub repository
2. Go to [render.com](https://render.com) and sign up free
3. Click **New** → **Web Service**
4. Connect your GitHub repo
5. Render auto-detects settings from `render.yaml`
6. Click **Deploy** — takes ~5-10 minutes
7. Your API URL will be: `https://caloriebit-api.onrender.com`

## API Endpoints

### GET /health
Check if API is running.

### POST /analyse
Send a fruit image, get calorie analysis back.

**Request:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "fruit": "Apple",
  "emoji": "🍎",
  "confidence": "Detected with 95% confidence",
  "volume_cm3": 145.2,
  "mass_g": 123.4,
  "calories": 64.2,
  "calories_per_100g": 52,
  "protein": "0.37g",
  "carbs": "17.27g",
  "fat": "0.25g",
  "portion": "~123.4g estimated portion",
  "health_tags": ["High Fiber", "Vitamin C", "Antioxidants", "Low Calorie"],
  "description": "Apple is a nutritious fruit...",
  "pipeline": {
    "detection": "apple",
    "box": [x1, y1, x2, y2],
    "volume_cm3": 145.2,
    "mass_g": 123.4
  }
}
```
