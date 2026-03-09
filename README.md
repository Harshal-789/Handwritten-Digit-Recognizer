# Digit Intelligence — MNIST Web App

A dark, modern web interface for your trained MNIST CNN model.

## Project Structure

```
digit-app/
├── app.py                  # Flask backend (prediction API)
├── templates/
│   └── index.html          # Frontend UI (dark, modern)
├── mnist_final_model.h5    # ← PLACE YOUR MODEL HERE
├── requirements.txt
├── Procfile                # For Render / Railway / Heroku
└── README.md
```

---

## 1. Run Locally

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Place your model in the app folder
cp path/to/mnist_final_model.h5 digit-app/

# 2. Install dependencies
cd digit-app
pip install -r requirements.txt

# 3. Run
python app.py
```

Open http://localhost:5000 in your browser.

---

## 2. Deploy to Render (Free)

1. Push the `digit-app/` folder to a GitHub repo
2. Include `mnist_final_model.h5` in the repo
3. Go to https://render.com → New → Web Service
4. Connect your GitHub repo
5. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Instance Type:** Free (or Starter for faster cold starts)
6. Deploy → get a live URL!

---

## 3. Deploy to Railway

1. Push to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add env variable if needed: `MODEL_PATH=mnist_final_model.h5`
4. Railway auto-detects Procfile and deploys

---

## 4. Deploy to Hugging Face Spaces (Recommended for ML)

1. Create a new Space at https://huggingface.co/spaces
2. Choose **Gradio** or **Static** — select **Docker**
3. Upload all files including the model
4. The app will build and deploy automatically

---

## API Endpoint

### POST `/predict`
- **Body:** `multipart/form-data` with field `file` (PNG/JPG)
- **Response:**
```json
{
  "predicted_digit": 7,
  "confidence": 99.87,
  "top3": [
    {"digit": 7, "confidence": 99.87},
    {"digit": 1, "confidence": 0.08},
    {"digit": 9, "confidence": 0.03}
  ],
  "all_probabilities": [0.01, 0.08, 0.01, ...],
  "processed_image": "<base64 PNG string>"
}
```

---

## Notes

- The model uses **adaptive thresholding** — it works well with real phone camera photos
- Supports PNG, JPG, JPEG
- The app shows both the original and the 28×28 preprocessed image it fed to the model
- Confidence ring, full probability bars, and top-3 predictions are displayed
