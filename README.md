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


Open http://localhost:5000 in your browser.


## Notes

- The model uses **adaptive thresholding** — it works well with real phone camera photos
- Supports PNG, JPG, JPEG
- The app shows both the original and the 28×28 preprocessed image it fed to the model
- Confidence ring, full probability bars, and top-3 predictions are displayed
