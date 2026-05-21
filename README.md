# 🎨 NeuralArt — Neural Style Transfer Web App

> Reimagine any photo as a work of art using AI-powered style transfer.

---

## What It Does

NeuralArt lets you upload any photo and apply the visual style of another image to it — turning a portrait into a pencil sketch, a cityscape into an oil painting, or any photo into your chosen artistic style.

It uses **AdaIN (Adaptive Instance Normalization)**, a fast neural style transfer technique that separates content from style and blends them in real time using a pre-trained VGG encoder and custom-trained decoder.

---

## Demo

| Content Image | Style Reference | Result |
|---|---|---|
| Photo of a face | Pencil sketch | Sketch-style portrait |
| Landscape photo | Van Gogh painting | Painterly landscape |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Bootstrap |
| Backend | Python, Flask, Flask-WTF |
| Deep Learning | PyTorch, TorchVision |
| Model Architecture | VGG Encoder + Custom AdaIN Decoder |
| Image Processing | Pillow |
| Deployment | Render (Gunicorn) |

---

## How It Works

1. User uploads a **content image** (the photo to stylize)
2. User uploads a **style image** (the artistic reference)
3. Both images are passed through a **VGG encoder** to extract feature maps
4. **AdaIN** aligns the mean and variance of content features to match style features
5. A **custom decoder** reconstructs the stylized image from the adapted features
6. The result is displayed and available for download

The **style strength slider (alpha)** controls the blend — 0.0 keeps the original content, 1.0 applies full style transfer.

---

## Project Structure

```
Neural-style-transfer/
├── app.py                  # Flask application (routes, form handling, inference)
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
├── Procfile                # Render/Heroku deployment config
├── vgg_normalised.pth      # Pre-trained VGG encoder weights
├── decoder_final.pth       # Trained AdaIN decoder weights
├── utils/
│   ├── models.py           # VGGEncoder and Decoder architecture
│   └── utils.py            # AdaIN helper functions
├── templates/
│   └── index.html          # Main UI template
├── static/uploads/         # Uploaded and generated images
├── content_data/           # Sample content images for training
├── style_data/             # Sample style images for training
└── examples/               # Example input/output pairs
```

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/AshishShetty1854/Neural-style-transfer.git
cd Neural-style-transfer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py
```

**4. Open in browser**
```
http://localhost:5000
```

---

## Deployment (Render)

This app is deployed on [Render](https://render.com). Key configuration:

**Start Command:**
```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1
```

- `--timeout 300` — gives style transfer enough time to complete on CPU
- `--workers 1` — prevents out-of-memory errors from loading the model multiple times
- `--bind 0.0.0.0:$PORT` — binds to Render's dynamically assigned port

**Upload folder** is set to `/tmp/uploads` since Render's filesystem is ephemeral.

---

## Model Details

| Component | Details |
|---|---|
| Encoder | VGG-19 (normalised), frozen, up to relu4_1 |
| Decoder | Mirror of encoder, trained from scratch |
| Method | AdaIN — aligns feature statistics (mean + std) |
| Training Data | MS-COCO (content) + WikiArt (style) |
| Loss | Content loss (relu4_1) + Style loss (relu1_1 to relu4_1) |
| Input Size | Resized to 256×256 during inference |

---

## Style Strength (Alpha)

The alpha parameter controls the interpolation between content and stylized features:

```
alpha = 0.0  →  Original content image (no style applied)
alpha = 0.5  →  Blend of content and style
alpha = 1.0  →  Full style transfer (default)
```

---

## Requirements

```
Flask
Flask-WTF
Flask-Bootstrap
torch (CPU)
torchvision
Pillow
gunicorn
WTForms
```

Full list in `requirements.txt`.

---

## Known Limitations

- Runs on **CPU only** on Render free tier — inference takes 30–90 seconds per image
- Uploaded images are stored in `/tmp` and cleared on server restart
- Large images (>1024px) may slow down inference significantly

---

## Author

**Ashish Shetty**
[GitHub @AshishShetty1854](https://github.com/AshishShetty1854)

---

## License

This project is open source and available under the [MIT License](LICENSE).
