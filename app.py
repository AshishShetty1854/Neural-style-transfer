import os
import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Import AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# =========================================================
# Flask App Configuration
# =========================================================

app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB upload limit

Bootstrap(app)

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# =========================================================
# Form
# =========================================================

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField('Alpha', default=1.0)

    submit = SubmitField('Transfer Style')


# =========================================================
# Device Configuration
# =========================================================

# Force CPU for Render deployment
device = torch.device("cpu")


# =========================================================
# Model Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vgg_path = os.path.join(BASE_DIR, "vgg_normalised.pth")

decoder_path = os.path.join(
    BASE_DIR,
    "experiment",
    "final_exp",
    "decoder_final.pth"
)


# =========================================================
# Load Models
# =========================================================

print("Loading encoder...")
encoder = VGGEncoder(vgg_path).to(device)

print("Loading decoder...")
decoder = Decoder().to(device)

print("Loading decoder weights...")
decoder.load_state_dict(
    torch.load(decoder_path, map_location=device)
)

encoder.eval()
decoder.eval()

print("Models loaded successfully!")


# =========================================================
# Helper Functions
# =========================================================

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):

    print("Starting transforms...")

    content_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    print("Images moved to device")

    with torch.no_grad():

        print("Encoding content image...")
        content_feats = encoder(content_image, is_test=True)

        print("Encoding style image...")
        style_feats = encoder(style_image, is_test=True)

        print("Applying AdaIN...")
        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        stylized_feats = (
            alpha * stylized_feats +
            (1 - alpha) * content_feats
        )

        print("Decoding stylized image...")
        stylized_image = decoder(stylized_feats)

        print("Style transfer complete!")

    return stylized_image


def save_image(image, path):

    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)

    print(f"Image saved at: {path}")


# =========================================================
# Routes
# =========================================================

@app.route('/', methods=['GET', 'POST'])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        print("Form submitted")

        # =================================================
        # Content Image
        # =================================================

        if form.content.data and form.content.data.filename:

            if allowed_file(form.content.data.filename):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    content_filename
                )

                form.content.data.save(content_path)

                form.content_path.data = content_filename

                print(f"Content image saved: {content_filename}")

        else:
            content_filename = form.content_path.data

        # =================================================
        # Style Image
        # =================================================

        if form.style.data and form.style.data.filename:

            if allowed_file(form.style.data.filename):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    style_filename
                )

                form.style.data.save(style_path)

                form.style_path.data = style_filename

                print(f"Style image saved: {style_filename}")

        else:
            style_filename = form.style_path.data

        # =================================================
        # Style Transfer
        # =================================================

        if content_filename and style_filename:

            content_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                content_filename
            )

            style_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                style_filename
            )

            try:

                print("Opening images...")

                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)

                print("Starting style transfer...")

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                result_filename = 'stylized_' + content_filename

                result_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    result_filename
                )

                save_image(stylized_image, result_path)

                result_image = result_filename

                print("Transfer successful!")

            except Exception as e:

                print("ERROR:", e)

                error = str(e)

        else:

            if not content_filename and not style_filename:
                error = 'Please upload both content and style images.'

            elif not content_filename:
                error = 'Please upload a content image.'

            elif not style_filename:
                error = 'Please upload a style image.'

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


@app.route('/uploads/<filename>')
def send_image(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


@app.route('/examples/<path:filename>')
def send_example(filename):

    return send_from_directory(
        'examples',
        filename
    )


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )