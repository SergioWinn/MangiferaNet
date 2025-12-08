import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image
from src.config import IMAGE_SIZE

def prepare_image(uploaded_file):
    """
    Accepts a path, file-like (UploadedFile / BytesIO), or PIL Image.
    Returns numpy array shape (1, H, W, 3), dtype float32, rescaled to [0,1].
    Raises ValueError on invalid image.
    """
    try:
        # If already a PIL Image
        if isinstance(uploaded_file, Image.Image):
            img = uploaded_file
        else:
            img = Image.open(uploaded_file)

        # Fix rotation from EXIF if present
        img = ImageOps.exif_transpose(img).convert('RGB')

        # Resize
        img = img.resize(IMAGE_SIZE)

        # To array, add batch dim, normalize
        img_array = image.img_to_array(img).astype('float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        return img_array
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")