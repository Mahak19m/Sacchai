import pytesseract
from PIL import Image
import io

# ─────────────────────────────────────────
# Tesseract path (Windows)
# ─────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─────────────────────────────────────────
# Main OCR function
# ─────────────────────────────────────────
def extract_text_from_image(image_file):
    """
    Extract text from a WhatsApp screenshot or any image.
    Supports English and Hindi.
    Returns extracted text or error message.
    """
    try:
        # Open image from uploaded file bytes
        image = Image.open(io.BytesIO(image_file.read()))

        # Convert to RGB if needed (handles PNG transparency)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Try English + Hindi together
        text = pytesseract.image_to_string(
            image,
            lang='eng+hin',
            config='--psm 6'  # Assume uniform block of text
        )

        # Clean up extracted text
        text = text.strip()

        if not text or len(text) < 10:
            return {
                "status" : "no_text",
                "message": "No readable text found in image. Try a clearer screenshot.",
                "text"   : ""
            }

        return {
            "status" : "success",
            "message": f"Text extracted successfully ({len(text.split())} words)",
            "text"   : text
        }

    except Exception as e:
        return {
            "status" : "error",
            "message": f"Could not process image: {str(e)}",
            "text"   : ""
        }


# ─────────────────────────────────────────
# Test
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Test with a sample image file if available
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            class FakeUpload:
                def __init__(self, data):
                    self._data = data
                def read(self):
                    return self._data

            result = extract_text_from_image(FakeUpload(f.read()))
            print(f"Status  : {result['status']}")
            print(f"Message : {result['message']}")
            print(f"Text    :\n{result['text'][:500]}")
    else:
        print("Usage: python ocr_extractor.py <image_path>")
        print("Example: python ocr_extractor.py whatsapp_screenshot.png")