from google import genai
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

# for model in client.models.list():
#     print(model.name)

image_path = input("Masukkan path gambar: ")

img = Image.open(image_path)

response = client.models.generate_content(
    model="gemma-4-31b-it",
    contents=[
        """
        Anda adalah inspektor tower transmisi.

        Analisis gambar dan tentukan apakah terdapat korosi.

        Format jawaban:

        Status: Korosi atau Normal
        Confidence: xx%
        Alasan: ...
        """,
        img
    ]
)

print(response.text)