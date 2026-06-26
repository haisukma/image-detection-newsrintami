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

detected_item = input("Masukkan hasil deteksi YOLO: ")

img = Image.open(image_path)

CONDITIONS = {
    "Bracing": [
        "Normal",
        "Korosi",
        "Bengkok",
        "Hilang"
    ],
    "Jumper": [
        "Normal",
        "Korosi",
        "Rantas",
        "Putus",
        "Lepas",
        "Hilang"
    ],
    "Insulator": [
        "Normal",
        "Retak Rambut",
        "Gumpil",
        "Pecah",
        "Flashover"
    ],
    "Aksesoris Sisi Cold": [
        "Normal",
        "Korosi"
    ],
    "Aksesoris Sisi Hot": [
        "Normal",
        "Korosi"
    ]
}

if detected_item not in CONDITIONS:
    print(f"Item '{detected_item}' tidak dikenali.")
    exit()

prompt = f"""
Anda adalah seorang inspektor tower transmisi tenaga listrik yang berpengalaman.

Hasil deteksi YOLO menunjukkan bahwa item pada gambar adalah:

{detected_item}

Analisis kondisi item tersebut berdasarkan gambar.

Anda HANYA boleh memilih SATU kondisi berikut:

{chr(10).join("- " + c for c in CONDITIONS[detected_item])}

Jangan memilih kondisi yang tidak ada pada daftar.

Jika kondisi tidak dapat ditentukan karena gambar tidak jelas atau objek tidak terlihat, jawab:

Kondisi: Tidak Dapat Dinilai

Jawab dengan format berikut:

Item: {detected_item}
Kondisi:
Confidence:
"""

response = client.models.generate_content(
    model="gemma-4-31b-it",
    contents=[
        prompt,
        img
    ]
)

print(response.text)