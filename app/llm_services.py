# from app.config import client
# from app.prompts import build_prompt

# def analyze_condition(img, item_name):

#     prompt = build_prompt(item_name)

#     response = client.models.generate_content(
#         model="gemma-4-31b-it",
#         contents=[
#             prompt,
#             img
#         ]
#     )

#     return response.text

from io import BytesIO
import ollama
from app.prompts import build_prompt

def analyze_condition(img, item_name):
    prompt = build_prompt(item_name)

    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_bytes],
            }
        ]
    )

    return response["message"]["content"]