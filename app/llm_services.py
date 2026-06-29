from app.config import client
from app.prompts import build_prompt

def analyze_condition(img, item_name):

    prompt = build_prompt(item_name)

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=[
            prompt,
            img
        ]
    )

    return response.text