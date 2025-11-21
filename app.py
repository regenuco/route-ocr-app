import streamlit as st
import requests
import base64
import csv
import io
from PIL import Image
import re

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Route Sheet OCR (OpenAI Vision)", layout="wide")
st.title("📦 Route Sheet OCR + Business Verification (OpenAI Vision)")
st.write("Upload your route sheet image, and the AI will extract **Name, Address, Gallons, and Date** automatically.")

# -----------------------------
# Load API Key
# -----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# Convert image → base64
# -----------------------------
def encode_image(image_file):
    img_bytes = image_file.read()
    return base64.b64encode(img_bytes).decode("utf-8")

# -----------------------------
# Call OpenAI Vision to read the sheet
# -----------------------------
def call_openai_vision(base64_image):
    url = "https://api.openai.com/v1/responses"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    ### PROMPT:
    system_prompt = """
You are an OCR and document extraction expert.
Extract every row from this handwritten route sheet image.

For each row, return ONLY structured JSON:

[
  {
    "name": "...",
    "address": "...",
    "gallons": "...",
    "date": "..."
  }
]

- Fix partial or misspelled restaurant names.
- Fix addresses whenever possible.
- Keep gallons and date exactly as written.
- If unsure, guess but mark with "(uncertain)".
"""

    payload = {
        "model": "gpt-4.1",
        "input": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract all rows from this route sheet."
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    return response.json()

# -----------------------------
# Try to parse JSON returned by OpenAI
# -----------------------------
def parse_json_from_response(resp):
    try:
        # gpt-4.1 returns: response['output'][0]['content'][0]['text']
        text = resp["output"][0]["content"][0]["text"]
        return eval(text)
    except:
        return None

# -----------------------------
# Streamlit File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your route sheet", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Sheet", use_column_width=True)

    with st.spinner("Reading sheet using OpenAI Vision..."):
        base64_img = encode_image(uploaded_file)
        raw_resp = call_openai_vision(base64_img)
        parsed_rows = parse_json_from_response(raw_resp)

    if parsed_rows is None:
        st.error("OpenAI could not parse the OCR output. Here is the raw response:")
        st.write(raw_resp)
    else:
        st.success("OCR Completed!")
        st.subheader("Extracted Rows")

        for idx, row in enumerate(parsed_rows, start=1):
            st.markdown(f"### Row {idx}")
            st.markdown(f"- **Name:** {row.get('name','')}")
            st.markdown(f"- **Address:** {row.get('address','')}")
            st.markdown(f"- **Gallons:** {row.get('gallons','')}")
            st.markdown(f"- **Date:** {row.get('date','')}")
            st.markdown("---")

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["name", "address", "gallons", "date"])
        for r in parsed_rows:
            writer.writerow([
                r.get("name", ""),
                r.get("address", ""),
                r.get("gallons", ""),
                r.get("date", "")
            ])

        st.download_button(
            label="⬇️ Download CSV",
            data=output.getvalue(),
            file_name="route_sheet_processed.csv",
            mime="text/csv"
        )
