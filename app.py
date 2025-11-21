import streamlit as st
import pandas as pd
import numpy as np
import requests
import csv
import io
from PIL import Image
import re
from openai import OpenAI

# -------------------------
# Load OpenAI Client
# -------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------
# Call OpenAI Vision OCR
# -------------------------
def run_vision_ocr(pil_img):
    """
    Sends the uploaded image to OpenAI Vision and returns the raw text.
    """
    # Convert to bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract ALL text from this image. Include line breaks."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_bytes.hex()}"}
                ]
            }
        ]
    )

    return response.output_text

# -------------------------
# Parse OCR into structured rows
# -------------------------
def extract_rows(raw_text):
    rows = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        if any(x in low for x in ["company", "phone", "plate", "driver", "name", "date"]):
            continue

        # gallons + date pattern
        m = re.search(r"(\d+)\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})$", line)
        if not m:
            continue

        gallons = m.group(1)
        date = m.group(2)
        before = line[:m.start()].strip()

        # find start of address
        addr_idx = next((i for i, ch in enumerate(before) if ch.isdigit()), None)

        if addr_idx is not None:
            name = before[:addr_idx].strip(" ,")
            addr = before[addr_idx:].strip(" ,")
        else:
            name = before
            addr = ""

        rows.append({
            "name_ocr": name,
            "address_ocr": addr,
            "gallons": gallons,
            "date": date
        })

    return rows

# -------------------------
# Address verification
# -------------------------
def lookup_address(address):
    if not address:
        return None, "<red>(no address parsed)</red>"

    try:
        params = {
            "q": address,
            "format": "json",
            "limit": 1
        }
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "RouteOCR-App"}
        )
        data = r.json()
        if not data:
            return None, "<red>(no match found)</red>"
        return data[0]["display_name"], None
    except:
        return None, "<red>(lookup failed)</red>"

# -------------------------
# Verify & correct rows
# -------------------------
def verify_row(row):
    addr = row["address_ocr"]
    name = row["name_ocr"]

    verified, issue = lookup_address(addr)

    if issue:
        row["name_final"] = f"{name} {issue}"
        row["address_final"] = f"{addr} {issue}"
        return row

    if name and name.lower() not in verified.lower():
        row["name_final"] = f"{verified} <red>(OCR name mismatch: {name})</red>"
    else:
        row["name_final"] = name or verified

    row["address_final"] = verified
    return row

# -------------------------
# CSV Export
# -------------------------
def make_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["name", "address", "gallons", "date"])
    for r in rows:
        writer.writerow([
            r["name_final"],
            r["address_final"],
            r["gallons"],
            r["date"]
        ])
    return out.getvalue().encode("utf-8")

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Route Sheet OCR → CSV", layout="centered")
st.title("📄 Route Sheet OCR → CSV (OpenAI Vision + Verification)")

uploaded = st.file_uploader("Upload route sheet image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Sheet"):
        with st.spinner("Running Vision OCR…"):
            raw_text = run_vision_ocr(pil_img)

        with st.spinner("Extracting rows…"):
            extracted = extract_rows(raw_text)

        with st.spinner("Verifying addresses…"):
            verified = [verify_row(r) for r in extracted]

        if not verified:
            st.error("No valid rows detected.")
        else:
            df = pd.DataFrame(verified)
            st.subheader("Extracted & Verified Data")
            st.dataframe(df, use_container_width=True)

            csv_bytes = make_csv(verified)
            st.download_button(
                "⬇️ Download CSV",
                csv_bytes,
                "route_sheet.csv",
                "text/csv"
            )
