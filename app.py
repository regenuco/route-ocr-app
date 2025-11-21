import streamlit as st
import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
import io
import csv
import re
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------------------
# LOAD TrOCR (small printed)
# ---------------------------
@st.cache_resource
def load_ocr():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    model.eval()
    return processor, model

processor, trocr_model = load_ocr()

# ---------------------------
# Detect & Auto-Zoom Lines
# ---------------------------
def detect_and_crop_lines(pil_image: Image.Image):
    """
    Detect horizontal text lines by contour, crop them, zoom 2x, enhance, and return list of OpenCV images.
    """
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    cnts, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

    crops = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        # Ignore small blobs
        if h < 25 or w < 200:
            continue

        crop = img[y:y+h, x:x+w]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.convertScaleAbs(gray2, alpha=1.6, beta=0)

        thresh = cv2.adaptiveThreshold(
            gray2, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 8
        )

        crops.append(thresh)

    return crops

# ---------------------------
# OCR a single line with TrOCR
# ---------------------------
def trocr_ocr_line(cv2_img):
    """
    cv2_img: single-channel or 3-channel OpenCV image.
    Returns recognized text string.
    """
    if len(cv2_img.shape) == 2:
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)

    with torch.no_grad():
        inputs = processor(images=pil_img, return_tensors="pt")
        out = trocr_model.generate(**inputs, max_length=64)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return text.strip()

# ---------------------------
# OCR each cropped line
# ---------------------------
def ocr_each_line(crops):
    lines = []
    for img in crops:
        text = trocr_ocr_line(img)
        lines.append(text)
    return "\n".join(lines)

# ---------------------------
# Parse OCR into rows
# ---------------------------
def extract_rows(text: str):
    rows = []
    for line in [x.strip() for x in text.splitlines() if x.strip()]:
        low = line.lower()

        # Skip header-ish junk
        if any(x in low for x in ["company", "phone", "dot", "plate", "driver", "name", "date"]):
            continue

        # gallons + date at end
        m = re.search(r"(\d+)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$", line)
        if not m:
            continue

        gallons = m.group(1)
        date = m.group(2)
        before = line[:m.start()].strip()

        # address begins at first digit (street number)
        addr_idx = next((i for i, ch in enumerate(before) if ch.isdigit()), None)

        if addr_idx is not None:
            name = before[:addr_idx].strip(" ,")
            addr = before[addr_idx:].strip(" ,")
        else:
            name = before.strip(" ,")
            addr = ""

        rows.append({
            "name_ocr": name,
            "address_ocr": addr,
            "gallons": gallons,
            "date": date
        })
    return rows

# ---------------------------
# Address verification (Nominatim)
# ---------------------------
def lookup_business(address: str):
    try:
        if not address:
            return None, "<red>(no address parsed)</red>"

        params = {"q": address, "format": "json", "limit": 1}
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "RouteOCR-App"}
        )
        data = r.json()
        if not data:
            return None, "<red>(no match found)</red>"

        return data[0].get("display_name", ""), None
    except Exception:
        return None, "<red>(lookup failed)</red>"

# ---------------------------
# Verify & correct row (with red notes)
# ---------------------------
def verify_and_correct(row):
    addr = row["address_ocr"]
    name = row["name_ocr"]

    verified, issue = lookup_business(addr)

    # If lookup failed or no result
    if issue:
        row["name_final"] = f"{name} {issue}"
        row["address_final"] = f"{addr} {issue}"
        return row

    # If verified string doesn't contain OCR name
    if name and name.lower() not in verified.lower():
        row["name_final"] = f"{verified} <red>(OCR name mismatch: {name})</red>"
    else:
        row["name_final"] = name or verified

    row["address_final"] = verified
    return row

# ---------------------------
# CSV export
# ---------------------------
def rows_to_csv_bytes(rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["name", "address", "gallons", "date"])
    for r in rows:
        writer.writerow([
            r.get("name_final", ""),
            r.get("address_final", ""),
            r.get("gallons", ""),
            r.get("date", "")
        ])
    return out.getvalue().encode("utf-8")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Route Sheet OCR → CSV", layout="centered")
st.title("📄 Route Sheet OCR → CSV (TrOCR small)")

st.write(
    "Upload a photo of a handwritten route sheet. "
    "I'll auto-zoom each line, read it, verify addresses, mark uncertainties in red, "
    "and give you a CSV."
)

uploaded = st.file_uploader("Upload route sheet image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Sheet"):
        with st.spinner("Processing… this may take a bit the first time while the model loads."):
            crops = detect_and_crop_lines(pil_img)
            if not crops:
                st.error("No text-like lines detected. Try a clearer top-down photo.")
            else:
                text = ocr_each_line(crops)
                extracted = extract_rows(text)
                verified = [verify_and_correct(r) for r in extracted]

                if not verified:
                    st.error("No valid rows (name/address/gallons/date) detected.")
                else:
                    display_rows = []
                    for r in verified:
                        display_rows.append({
                            "Name": r["name_final"],
                            "Address": r["address_final"],
                            "Gallons": r["gallons"],
                            "Date": r["date"]
                        })

                    df = pd.DataFrame(display_rows)
                    st.subheader("Extracted & Verified Data")
                    st.dataframe(df, use_container_width=True)

                    csv_bytes = rows_to_csv_bytes(verified)
                    st.download_button(
                        "⬇️ Download CSV",
                        csv_bytes,
                        "route_sheet.csv",
                        "text/csv"
                    )
