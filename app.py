import streamlit as st
import cv2
import numpy as np
import easyocr
import requests
import pandas as pd
from PIL import Image
import io
import csv
import re

# ----------------------
# EasyOCR Reader (loads once)
# ----------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# ----------------------
# Detect & Auto-Zoom Handwritten Lines
# ----------------------

def detect_and_crop_lines(pil_image):
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

# ----------------------
# OCR Each Line Using EasyOCR
# ----------------------

def ocr_each_line(crops):
    lines = []
    for img in crops:
        results = reader.readtext(img, detail=0, paragraph=True)
        text = " ".join(results)
        lines.append(text)
    return "\n".join(lines)

# ----------------------
# Parse OCR Text
# ----------------------

def extract_rows(text):
    rows = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        low = line.lower()
        if any(x in low for x in ["company", "phone", "dot", "plate", "driver", "name", "date"]):
            continue

        m = re.search(r"(\d+)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$", line)
        if not m:
            continue

        gallons = m.group(1)
        date = m.group(2)
        before = line[:m.start()].strip()

        addr_idx = next((i for i, ch in enumerate(before) if ch.isdigit()), None)

        if addr_idx is not None:
            name = before[:addr_idx].strip(" ,")
            addr = before[addr_idx:].strip(" ,")
        else:
            name = before.strip(" ,")
            addr = ""

        rows.append({
            "raw": line,
            "name_ocr": name,
            "address_ocr": addr,
            "gallons": gallons,
            "date": date
        })

    return rows

# ----------------------
# Online Lookup (Nominatim)
# ----------------------

def lookup_business(address):
    try:
        if not address:
            return None, "<red>(no address parsed)</red>"

        params = {"q": address, "format": "json", "limit": 1}
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "RouteSheetOCR-App"}
        )
        data = r.json()

        if not data:
            return None, "<red>(no match found)</red>"

        return data[0].get("display_name", ""), None

    except:
        return None, "<red>(lookup failed)</red>"

# ----------------------
# Verify & Correct
# ----------------------

def verify_and_correct(row):
    addr = row["address_ocr"]
    name = row["name_ocr"]

    verified, issue = lookup_business(addr)

    if issue:
        row["name_final"] = f"{name} {issue}"
        row["address_final"] = f"{addr} {issue}"
        return row

    if name.lower() not in verified.lower():
        row["name_final"] = f"{verified} <red>(OCR name mismatch: {name})</red>"
    else:
        row["name_final"] = name

    row["address_final"] = verified
    return row

# ----------------------
# CSV Export
# ----------------------

def rows_to_csv_bytes(rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["name", "address", "gallons", "date"])

    for r in rows:
        writer.writerow([
            r.get("name_final",""),
            r.get("address_final",""),
            r.get("gallons",""),
            r.get("date","")
        ])

    return out.getvalue().encode("utf-8")

# ----------------------
# STREAMLIT UI
# ----------------------

st.set_page_config(page_title="Route Sheet OCR", layout="centered")

st.title("📄 Route Sheet OCR → CSV")
st.write("Upload your handwritten route sheet. I’ll read it, verify addresses, "
         "flag uncertainties in red, and export a CSV.")

uploaded = st.file_uploader("Upload route sheet image", type=["jpg","jpeg","png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Sheet"):
        with st.spinner("Reading handwriting… please wait..."):
            crops = detect_and_crop_lines(pil_img)
            text = ocr_each_line(crops)
            extracted = extract_rows(text)
            verified = [verify_and_correct(r) for r in extracted]

        if not verified:
            st.error("No valid rows detected. Try a clearer photo.")
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
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name="route_sheet.csv",
                mime="text/csv"
            )
