import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import requests
import csv
import io

st.set_page_config(page_title="Handwritten Route Sheet OCR", layout="wide")

st.title("📦 Route Sheet OCR + Business Verification")
st.write("Upload a handwritten route sheet and download a clean CSV.")

# ----------------------------------------
# AUTO-ZOOM LINE DETECTION
# ----------------------------------------
def detect_and_crop_lines(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

    crops = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h < 25 or w < 200:
            continue

        crop = img[y:y+h, x:x+w]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.convertScaleAbs(gray2, alpha=1.6, beta=0)

        thresh = cv2.adaptiveThreshold(
            gray2, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            cv2.THRESH_BINARY,
            31, 8
        )

        crops.append(thresh)
    return crops


# ----------------------------------------
# OCR EACH ZOOMED CROP
# ----------------------------------------
def ocr_each_line(crops):
    texts = []
    for c in crops:
        pil_img = Image.fromarray(c)
        t = pytesseract.image_to_string(pil_img, config="--psm 6")
        texts.append(t)
    return "\n".join(texts)


# ----------------------------------------
# EXTRACT ROWS (NAME, ADDRESS, GALLONS, DATE)
# ----------------------------------------
def extract_rows(text):
    rows = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        # Skip headers
        if any(x in line.lower() for x in ["company","phone","dot","plate","driver","name","date"]):
            continue

        match = re.search(r"(\d+)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$", line)
        if not match:
            continue

        gallons = match.group(1)
        date = match.group(2)
        before = line[:match.start()].strip()

        addr_idx = next((i for i,c in enumerate(before) if c.isdigit()), None)
        if addr_idx:
            name = before[:addr_idx].strip()
            addr = before[addr_idx:].strip()
        else:
            name = before
            addr = ""

        rows.append({
            "raw": line,
            "name_ocr": name,
            "address_ocr": addr,
            "gallons": gallons,
            "date": date
        })

    return rows


# ----------------------------------------
# BUSINESS LOOKUP (NOMINATIM - FREE)
# ----------------------------------------
def lookup_business(addr):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": addr, "format": "json", "limit": 1},
            headers={"User-Agent": "StreamlitOCR-App"}
        )
        data = r.json()
        if not data:
            return None, "<red>(no match found)</red>"
        return data[0]["display_name"], None
    except:
        return None, "<red>(lookup failed)</red>"


# ----------------------------------------
# VERIFY ROW AND ADD RED NOTES
# ----------------------------------------
def verify_and_correct(row):
    verified, issue = lookup_business(row["address_ocr"])

    if issue:
        row["name_final"] = f"{row['name_ocr']} {issue}"
        row["address_final"] = f"{row['address_ocr']} {issue}"
        return row

    if row["name_ocr"].lower() not in verified.lower():
        row["name_final"] = f"{verified} <red>(OCR mismatch: {row['name_ocr']})</red>"
    else:
        row["name_final"] = row["name_ocr"]

    row["address_final"] = verified
    return row


# ----------------------------------------
# STREAMLIT UI
# ----------------------------------------
uploaded_file = st.file_uploader("Upload Route Sheet Image", type=["jpg","jpeg","png","heic"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Sheet", use_column_width=True)

    image = Image.open(uploaded_file)
    crops = detect_and_crop_lines(image)
    raw_text = ocr_each_line(crops)
    extracted = extract_rows(raw_text)
    final_rows = [verify_and_correct(r) for r in extracted]

    # Show Preview
    st.subheader("Extracted & Verified Rows")
    st.write(final_rows)

    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["name", "address", "gallons", "date"])

    for r in final_rows:
        writer.writerow([r["name_final"], r["address_final"], r["gallons"], r["date"]])

    st.download_button(
        label="Download CSV",
        data=output.getvalue(),
        file_name="route_sheet_processed.csv",
        mime="text/csv"
    )
