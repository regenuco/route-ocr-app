import io
import re
import csv
import cv2
import numpy as np
import pytesseract
import requests
import streamlit as st
import pandas as pd
from PIL import Image

# If Tesseract isn't on PATH, uncomment and set this:
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # or "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# ------------- IMAGE LINE DETECTION + AUTO ZOOM ------------- #

def detect_and_crop_lines(pil_image: Image.Image):
    """
    Detect handwritten rows by finding horizontal contours.
    Returns list of cropped, zoomed OpenCV images ready for OCR.
    """
    # Convert PIL -> OpenCV (BGR)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth & de-noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Dilate to merge strokes into line-blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])  # sort top->bottom

    crops = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        # ignore tiny boxes (not rows)
        if h < 25 or w < 200:
            continue

        crop = img[y:y + h, x:x + w]

        # Zoom for better OCR
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Enhance contrast
        gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.convertScaleAbs(gray2, alpha=1.6, beta=0)

        # Threshold for cleaner OCR
        thresh = cv2.adaptiveThreshold(
            gray2, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 8
        )

        crops.append(thresh)
    return crops


# ------------- OCR FOR EACH LINE ------------- #

def ocr_each_line(crops):
    texts = []
    for img in crops:
        text = pytesseract.image_to_string(img, config="--psm 6")
        texts.append(text)
    return "\n".join(texts)


# ------------- PARSE OCR TEXT INTO ROWS ------------- #

def extract_rows(text: str):
    """
    Extract lines that look like: NAME ADDRESS ... GALLONS DATE
    """
    rows = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        low = line.lower()

        # Skip header-ish noise
        if any(x in low for x in ["company", "phone", "dot", "plate", "driver", "name", "date"]):
            continue

        # gallons + date at end, e.g. "80 10/31/25"
        m = re.search(r"(\d+)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$", line)
        if not m:
            continue

        gallons = m.group(1)
        date = m.group(2)
        before = line[:m.start()].strip()

        # Find first digit in prefix as address start
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


# ------------- ONLINE LOOKUP (NOMINATIM) ------------- #

def lookup_business(address: str):
    """
    Try to verify/clean the address & business info using Nominatim (OpenStreetMap).
    Returns (display_name, issue_note or None).
    """
    try:
        if not address:
            return None, "<red>(no address parsed)</red>"

        params = {
            "q": address,
            "format": "json",
            "limit": 1
        }
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "RouteSheetOCR-Demo"}
        )
        data = r.json()

        if not data:
            return None, "<red>(no match found)</red>"

        display_name = data[0].get("display_name", "")
        return display_name, None

    except Exception:
        return None, "<red>(lookup failed)</red>"


# ------------- VERIFICATION + RED NOTES ------------- #

def verify_and_correct(row):
    """
    Use the raw OCR name/address and try to verify/fix via Nominatim.
    Add RED notes for anything uncertain, in parentheses.
    """
    addr = row["address_ocr"]
    name = row["name_ocr"]

    verified_display, issue = lookup_business(addr)

    # If lookup failed or no data
    if issue:
        # mark both name & address with red note
        row["name_final"] = f"{name} {issue}"
        row["address_final"] = f"{addr} {issue}"
        return row

    # If we got a verified display string:
    # Example: "Coco Rosa Restaurant, 1155 Webster Ave, Bronx, New York, USA"
    verified_str = verified_display

    # Decide on final name:
    if name and name.lower() not in verified_str.lower():
        # Name doesn't appear in verified address string -> mismatch
        row["name_final"] = f"{verified_str} <red>(OCR name mismatch: {name})</red>"
    else:
        # Name appears to match or is empty
        row["name_final"] = name or verified_str

    # Use verified display string as address line
    row["address_final"] = verified_str

    return row


# ------------- MAIN PROCESSING PIPELINE ------------- #

def process_image(pil_image: Image.Image):
    """
    Full pipeline:
     1) detect & crop lines (auto zoom)
     2) OCR each line
     3) parse rows
     4) online lookup & verification
    """
    crops = detect_and_crop_lines(pil_image)
    if not crops:
        return []

    raw_text = ocr_each_line(crops)
    extracted = extract_rows(raw_text)
    verified = [verify_and_correct(r) for r in extracted]
    return verified


def rows_to_csv_bytes(rows):
    """
    Convert rows into CSV bytes for download.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["name", "address", "gallons", "date"])

    for r in rows:
        writer.writerow([
            r.get("name_final", ""),
            r.get("address_final", ""),
            r.get("gallons", ""),
            r.get("date", "")
        ])

    return output.getvalue().encode("utf-8")


# ------------- STREAMLIT WEB APP UI ------------- #

st.set_page_config(page_title="Route Sheet OCR Demo", layout="centered")

st.title("📄 Route Sheet OCR → CSV")
st.write("Upload a photo of a handwritten route sheet. I’ll read it, look up the locations, "
         "add red notes for anything uncertain, and give you a downloadable CSV.")

uploaded_file = st.file_uploader("Upload route sheet image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Sheet"):
        with st.spinner("Reading handwriting, looking up addresses, and building your CSV..."):
            rows = process_image(pil_img)

        if not rows:
            st.error("No valid rows detected. Try a clearer photo (flat, good light, top-down).")
        else:
            # Show results as a table
            display_rows = []
            for r in rows:
                display_rows.append({
                    "Name (corrected)": r.get("name_final", ""),
                    "Address (verified)": r.get("address_final", ""),
                    "Gallons": r.get("gallons", ""),
                    "Date": r.get("date", "")
                })
            df = pd.DataFrame(display_rows)
            st.subheader("Extracted & Verified Data")
            st.dataframe(df, use_container_width=True)

            # CSV download
            csv_bytes = rows_to_csv_bytes(rows)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name="route_sheet_processed.csv",
                mime="text/csv"
            )
