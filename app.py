import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import re
import requests
import csv
import io

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Route Sheet OCR + Business Verification", layout="wide")

st.title("📦 Route Sheet OCR + Business Verification")
st.write("Upload a handwritten route sheet image and download a cleaned CSV with verified business names & addresses.")


# -----------------------------
# LOAD EASYOCR READER (CPU ONLY)
# -----------------------------
@st.cache_resource
def load_reader():
    # English only; GPU disabled for Streamlit Cloud (CPU-only)
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()


# -----------------------------
# AUTO-DETECT & CROP LINES
# -----------------------------
def detect_and_crop_lines(pil_image: Image.Image):
    """
    Detects handwritten rows by finding horizontal contours.
    Returns list of cropped, zoomed images as numpy arrays.
    """
    # Convert PIL -> OpenCV BGR
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    crops = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Filter out tiny noise
        if h < 25 or w < 200:
            continue

        crop = img[y:y + h, x:x + w]

        # Zoom for better OCR
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Enhance for OCR
        gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.convertScaleAbs(gray2, alpha=1.6, beta=0)

        thresh = cv2.adaptiveThreshold(
            gray2,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8
        )

        crops.append(thresh)

    return crops


# -----------------------------
# OCR EACH CROPPED LINE WITH EASYOCR
# -----------------------------
def ocr_each_line(crops):
    """
    Run EasyOCR on each zoomed/threshed line and join text.
    """
    lines = []
    for crop in crops:
        # EasyOCR expects RGB array
        if len(crop.shape) == 2:
            # grayscale -> RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        else:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        result = reader.readtext(crop_rgb, detail=0)
        text = " ".join(result).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


# -----------------------------
# EXTRACT ROWS (NAME, ADDRESS, GALLONS, DATE)
# -----------------------------
def extract_rows(text: str):
    rows = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        # Skip obvious header junk
        if any(x in line.lower() for x in [
            "company", "phone", "dot", "plate", "driver", "name", "date"
        ]):
            continue

        # Detect gallons and date at the end of the line
        match = re.search(r"(\d+)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$", line)
        if not match:
            continue

        gallons = match.group(1)
        date = match.group(2)
        before = line[:match.start()].strip()

        # Find first digit in the "before" part = start of address
        addr_idx = next((i for i, c in enumerate(before) if c.isdigit()), None)

        if addr_idx is not None:
            name = before[:addr_idx].strip()
            addr = before[addr_idx:].strip()
        else:
            name = before
            addr = ""

        rows.append(
            {
                "raw": line,
                "name_ocr": name,
                "address_ocr": addr,
                "gallons": gallons,
                "date": date,
            }
        )

    return rows


# -----------------------------
# BUSINESS LOOKUP (Nominatim)
# -----------------------------
def lookup_business(addr: str):
    """
    Use Nominatim (OpenStreetMap) to look up an address.
    Returns (display_name, issue_note_or_None).
    If uncertain, returns red-tagged note.
    """
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": addr, "format": "json", "limit": 1},
            headers={"User-Agent": "Streamlit-OCR-App"}
        )
        data = r.json()
        if not data:
            return None, "<red>(no match found)</red>"
        display = data[0].get("display_name", "")
        return display, None
    except Exception:
        return None, "<red>(lookup failed)</red>"


# -----------------------------
# VERIFY & ADD RED NOTES
# -----------------------------
def verify_and_correct(row):
    addr = row["address_ocr"]
    name = row["name_ocr"]

    verified_display, issue = lookup_business(addr)

    # If lookup failed or no match
    if issue:
        row["name_final"] = f"{name} {issue}"
        row["address_final"] = f"{addr} {issue}"
        return row

    # If we got a verified address string
    if verified_display:
        # If name not obviously contained in verified string
        if name and name.lower() not in verified_display.lower():
            row["name_final"] = f"{verified_display} <red>(OCR name mismatch: {name})</red>"
        else:
            row["name_final"] = name or verified_display

        row["address_final"] = verified_display
    else:
        # No verified result, mark in red
        row["name_final"] = f"{name} <red>(no verified business name)</red>"
        row["address_final"] = f"{addr} <red>(no verified address)</red>"

    return row


# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded_file = st.file_uploader("Upload Route Sheet Image", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Sheet", use_column_width=True)

    pil_img = Image.open(uploaded_file)

    with st.spinner("Processing image (detecting lines, running OCR, verifying addresses)..."):
        crops = detect_and_crop_lines(pil_img)
        if not crops:
            st.error("Could not detect any handwritten lines. Try a clearer or higher-resolution image.")
        else:
            raw_text = ocr_each_line(crops)
            extracted = extract_rows(raw_text)
            final_rows = [verify_and_correct(r) for r in extracted]

            st.subheader("Extracted & Verified Rows (preview)")
            if not final_rows:
                st.warning("No valid rows (with gallons + date) were detected from this sheet.")
            else:
                # Show as a simple table-like structure
                for idx, r in enumerate(final_rows, start=1):
                    st.markdown(f"**Row {idx}:**")
                    st.markdown(f"- Name: {r['name_final']}")
                    st.markdown(f"- Address: {r['address_final']}")
                    st.markdown(f"- Gallons: `{r['gallons']}`")
                    st.markdown(f"- Date: `{r['date']}`")
                    st.markdown("---")

                # Build CSV in memory
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["name", "address", "gallons", "date"])
                for r in final_rows:
                    writer.writerow([
                        r["name_final"],
                        r["address_final"],
                        r["gallons"],
                        r["date"],
                    ])

                st.download_button(
                    label="⬇️ Download CSV",
                    data=output.getvalue(),
                    file_name="route_sheet_processed.csv",
                    mime="text/csv"
                )
else:
    st.info("Upload a route sheet image to begin.")
