from __future__ import annotations
import io
import json
import time
import base64
import mimetypes
from typing import Dict, Any, List

import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI

# ============== PAGE SETUP ==============
st.set_page_config(page_title="Nutrition Extractor (Image → Text)", layout="centered")
st.title("Nutrition Extractor (Image → Text)")
st.caption("Upload nutrition labels, get JSON back — schema-less, robust, no response_format parameter.")

# ============== AUTH IN UI ==============
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

with st.sidebar:
    st.subheader("Authentication")
    st.session_state["api_key"] = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="Stored only in this session; not written to disk."
    )

api_key = st.session_state["api_key"]
if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to enable extraction.")
    st.stop()

client = OpenAI(api_key=api_key)

# Optional diagnostics
try:
    import sys, openai  # type: ignore
    st.sidebar.caption(f"Diagnostics ▸ openai {openai.__version__} | python {sys.version.split()[0]}")
except Exception:
    pass

# ============== HELPERS ==============
def infer_product_id(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    return stem or "unknown"

def to_data_url(file_bytes: bytes, filename: str) -> str:
    """
    Convert image bytes to a base64 data URL so we can send it inline to the Responses API.
    """
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# Instruction for schema-less JSON
INSTRUCTION = (
    "Extract all nutrition and related values you can read from this label into a single JSON object.\n"
    "- Include per-100g/ml and per-serving if present.\n"
    "- Use numbers for numeric fields (no units inside numbers).\n"
    "- If an item is not printed on the label, omit it (do not guess).\n"
    "- Include a top-level 'product_id' with the provided value.\n"
    "- Examples of helpful keys (use only if printed): "
    "energy_kJ_100, energy_kcal_100, energy_kJ_serv, energy_kcal_serv, "
    "fat_g_100, saturates_g_100, fat_g_serv, saturates_g_serv, carbohydrate_g_100, sugars_g_100, "
    "carbohydrate_g_serv, sugars_g_serv, fibre_g_100, fibre_g_serv, protein_g_100, protein_g_serv, "
    "salt_g_100, salt_g_serv, sodium_g_100, sodium_g_serv, alcohol_g_100, alcohol_g_serv, "
    "vitamins:{...}, minerals:{...}, amino_acids:[...], probiotics:[...], botanicals:[...], "
    "other_nutrients:[...], ingredients_text, allergens_present, may_contain, sweeteners_present.\n"
    "- Return JSON only via the function call (no prose)."
)

# Minimal tool (function) to force a JSON object back (schema-less but structured)
NUTRITION_TOOL_MINIMAL = [{
    "type": "function",
    "name": "NutritionDump",
    "description": "Return any nutrition facts found as a single JSON object.",
    "parameters": {
        "type": "object",
        "properties": {},               # <-- REQUIRED, even if empty
        "additionalProperties": True    # allow any keys/shape
    }
}]


def extract_from_image_bytes_noschema(
    client: OpenAI,
    model_name: str,
    img_bytes: bytes,
    filename: str,
    product_id: str
) -> Dict[str, Any]:
    """
    Responses API (no response_format). We force a function/tool call that returns a JSON object.
    Image is sent inline as a base64 data URL. Returns a Python dict.
    """
    # Normalize orientation; re-encode to JPEG to keep payload efficient
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
        img_bytes = buf.getvalue()
    except Exception:
        pass

    data_url = to_data_url(img_bytes, filename)

    resp = client.responses.create(
        model=model_name,  # e.g., "gpt-4o"
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": INSTRUCTION + f"\nproduct_id to include: {product_id}"},
                {"type": "input_image",
                 "image_url": data_url}
            ]
        }],
        tools=NUTRITION_TOOL_MINIMAL,
        tool_choice={"type": "function", "name": "NutritionDump"},
        temperature=0
    )

    # ---- Parse the tool call arguments into a dict ----
    # Preferred typed-object path
    try:
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") in ("tool_call", "function_call"):
                    tc = getattr(c, "tool_call", None)
                    if tc is not None:
                        func = getattr(tc, "function", None)
                        args = (getattr(func, "arguments", None)
                                if func is not None else None) or getattr(tc, "arguments", None)
                        if args:
                            data = json.loads(args)
                            data.setdefault("product_id", product_id)
                            return data
    except Exception:
        pass

    # Raw-dict fallback
    raw = {}
    if hasattr(resp, "model_dump"):
        try:
            raw = resp.model_dump() or {}
        except Exception:
            raw = {}
    if not raw and hasattr(resp, "json"):
        try:
            raw = json.loads(resp.json())
        except Exception:
            raw = {}

    for item in raw.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("tool_call", "function_call"):
                tc = c.get("tool_call", {}) or {}
                args = tc.get("function", {}).get("arguments") or tc.get("arguments")
                if args:
                    data = json.loads(args)
                    data.setdefault("product_id", product_id)
                    return data

    # Last resort: wrap any text we got (shouldn’t trigger with tool_choice forced)
    text = getattr(resp, "output_text", "") or raw.get("output_text", "") or "{}"
    try:
        data = json.loads(text)
    except Exception:
        data = {"raw": text}
    data.setdefault("product_id", product_id)
    return data

def pretty_json_block(d: Dict[str, Any]) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2)

# ============== SIDEBAR SETTINGS ==============
with st.sidebar:
    st.subheader("Settings")
    model = st.selectbox("Model", options=["gpt-4o"], index=0, help="Vision-capable model.")
    max_items = st.number_input("Max images to process", min_value=1, max_value=500, value=500, step=1)

# ============== MAIN UI ==============
uploads = st.file_uploader(
    "Upload nutrition label images (JPG/PNG/WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="uploader",
)

if uploads:
    st.write(f"Selected **{len(uploads)}** file(s). Filenames will be used as product IDs.")

run = st.button("Extract nutrition values", type="primary", use_container_width=True)

if run:
    if not uploads:
        st.stop()

    outputs: List[str] = []
    total = min(len(uploads), max_items)
    progress = st.progress(0.0, text="Starting…")

    for idx, up in enumerate(uploads[:total], start=1):
        filename = up.name
        product_id = infer_product_id(filename)
        try:
            img_bytes = up.read()
            payload = extract_from_image_bytes_noschema(client, model, img_bytes, filename, product_id)
            block = pretty_json_block(payload)
            outputs.append(block + "\n---\n")
        except Exception as e:
            outputs.append(pretty_json_block({"product_id": product_id, "error": str(e)}) + "\n---\n")
        finally:
            progress.progress(idx / max(1, total), text=f"Processed {idx} / {total}")
            time.sleep(0.05)

    final_text = "\n".join(outputs).strip()
    st.success("Completed.")
    st.text_area("Preview (first 5,000 chars)", final_text[:5000], height=300)

    st.download_button(
        label="Download .txt",
        data=final_text.encode("utf-8"),
        file_name="nutrition_extractions.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.caption("Tip: Use clear, sharp, square-on photos to improve extraction quality.")
