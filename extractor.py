from __future__ import annotations
import io
import json
import time
import base64
import mimetypes
import math
import csv
from typing import Dict, Any, List, Tuple

import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="Nutrition Extractor (Image → JSON)", layout="centered")
st.title("Nutrition Extractor (Image → JSON)")
st.caption("Upload nutrition labels, get JSON back — freeform. Includes cost estimate and CSV export.")

# ==================== AUTH IN UI ====================
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

# ==================== COST ESTIMATION HELPERS ====================

def estimate_image_tokens(width: int, height: int, tile: int = 512, base: int = 70, per_tile: int = 140) -> int:
    """Estimate visual tokens for an image based on tiling.
    tiles = ceil(W/512) * ceil(H/512); tokens = base + per_tile * tiles
    """
    tiles = math.ceil(width / tile) * math.ceil(height / tile)
    return base + per_tile * tiles


def estimate_cost_for_image(width: int, height: int,
                            text_tokens: int,
                            out_tokens: int,
                            input_price_per_m: float,
                            output_price_per_m: float,
                            tile: int = 512,
                            base: int = 70,
                            per_tile: int = 140) -> Tuple[int, float]:
    """Return (estimated_input_tokens_total, estimated_cost_usd) for one image."""
    img_tok = estimate_image_tokens(width, height, tile=tile, base=base, per_tile=per_tile)
    in_tok = img_tok + int(text_tokens)
    cost = (in_tok / 1e6) * float(input_price_per_m) + (int(out_tokens) / 1e6) * float(output_price_per_m)
    return in_tok, cost


def human_usd(x: float) -> str:
    return f"$ {x:,.2f}"

# ==================== EXTRACTION HELPERS ====================

def infer_product_id(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    return stem or "unknown"


def to_data_url(file_bytes: bytes, filename: str) -> str:
    """Convert image bytes to a base64 data URL to send inline to the Responses API."""
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def robust_json_from_text(txt: str, product_id: str) -> Dict[str, Any]:
    """Parse JSON even if the model returns extra text around it."""
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            data.setdefault("product_id", product_id)
            return data
    except Exception:
        pass
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = txt[start:end+1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                data.setdefault("product_id", product_id)
                return data
        except Exception:
            pass
    return {"product_id": product_id, "raw": txt.strip()}


INSTRUCTION = (
    "Extract all nutrition and related values you can read from this label into a single JSON object.\n"
    "- Include per-100g/ml and per-serving if present.\n"
    "- Use numbers for numeric fields (no units inside numbers).\n"
    "- If an item is not printed on the label, omit it (do not guess).\n"
    "- Include a top-level 'product_id' with the provided value.\n"
    "- Return JSON only (no prose, no markdown)."
)


def extract_from_image_bytes_freeform(
    client: OpenAI,
    model_name: str,
    img_bytes: bytes,
    filename: str,
    product_id: str
) -> Tuple[Dict[str, Any], Tuple[int, int]]:
    """
    Responses API (no tools, no response_format). Ask for plain JSON in text.
    Returns: (payload_dict, (input_tokens, output_tokens)) — tokens may be (0,0) if SDK omits usage.
    """
    # Normalize orientation; re-encode to JPEG
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
        temperature=0
    )

    # Try to read token usage
    in_tok = out_tok = 0
    try:
        u = getattr(resp, "usage", None)
        if u:
            in_tok = int(getattr(u, "input_tokens", 0) or (u.get("input_tokens") if isinstance(u, dict) else 0))
            out_tok = int(getattr(u, "output_tokens", 0) or (u.get("output_tokens") if isinstance(u, dict) else 0))
    except Exception:
        pass
    if (in_tok, out_tok) == (0, 0):
        try:
            raw = resp.model_dump()
            u = raw.get("usage", {}) if isinstance(raw, dict) else {}
            in_tok = int(u.get("input_tokens", 0))
            out_tok = int(u.get("output_tokens", 0))
        except Exception:
            pass

    # Parse text
    txt = getattr(resp, "output_text", "") or ""
    if not txt and hasattr(resp, "model_dump"):
        try:
            raw = resp.model_dump()
            txt = raw.get("output_text", "") or ""
            if not txt:
                for item in raw.get("output", []) or []:
                    for c in item.get("content", []) or []:
                        if c.get("type") in ("output_text", "text") and c.get("text"):
                            txt = c["text"]
                            break
                    if txt:
                        break
        except Exception:
            txt = ""

    return robust_json_from_text(txt or "{}", product_id), (in_tok, out_tok)


def pretty_json_block(d: Dict[str, Any]) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2)


def flatten_record(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dicts with dotted keys; lists -> JSON strings."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten_record(v, key, sep=sep))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


# ==================== SIDEBAR SETTINGS ====================
with st.sidebar:
    st.subheader("Settings")
    model = st.selectbox("Model", options=["gpt-4o"], index=0, help="Vision-capable model.")
    max_items = st.number_input("Max images to process", min_value=1, max_value=500, value=500, step=1)

    st.markdown("---")
    st.subheader("Cost estimator")
    input_price_per_m = st.number_input("Input price per 1M tokens (USD)", value=5.00, min_value=0.0, step=0.10, format="%.2f")
    output_price_per_m = st.number_input("Output price per 1M tokens (USD)", value=20.00, min_value=0.0, step=0.10, format="%.2f")
    text_tokens = st.number_input("Assumed prompt tokens per image", value=200, min_value=0, step=50)
    out_tokens_assumed = st.number_input("Assumed output tokens per image", value=300, min_value=0, step=50)

    with st.expander("Advanced token model", expanded=False):
        tile = st.number_input("Tile size (px)", value=512, min_value=256, step=64)
        base_tokens = st.number_input("Base tokens per image", value=70, min_value=0, step=10)
        per_tile_tokens = st.number_input("Tokens per tile", value=140, min_value=0, step=10)

# ==================== MAIN UI ====================
uploads = st.file_uploader(
    "Upload nutrition label images (JPG/PNG/WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="uploader",
)

if uploads:
    st.write(f"Selected **{len(uploads)}** file(s). Filenames will be used as product IDs.")

# ---- Cost estimation button ----
estimate = st.button("Estimate run cost", use_container_width=True)
if estimate:
    if not uploads:
        st.warning("Please upload images first.")
    else:
        total_in_tok = 0
        total_cost = 0.0
        dims: List[Tuple[str, int, int]] = []
        for up in uploads:
            try:
                img = Image.open(io.BytesIO(up.getvalue()))
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                dims.append((up.name, w, h))
                in_tok, cost = estimate_cost_for_image(
                    w, h, text_tokens, out_tokens_assumed,
                    input_price_per_m, output_price_per_m,
                    tile=tile, base=base_tokens, per_tile=per_tile_tokens
                )
                total_in_tok += in_tok
                total_cost += cost
            except Exception:
                dims.append((up.name, -1, -1))
        st.success(
            f"Estimated input tokens: {total_in_tok:,}  •  Estimated total cost: {human_usd(total_cost)}"
        )
        with st.expander("Details (image dimensions)"):
            for name, w, h in dims:
                st.write(f"{name}: {w}×{h} px")

run = st.button("Extract nutrition values", type="primary", use_container_width=True)

if run:
    if not uploads:
        st.stop()

    outputs: List[str] = []
    payloads: List[Dict[str, Any]] = []
    flat_payloads: List[Dict[str, Any]] = []

    total = min(len(uploads), max_items)
    progress = st.progress(0.0, text="Starting…")

    sum_in_tokens = 0
    sum_out_tokens = 0

    for idx, up in enumerate(uploads[:total], start=1):
        filename = up.name
        product_id = infer_product_id(filename)
        try:
            img_bytes = up.getvalue()
            payload, (in_tok, out_tok) = extract_from_image_bytes_freeform(client, model, img_bytes, filename, product_id)
            sum_in_tokens += in_tok
            sum_out_tokens += out_tok

            payloads.append(payload)
            flat_payloads.append(flatten_record(payload))

            block = pretty_json_block(payload)
            outputs.append(block + "\n---\n")
        except Exception as e:
            err = {"product_id": product_id, "error": str(e)}
            payloads.append(err)
            flat_payloads.append(flatten_record(err))
            outputs.append(pretty_json_block(err) + "\n---\n")
        finally:
            progress.progress(idx / max(1, total), text=f"Processed {idx} / {total}")
            time.sleep(0.05)

    # ---- TXT download ----
    final_text = "\n".join(outputs).strip()
    st.success("Extraction complete.")
    st.text_area("Preview (first 5,000 chars)", final_text[:5000], height=300)

    st.download_button(
        label="Download TXT",
        data=final_text.encode("utf-8"),
        file_name="nutrition_extractions.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # ---- CSV download ----
    # Union of all CSV columns
    fieldnames: List[str] = []
    seen = set()
    for rec in flat_payloads:
        for k in rec.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for rec in flat_payloads:
        # Ensure every field exists; missing -> empty string
        row = {k: rec.get(k, "") for k in fieldnames}
        writer.writerow(row)

    st.download_button(
        label="Download CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="nutrition_extractions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ---- Actual cost (if usage was available) ----
    actual_cost = (sum_in_tokens / 1e6) * float(input_price_per_m) + (sum_out_tokens / 1e6) * float(output_price_per_m)
    st.info(
        f"Actual usage (reported by API where available): input tokens = {sum_in_tokens:,}, output tokens = {sum_out_tokens:,}.\n"
        f"Approximate cost at current rates: {human_usd(actual_cost)}"
    )

    st.caption("Tip: keep images sharp and square-on for best results. Resize to ~1024–1600 px long edge for good accuracy vs cost.")
