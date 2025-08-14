from __future__ import annotations
import io
import os
import json
import time
import base64
import mimetypes
from typing import Dict, Any, List

import streamlit as st
from PIL import Image  
from openai import OpenAI

# ============== PAGE SETUP ==============
st.set_page_config(page_title="Nutrition Extractor (Image → Text)", layout="centered")
st.title("Nutrition Extractor (Image → Text)")
st.caption("Upload nutrition label images, extract values as plain text. No cloud infra required.")

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

# ============== JSON SCHEMA (STRICT) ==============
SCHEMA: Dict[str, Any] = {
    "name": "NutritionPanel",
    "schema": {
        "type": "object",
        "properties": {
            "product_id": {"type": "string"},
            "panel_detected": {"type": "boolean"},
            "basis": {
                "type": "object",
                "properties": {
                    "per_100g": {"type": "boolean"},
                    "per_serving": {"type": "boolean"},
                    "serving_size": {"type": ["string", "null"]}
                },
                "required": ["per_100g", "per_serving", "serving_size"]
            },
            "energy": {
                "type": "object",
                "properties": {
                    "kJ_100": {"type": "number"},
                    "kcal_100": {"type": "number"},
                    "kJ_serv": {"type": ["number", "null"]},
                    "kcal_serv": {"type": ["number", "null"]}
                },
                "required": ["kJ_100", "kcal_100", "kJ_serv", "kcal_serv"]
            },
            "fat": {
                "type": "object",
                "properties": {
                    "g_100": {"type": "number"},
                    "g_serv": {"type": ["number", "null"]},
                    "saturates_g_100": {"type": ["number", "null"]},
                    "saturates_g_serv": {"type": ["number", "null"]}
                },
                "required": ["g_100", "g_serv", "saturates_g_100", "saturates_g_serv"]
            },
            "carbohydrate": {
                "type": "object",
                "properties": {
                    "g_100": {"type": "number"},
                    "g_serv": {"type": ["number", "null"]},
                    "sugars_g_100": {"type": ["number", "null"]},
                    "sugars_g_serv": {"type": ["number", "null"]}
                },
                "required": ["g_100", "g_serv", "sugars_g_100", "sugars_g_serv"]
            },
            "fibre": {
                "type": "object",
                "properties": {
                    "g_100": {"type": ["number", "null"]},
                    "g_serv": {"type": ["number", "null"]}
                },
                "required": ["g_100", "g_serv"]
            },
            "protein": {
                "type": "object",
                "properties": {
                    "g_100": {"type": "number"},
                    "g_serv": {"type": ["number", "null"]}
                },
                "required": ["g_100", "g_serv"]
            },
            "salt": {
                "type": "object",
                "properties": {
                    "g_100": {"type": ["number", "null"]},
                    "g_serv": {"type": ["number", "null"]},
                    "sodium_g_100": {"type": ["number", "null"]},
                    "sodium_g_serv": {"type": ["number", "null"]}
                },
                "required": ["g_100", "g_serv", "sodium_g_100", "sodium_g_serv"]
            },
            "ri_percent": {
                "type": "object",
                "properties": {
                    "kcal_serv": {"type": ["number", "null"]},
                    "fat": {"type": ["number", "null"]},
                    "saturates": {"type": ["number", "null"]},
                    "carb": {"type": ["number", "null"]},
                    "sugars": {"type": ["number", "null"]},
                    "protein": {"type": ["number", "null"]},
                    "salt": {"type": ["number", "null"]}
                },
                "required": ["kcal_serv", "fat", "saturates", "carb", "sugars", "protein", "salt"]
            },
            "notes": {"type": "array", "items": {"type": "string"}},
            "model_confidence": {"type": "number"},
            "source_image": {"type": "string"}
        },
        "required": [
            "product_id", "panel_detected", "basis", "energy", "fat",
            "carbohydrate", "protein", "salt", "ri_percent", "notes",
            "model_confidence", "source_image"
        ],
        "additionalProperties": False
    },
    "strict": True
}

# Tool (function) definition for strict structured output
NUTRITION_TOOL = [{
    "type": "function",
    "name": "NutritionPanel",
    "description": "Extract nutrition panel values as strict JSON.",
    "parameters": SCHEMA["schema"]   # reuse your existing JSON Schema object
}]

SYSTEM_PROMPT = (
    "You are an expert at reading EU/UK nutrition labels.\n"
    "- Read the nutrition information panel (NIP) from the image.\n"
    "- Report per 100g/ml and per serving if present.\n"
    "- Return only JSON conforming to the function schema. Numbers only, no units in fields.\n"
    "- If sodium is present, include sodium; if salt is shown, include salt as well.\n"
)

# ============== HELPERS ==============
def infer_product_id(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    return stem or "unknown"

def to_data_url(file_bytes: bytes, filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_from_image_bytes(
    client: OpenAI,
    model_name: str,
    img_bytes: bytes,
    filename: str,
    product_id: str
) -> Dict[str, Any]:
    """
    Extract nutrition data from an image using the Responses API and tool/function calling.

    Prereqs in your module:
      - SCHEMA: the JSON Schema dict (with keys "name", "schema", "strict")
      - NUTRITION_TOOL: [{"type":"function","name":"NutritionPanel","description":"...","parameters": SCHEMA["schema"]}]
      - SYSTEM_PROMPT: instruction string for the model
      - to_data_url(bytes, filename): returns a base64 data URL string for the image
    """
    data_url = to_data_url(img_bytes, filename)

    resp = client.responses.create(
        model=model_name,  # e.g., "gpt-4o"
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": SYSTEM_PROMPT + f"Product ID: {product_id}"},
                {"type": "input_image",
                 "image_url": data_url}
            ]
        }],
        tools=NUTRITION_TOOL,
        # Force a single function/tool call so we always get structured JSON back
        tool_choice={"type": "function", "name": "NutritionPanel"},
        temperature=0
    )

    # ---- Parse tool call arguments (strict JSON per your schema) ----
    # 1) Typed-object path (preferred)
    try:
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") in ("tool_call", "function_call"):
                    tc = getattr(c, "tool_call", None)
                    if tc is not None:
                        # Different SDK builds expose either .function.arguments or .arguments
                        func = getattr(tc, "function", None)
                        args = (
                            getattr(func, "arguments", None)
                            if func is not None else None
                        ) or getattr(tc, "arguments", None)
                        if args:
                            return json.loads(args)
    except Exception:
        pass

    # 2) Raw-dict path (fallback)
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

    outputs = raw.get("output", [])
    for item in outputs or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("tool_call", "function_call"):
                tc = c.get("tool_call", {}) or {}
                args = (tc.get("function", {}) or {}).get("arguments") or tc.get("arguments")
                if args:
                    return json.loads(args)

    # 3) Last resort (shouldn’t trigger with tool_choice forced)
    text = getattr(resp, "output_text", "") or raw.get("output_text", "") or "{}"
    return json.loads(text)


def as_human_text(p: Dict[str, Any]) -> str:
    b = p.get("basis", {})
    e = p.get("energy", {})
    fat = p.get("fat", {})
    carb = p.get("carbohydrate", {})
    fib = p.get("fibre", {})
    pro = p.get("protein", {})
    salt = p.get("salt", {})
    rp = p.get("ri_percent", {})

    lines = [
        f"Product_ID: {p.get('product_id','')}",
        f"Panel_Detected: {p.get('panel_detected', False)}",
        f"Basis: per_100g={b.get('per_100g')}, per_serving={b.get('per_serving')}, serving_size={b.get('serving_size')}",
        (
            "Per_100g: "
            f"Energy {e.get('kJ_100')} kJ / {e.get('kcal_100')} kcal; "
            f"Fat {fat.get('g_100')} g (sat {fat.get('saturates_g_100')} g); "
            f"Carb {carb.get('g_100')} g (sugars {carb.get('sugars_g_100')} g); "
            f"Fibre {fib.get('g_100')}; Protein {pro.get('g_100')} g; Salt {salt.get('g_100')} g"
        ),
    ]

    if (e.get("kcal_serv") is not None) or (fat.get("g_serv") is not None) or (carb.get("g_serv") is not None) or (pro.get("g_serv") is not None) or (salt.get("g_serv") is not None):
        lines.append(
            "Per_Serving: "
            f"Energy {e.get('kJ_serv')} kJ / {e.get('kcal_serv')} kcal; "
            f"Fat {fat.get('g_serv')} g (sat {fat.get('saturates_g_serv')} g); "
            f"Carb {carb.get('g_serv')} g (sugars {carb.get('sugars_g_serv')} g); "
            f"Fibre {fib.get('g_serv')}; Protein {pro.get('g_serv')} g; Salt {salt.get('g_serv')} g"
        )

    if any(v is not None for v in rp.values()) if isinstance(rp, dict) else False:
        lines.append(
            "RI_Percent: "
            f"kcal={rp.get('kcal_serv')}%, fat={rp.get('fat')}%, sat={rp.get('saturates')}%, "
            f"carb={rp.get('carb')}%, sugars={rp.get('sugars')}%, protein={rp.get('protein')}%, salt={rp.get('salt')}%"
        )

    if p.get("notes"):
        lines.append("Notes: " + "; ".join(p["notes"]))

    return "\n".join(lines)

# ============== SIDEBAR SETTINGS ==============
with st.sidebar:
    st.subheader("Settings")
    model = st.selectbox("Model", options=["gpt-4o"], index=0, help="Vision-capable model for extraction.")
    max_items = st.number_input("Max images to process", min_value=1, max_value=500, value=500, step=1)

# ============== MAIN UI ==============
uploads = st.file_uploader(
    "Upload nutrition label images (JPG/PNG/WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="uploader",
)

if uploads:
    st.write(f"Selected **{len(uploads)}** file(s). Filenames will be used as Product IDs.")

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
            payload = extract_from_image_bytes(client, model, img_bytes, filename, product_id)
            payload["source_image"] = filename
            outputs.append(as_human_text(payload) + "\n---\n")
        except Exception as e:
            outputs.append(f"Product_ID: {product_id}\nERROR: {e}\n---\n")
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

    st.caption("Tip: keep images sharp and square-on for best results.")
