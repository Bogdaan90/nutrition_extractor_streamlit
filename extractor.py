from __future__ import annotations
import os, io, json, time, base64, textwrap
from typing import Dict, Any, List

import streamlit as st
from PIL import Image
from openai import OpenAI

# --------------------------- UI SETUP ---------------------------
st.set_page_config(page_title="Nutrition Extractor (Image ➜ Text)", layout="centered")
st.title("Nutrition Extractor (Image ➜ Text)")
st.caption("Drag & drop images of Nutrition Information Panels. Get clean text back.")

# --------------------------- AUTH VIA UI ---------------------------
# Users enter their API key in the sidebar; stored only in session memory for this run.
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

with st.sidebar:
    st.subheader("Authentication")
    st.session_state["api_key"] = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="",
        help="Stored in session memory only; not written to disk."
    )

api_key = st.session_state["api_key"]
if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to enable extraction.")
    st.stop()

# Create the client with the user-provided key
client = OpenAI(api_key=api_key)

# --------------------------- JSON SCHEMA ---------------------------
# This schema is enforced via Structured Outputs so the model must return valid JSON.
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

# Tool definition (function calling) — use your JSON Schema as "parameters"
NUTRITION_TOOL = [{
    "type": "function",
    "function": {
        "name": "NutritionPanel",
        "description": "Extract nutrition panel values as strict JSON.",
        "parameters": SCHEMA["schema"]  # the JSON Schema object you already have
    }
}]


SYSTEM_PROMPT = (
    "You are an expert at reading EU/UK nutrition labels.\n"
    "- Read the nutrition information panel (NIP) from the image.\n"
    "- Report per 100g/ml and per serving if present.\n"
    "- Return ONLY JSON per the schema. Numbers only, no units inside fields.\n"
    "- If sodium is given, include sodium; if salt is shown, include salt as well.\n"
)

# --------------------------- HELPERS ---------------------------

def infer_product_id(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    return stem or "unknown"


def upload_image_to_openai(file_bytes: bytes, filename: str) -> str:
    """Upload image to OpenAI Files API and return file_id."""
    # Streamlit's UploadedFile is file-like; convert to BytesIO
    bio = io.BytesIO(file_bytes)
    uploaded = client.files.create(file=(filename, bio), purpose="user_data")
    return uploaded.id


def extract_from_image_file(client, model_name: str, file_id: str, product_id: str):
    resp = client.responses.create(
        model=model_name,                # e.g., "gpt-4o"
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": SYSTEM_PROMPT + f"Product ID: {product_id}"},
                {"type": "input_image",
                 "image_file": {"file_id": file_id}}
            ]
        }],
        tools=NUTRITION_TOOL,
        # Force the model to call our function (no free-form prose)
        tool_choice={"type": "function", "function": {"name": "NutritionPanel"}},
        temperature=0
    )

    # --- Parse tool call arguments robustly ---
    # SDKs return typed objects; fall back to dict if needed.
    try:
        # Happy path: walk typed structure
        for item in resp.output:
            if not getattr(item, "content", None):
                continue
            for c in item.content:
                if getattr(c, "type", "") in ("tool_call", "function_call"):
                    args = c.tool_call.function.arguments  # stringified JSON
                    return json.loads(args)
    except Exception:
        pass

    # Fallback: dump to dict and traverse keys
    raw = getattr(resp, "model_dump", lambda: {})()
    if not raw:
        raw = json.loads(getattr(resp, "json", "{}")) if hasattr(resp, "json") else {}

    # Common raw shapes: output -> [ { content: [ { type: "tool_call", tool_call:{function:{arguments:"..."}} } ] } ]
    outputs = raw.get("output", [])
    for item in outputs:
        for c in item.get("content", []):
            if c.get("type") in ("tool_call", "function_call"):
                return json.loads(c["tool_call"]["function"]["arguments"])

    # If for some reason the model didn’t call the tool (shouldn’t happen with tool_choice forced),
    # last resort: try to parse any text output as JSON
    text = getattr(resp, "output_text", "") or ""
    return json.loads(text)



def as_human_text(p: Dict[str, Any]) -> str:
    """Format one extraction block into readable text."""
    b = p.get("basis", {})
    e = p.get("energy", {})
    fat = p.get("fat", {})
    carb = p.get("carbohydrate", {})
    fib = p.get("fibre", {})
    pro = p.get("protein", {})
    salt = p.get("salt", {})

    lines = [
        f"Product_ID: {p.get('product_id','')}",
        f"Panel_Detected: {p.get('panel_detected', False)}",
        f"Basis: per_100g={b.get('per_100g')}, per_serving={b.get('per_serving')}, serving_size={b.get('serving_size')}",
        f"Per_100g: Energy {e.get('kJ_100')} kJ / {e.get('kcal_100')} kcal; "
        f"Fat {fat.get('g_100')} g (sat {fat.get('saturates_g_100')} g); "
        f"Carb {carb.get('g_100')} g (sugars {carb.get('sugars_g_100')} g); "
        f"Fibre {fib.get('g_100')}; Protein {pro.get('g_100')} g; Salt {salt.get('g_100')} g",
    ]

    if (e.get("kcal_serv") is not None) or (fat.get("g_serv") is not None):
        lines.append(
            f"Per_Serving: Energy {e.get('kJ_serv')} kJ / {e.get('kcal_serv')} kcal; "
            f"Fat {fat.get('g_serv')} g (sat {fat.get('saturates_g_serv')} g); "
            f"Carb {carb.get('g_serv')} g (sugars {carb.get('sugars_g_serv')} g); "
            f"Fibre {fib.get('g_serv')}; Protein {pro.get('g_serv')} g; Salt {salt.get('g_serv')} g"
        )

    rp = p.get("ri_percent", {})
    if any(v is not None for v in rp.values()):
        lines.append(
            "RI_Percent: "
            f"kcal={rp.get('kcal_serv')}%, fat={rp.get('fat')}%, sat={rp.get('saturates')}%, "
            f"carb={rp.get('carb')}%, sugars={rp.get('sugars')}%, protein={rp.get('protein')}%, salt={rp.get('salt')}%"
        )

    if p.get("notes"):
        lines.append("Notes: " + "; ".join(p["notes"]))

    return "\n".join(lines)


# --------------------------- APP BODY ---------------------------
with st.sidebar:
    st.subheader("Settings")
    model = st.selectbox("Model", options=["gpt-4o"], index=0, help="Vision-capable model for extraction.")
    max_items = st.number_input("Max images to process", min_value=1, max_value=500, value=500, step=1)
    st.markdown("---")
    st.caption("Images are uploaded as temporary files to the API for processing.")

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
    progress = st.progress(0.0, text="Starting…")

    for idx, up in enumerate(uploads[:max_items], start=1):
        filename = up.name
        product_id = infer_product_id(filename)
        try:
            # 1) upload each image to OpenAI file storage
            file_id = upload_image_to_openai(up.read(), filename)
            # 2) extract via Responses API (Structured Outputs)
            payload = extract_from_image_file(file_id, product_id)
            payload["source_image"] = filename
            # 3) render as human-readable text block
            outputs.append(as_human_text(payload) + "\n---\n")
        except Exception as e:
            outputs.append(f"Product_ID: {product_id}\nERROR: {e}\n---\n")
        finally:
            progress.progress(idx / max(1, min(len(uploads), max_items)), text=f"Processed {idx} / {min(len(uploads), max_items)}")
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

    st.caption("Tip: Keep images sharp and square-on for best results.")
