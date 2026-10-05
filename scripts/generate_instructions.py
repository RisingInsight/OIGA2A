"""
LLM-based instruction generation for A2A V2 open-instruction system.

Generates diverse, natural, domain-expert cross-modal translation instructions
for all 12 modality pairs. Instructions are grounded in FLAIR-HUB paper domain
knowledge (Garioud et al., 2025) and remote sensing expert language.

Three instruction categories:
  - L1 (Attribute): Describe target modality's physical properties without naming it
  - L2 (Task-driven): Describe the application/task need, grounded in land cover/crop mapping
  - L3 (Style/Perception): Describe desired visual effect or output characteristics

CRITICAL: Instructions must be "open" (semantic/task-level), NOT "fine-grained" (pixel-level).
Open instructions = novel contribution. Fine-grained instructions = ChatGPT-4o territory.

Usage:
    python scripts/generate_instructions.py --api_key YOUR_KEY --api_base https://api.openai.com/v1
    python scripts/generate_instructions.py --api_key YOUR_KEY --api_base https://api.anthropic.com/v1 --model claude-sonnet-4-20250514
"""

import argparse
import json
import os
import re
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    exit(1)


# ─── Modality definitions (FLAIR-HUB grounded) ───────────────────────

MODALITIES = {
    "rgb": {
        "name": "Aerial RGB",
        "flair_name": "AERIAL RGBI",
        "description": "True-color aerial photography at 0.2m VHR resolution. 3-channel (R,G,B), UInt8 [0,255]. "
                       "From IGN ORTHO HR product — radiometric equalization applied, relative calibration. "
                       "No clouds by design. Shows visible-spectrum appearance of the landscape.",
        "physical_properties": "Visible-spectrum reflectance (400-700nm), relative radiometric calibration, "
                               "0.20m ground sampling distance, 512x512 patches",
        "applications": "Land cover semantic segmentation (19 COSIA classes: building, greenhouse, swimming pool, "
                        "impervious/pervious surface, bare soil, water, snow, herbaceous vegetation, agricultural land, "
                        "plowed land, vineyard, deciduous, coniferous, brushwood), "
                        "urban monitoring, OCS-GE national land cover mapping, soil land take assessment",
        "visual_characteristics": "Photorealistic, high-resolution, true colors, sharp edges, shadows visible, "
                                  "radiometric equalization across domains",
        "expert_terminology": ["VHR orthophoto", "ORTHO HR", "relative radiometric calibration",
                               "visible-spectrum reflectance", "aerial survey mosaic", "photo-interpretation"],
    },
    "nir": {
        "name": "Near-Infrared (NIR)",
        "flair_name": "AERIAL RGBI (NIR channel)",
        "description": "Single-band near-infrared aerial imagery replicated to 3 channels. 0.2m VHR resolution. "
                       "Captured simultaneously with RGB by the same airborne sensor (Infrared-Red-Green bands). "
                       "Sensitive to chlorophyll reflectance (vegetation appears bright), water absorption, soil moisture.",
        "physical_properties": "Near-infrared reflectance (700-1000nm), chlorophyll absorption edge, "
                               "0.20m GSD, temporally coherent with RGB (same aerial survey)",
        "applications": "Vegetation health assessment (NDVI computation), crop stress detection, "
                        "water body delineation, forest inventory (deciduous vs coniferous discrimination), "
                        "wetland mapping, drought monitoring, herbaceous vegetation identification",
        "visual_characteristics": "Grayscale/monochromatic, vegetation appears bright white, water appears dark, "
                                  "uniform grey-scale, no color differentiation, high contrast for vegetation",
        "expert_terminology": ["chlorophyll reflectance", "near-infrared band", "NDVI",
                               "vegetation index", "canopy density", "photosynthetically active vegetation"],
    },
    "dem": {
        "name": "Digital Elevation Model (DEM)",
        "flair_name": "DEM ELEV",
        "description": "3-channel elevation data: DSM (digital surface model), DTM (digital terrain model), "
                       "and nDSM (normalized DSM = DSM - DTM, object height). 0.2m resolution, Float32, "
                       "absolute elevation in meters. DSM from dense matching of aerial images (vertical accuracy ~0.4m). "
                       "DTM from RGE ALTI national product (1m resolution, vertical accuracy 0.3-7m).",
        "physical_properties": "Absolute altitude in meters, DSM-DTM differencing for object height, "
                               "0.20m GSD (DSM), 1m GSD (DTM), vertical accuracy ~0.4m",
        "applications": "Terrain analysis, flood modeling, slope calculation, building height estimation, "
                        "forestry canopy height (nDSM), urban planning (building footprints from DSM-DTM), "
                        "drainage design, viewshed analysis, DSM-DTM differencing for land cover height proxy",
        "visual_characteristics": "Colorless structural map, geometric height representation, no texture/color, "
                                  "topographic contours, 3D-like terrain appearance, height gradients",
        "expert_terminology": ["DSM", "DTM", "nDSM", "digital surface model", "digital terrain model",
                               "normalized DSM", "dense matching photogrammetry", "RGE ALTI",
                               "vertical accuracy", "elevation tensor"],
    },
    "spot": {
        "name": "SPOT Satellite RGB",
        "flair_name": "SPOT RGBI",
        "description": "SPOT 6-7 satellite multispectral imagery (R,G,B channels). 1.6m resolution, "
                       "resampled from 1.50m. UInt16 Level-2A BOA (bottom-of-atmosphere) reflectance × 10000. "
                       "Annual mosaics, same year as aerial imagery. Orbital perspective, coarser than aerial.",
        "physical_properties": "BOA reflectance (absolute calibration), 1.60m GSD, "
                               "from SPOT 6-7 annual mosaics, UInt16 [0-10000] reflectance scale",
        "applications": "Large-area land cover mapping, regional vegetation monitoring, "
                        "agricultural inventory (crop type discrimination at parcel level), "
                        "change detection, satellite-based classification, SPOT-aerial cross-modal synthesis",
        "visual_characteristics": "Slightly blurred compared to aerial, coarser spatial detail, "
                                  "muted spectral response, orbital/satellite perspective, broader coverage sense, "
                                  "absolute radiometric calibration (BOA reflectance)",
        "expert_terminology": ["BOA reflectance", "bottom-of-atmosphere", "SPOT 6-7",
                               "orbital observation", "satellite multispectral", "Level-2A product",
                               "annual mosaic", "spaceborne sensor"],
    },
}

# ─── FLAIR-HUB Domain Knowledge ──────────────────────────────────────

FLAIR_HUB_CONTEXT = """
## FLAIR-HUB Dataset Context (Garioud et al., 2025)

FLAIR-HUB is a large-scale multimodal dataset for land cover and crop mapping over France.
It contains 6 spatially aligned modalities across 74 spatio-temporal domains.

### Land Cover Classes (LABEL-COSIA, 19 classes):
- Anthropized: Building, Greenhouse, Swimming pool, Impervious surface, Pervious surface
- Natural: Bare soil, Water, Snow, Herbaceous vegetation
- Agricultural: Agricultural land, Plowed land, Vineyard
- Forest: Deciduous, Coniferous, Brushwood, Clear cut, Ligneous, Mixed

### Crop Type Taxonomy (LABEL-LPIS, 29 Level-1 classes):
- Cereals: Wheat (winter/spring), Barley (winter/spring), Maize, Rice, Sorghum
- Oilseeds: Sunflower, Rapeseed, Soy
- Legumes: Alfalfa, Clover, Spring peas
- Root crops: Beetroots, Potatoes
- Permanent: Vineyard, Olive groves, Fruit orchards, Nut orchards
- Others: Hemp, Flax, Lavandin, Berries

### Key Applications:
- Land cover semantic segmentation (COSIA 19-class)
- Crop type mapping (LPIS 29-class hierarchy)
- Soil land take monitoring (EU regulation)
- Forest monitoring (deciduous/coniferous discrimination)
- DSM-DTM differencing for building height estimation
- Cross-modal super-resolution (Sentinel-2 → SPOT → Aerial)
- Domain adaptation across 74 geographic/temporal domains
"""

# ─── Modality pairs ──────────────────────────────────────────────────

MODALITY_PAIRS = [
    ("rgb", "nir"), ("rgb", "dem"), ("rgb", "spot"),
    ("nir", "rgb"), ("nir", "dem"), ("nir", "spot"),
    ("dem", "rgb"), ("dem", "nir"), ("dem", "spot"),
    ("spot", "rgb"), ("spot", "nir"), ("spot", "dem"),
]

# ─── Prompt templates ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a remote sensing expert specializing in Earth Observation (EO), land cover mapping,
and multimodal satellite/aerial image analysis. You work with the FLAIR-HUB dataset — a large-scale multimodal
dataset covering 74 spatio-temporal domains over France with 6 aligned modalities (aerial RGB, NIR, DEM, SPOT,
Sentinel-2 time series, Sentinel-1 SAR).

Your task is to generate diverse, natural-language instructions for cross-modal image translation.
These instructions train a vision-language model that takes a source image and a text instruction,
then produces the target modality image.

You understand:
- Land cover classification (19 COSIA classes: building, greenhouse, impervious/pervious surface,
  bare soil, water, snow, herbaceous vegetation, agricultural land, plowed land, vineyard,
  deciduous, coniferous, brushwood)
- Crop type mapping (29 LPIS Level-1 classes: wheat, barley, maize, sunflower, rapeseed, vineyard, etc.)
- Physical properties of each modality (chlorophyll reflectance, DSM-DTM differencing, BOA reflectance)
- Remote sensing applications (soil land take monitoring, forest inventory, flood risk, crop stress)

CRITICAL REQUIREMENTS:
1. Instructions must be OPEN (semantic/task-level) — describe WHAT you need the data for, not HOW to convert
2. NEVER produce fine-grained pixel-level instructions like "brighten the vegetation in the upper-left"
3. Instructions should sound like a remote sensing expert or agronomist making a data request
4. Use domain-specific terminology: chlorophyll absorption, canopy density, DSM-DTM differencing,
   BOA reflectance, VHR orthophoto, photo-interpretation, land cover segmentation
5. Each instruction should be 1-2 sentences, 10-30 words
6. Vary sentence structure, vocabulary, and framing
7. NEVER mention modality names directly (RGB, NIR, DEM, SPOT) — describe physical properties instead
8. NEVER use template patterns like "Convert", "Transform", "Translate" as sentence starters
9. Consider the SOURCE modality context — what information is available, its resolution, calibration type
10. Some instructions should be specific to this source→target pair, not generic across all pairs"""


def build_generation_prompt(src: str, tgt: str, num_instructions: int = 250) -> str:
    src_info = MODALITIES[src]
    tgt_info = MODALITIES[tgt]

    # Build expert terminology lists
    src_terms = ", ".join(src_info["expert_terminology"][:4])
    tgt_terms = ", ".join(tgt_info["expert_terminology"][:4])

    return f"""Generate exactly {num_instructions} diverse natural-language instructions for translating
from {src_info['name']} ({src_info['flair_name']}) to {tgt_info['name']} ({tgt_info['flair_name']}).

SOURCE modality ({src_info['flair_name']}):
  - Physical properties: {src_info['physical_properties']}
  - Visual appearance: {src_info['visual_characteristics']}
  - Key terminology: {src_terms}

TARGET modality ({tgt_info['flair_name']}):
  - Physical properties: {tgt_info['physical_properties']}
  - Visual appearance: {tgt_info['visual_characteristics']}
  - Applications: {tgt_info['applications']}
  - Key terminology: {tgt_terms}

{FLAIR_HUB_CONTEXT}

Generate instructions in FOUR categories (roughly equal distribution):

CATEGORY 1 — Attribute Description (describe target's physical properties):
  - Describe WHAT the output shows physically, not HOW to convert
  - Use physical/scientific terminology specific to the target modality
  - The reader should understand the target modality from the description alone
  - GOOD: "Reveal the chlorophyll reflectance patterns distinguishing deciduous from coniferous stands"
  - GOOD: "Show me the DSM-DTM height differential to identify building footprints"
  - BAD: "Convert this image to near-infrared" (too direct, mentions conversion)

CATEGORY 2 — Task-Driven (describe the application need grounded in FLAIR-HUB tasks):
  - Frame as a real-world remote sensing task: land cover mapping, crop monitoring, forest inventory
  - Reference FLAIR-HUB application domains: COSIA classification, LPIS crop mapping, soil land take
  - Should feel like a practitioner (agronomist, urban planner, forester) requesting specific data
  - GOOD: "I'm mapping winter wheat parcels for CAP subsidy compliance across this agricultural region"
  - GOOD: "Assessing building heights for urban densification monitoring in this peri-urban area"
  - BAD: "Make it look like a satellite image" (too vague, not task-driven)

CATEGORY 3 — Style/Perception Control (describe desired output characteristics):
  - Describe visual/structural properties of the desired output in domain terms
  - Mention resolution characteristics, contrast, emphasis areas relevant to remote sensing
  - Should guide HOW the output should look for the intended analysis task
  - GOOD: "Generate a monochromatic intensity map where vegetation stands out against bare soil boundaries"
  - GOOD: "Produce a height-colored terrain model emphasizing canopy structure over flat agricultural fields"
  - BAD: "Make it brighter" (too generic, no domain context)

CATEGORY 4 — Cross-Modal Context-Aware (reference source modality characteristics):
  - Acknowledge what information IS or IS NOT available in the source modality
  - Reference source's resolution, spectral coverage, calibration type as context for the task
  - The instruction should make sense ONLY for this specific source→target pair
  - GOOD (RGB→NIR): "Using the visible-spectrum color cues for vegetation detection, produce a near-infrared intensity map highlighting chlorophyll-rich canopy areas"
  - GOOD (SPOT→RGB): "Given the coarser orbital observation at 1.6m, reconstruct a high-resolution 0.2m true-color aerial view preserving parcel boundary details"
  - GOOD (DEM→NIR): "From the terrain height information showing building and tree elevations, synthesize a vegetation reflectance map distinguishing canopy from ground-level herbaceous cover"
  - BAD: "Convert this to another format" (no source context, works for any pair)

IMPORTANT RULES:
- Never use the words "RGB", "NIR", "DEM", "SPOT" in instructions
- Never use template patterns like "Convert", "Transform", "Translate" as sentence starters
- Use varied sentence starters: "Show me...", "I need...", "Help me identify...", "Generate...",
  "Produce...", "Create...", "Reveal...", "Highlight...", "Emphasize...", "Focus on...",
  "I'm assessing...", "For monitoring...", "To map...", "Delineate..."
- Each instruction should be self-contained and understandable without context
- Vary vocabulary extensively — do not repeat the same phrasing
- Instructions must be OPEN (task-level) not FINE-GRAINED (pixel-level)
- Ground instructions in real remote sensing applications and expert terminology

OUTPUT FORMAT: Return a JSON object with key "instructions" containing an array of objects, each with:
  - "instruction": the instruction text
  - "category": "attribute" | "task" | "style" | "cross_modal"

Return ONLY the JSON object, no other text."""


def generate_instructions(
    client: OpenAI,
    model: str,
    src: str,
    tgt: str,
    num_instructions: int = 250,
    max_retries: int = 3,
    batch_size: int = 100,
) -> list[dict]:
    """Generate instructions for a single modality pair.

    If num_instructions > batch_size, makes multiple API calls and merges results.
    """
    all_valid = []
    remaining = num_instructions

    while remaining > 0:
        this_batch = min(remaining, batch_size)
        batch_result = _generate_batch(client, model, src, tgt, this_batch, max_retries)
        all_valid.extend(batch_result)
        remaining -= len(batch_result)
        if len(batch_result) == 0:
            break  # API failed, stop trying
        time.sleep(1)

    return all_valid[:num_instructions]


def _generate_batch(
    client: OpenAI,
    model: str,
    src: str,
    tgt: str,
    num_instructions: int,
    max_retries: int,
) -> list[dict]:
    """Generate a single batch of instructions for a modality pair."""
    prompt = build_generation_prompt(src, tgt, num_instructions)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content

            # Parse JSON with fallback extraction for responses with extra text
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Use raw_decode to find the first valid JSON object/array
                decoder = json.JSONDecoder()
                try:
                    data, _ = decoder.raw_decode(content)
                except json.JSONDecodeError:
                    # Try regex for JSON object with nested arrays
                    match = re.search(r'\{[^{}]*\[[\s\S]*\][^{}]*\}', content)
                    if match:
                        data = json.loads(match.group())
                    else:
                        raise ValueError(f"No JSON found in response: {content[:200]}")

            # Handle both {"instructions": [...]} and [...] formats
            if isinstance(data, dict):
                instructions = data.get("instructions", data.get("data", []))
            elif isinstance(data, list):
                instructions = data
            else:
                raise ValueError(f"Unexpected response format: {type(data)}")

            # Validate
            valid = []
            for item in instructions:
                if isinstance(item, dict) and "instruction" in item:
                    item.setdefault("category", "attribute")
                    valid.append(item)
                elif isinstance(item, str):
                    valid.append({"instruction": item, "category": "attribute"})

            if len(valid) >= num_instructions * 0.5:  # At least 50% of requested
                print(f"  [{src}->{tgt}] Generated {len(valid)} instructions")
                return valid[:num_instructions]
            else:
                print(f"  [{src}->{tgt}] Only got {len(valid)} instructions, retrying...")

        except Exception as e:
            print(f"  [{src}->{tgt}] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    print(f"  [{src}->{tgt}] FAILED after {max_retries} attempts")
    return []


def main():
    parser = argparse.ArgumentParser(description="Generate cross-modal translation instructions using LLM")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--num_per_pair", type=int, default=250, help="Instructions per modality pair")
    parser.add_argument("--output", type=str, default="dataloaders/instructions_a2a_v2.json", help="Output JSON path")
    parser.add_argument("--pairs", type=str, nargs="*", default=None,
                        help="Specific pairs to generate (e.g., 'rgb_nir' 'dem_spot'). Default: all 12 pairs")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    # Parse pairs
    if args.pairs:
        pairs = [tuple(p.split("_")) for p in args.pairs]
    else:
        pairs = MODALITY_PAIRS

    all_instructions = {}
    total = 0

    print(f"Generating {args.num_per_pair} instructions per pair for {len(pairs)} pairs...")
    print(f"Model: {args.model}, API: {args.api_base}")
    print(f"Grounded in FLAIR-HUB domain knowledge (Garioud et al., 2025)\n")

    for src, tgt in pairs:
        key = f"{src}_to_{tgt}"
        instructions = generate_instructions(client, args.model, src, tgt, args.num_per_pair)
        all_instructions[key] = instructions
        total += len(instructions)
        time.sleep(1)  # Rate limiting

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_instructions, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {total} instructions for {len(pairs)} pairs")
    print(f"Saved to: {args.output}")

    # Print stats
    print("\nPer-pair statistics:")
    for key, instrs in all_instructions.items():
        categories = {}
        for inst in instrs:
            cat = inst.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        print(f"  {key}: {len(instrs)} total — {categories}")


if __name__ == "__main__":
    main()
