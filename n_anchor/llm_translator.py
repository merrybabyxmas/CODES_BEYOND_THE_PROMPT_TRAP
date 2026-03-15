#!/usr/bin/env python3
"""
LLM-based Phase 1 & 2 Automation for N-Anchor Pipeline
=======================================================
Uses GPT-4o to automatically extract anchors and translate prompts.

Phase 1: Global Narrative Anchoring
  A_entity, A_bg = Ψ_extract(N)

Phase 2: Visual Hallucination Control (two modes)
  full_prompt   = Ψ_translate(s_t | A)  ← identity + action (for text_concat)
  action_prompt = Ψ_action(s_t | A_bg)  ← action only (for n_anchor, identity via attention)

Reproducibility:
  - temperature=0, seed=42 for deterministic outputs
  - All LLM responses saved to JSON (subsequent runs load from cache)
  - NO proxy in code — direct OpenAI API calls only

Usage:
  python llm_translator.py [--force_regenerate]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set. "
        "Run: export OPENAI_API_KEY='sk-your-key-here'"
    )

MODEL = "gpt-4o"
TEMPERATURE = 0       # Deterministic
SEED = 42             # Reproducibility
DATA_DIR = Path(__file__).parent / "data"
SCENES_FILE = DATA_DIR / "novel_scenes.json"
OUTPUT_FILE = DATA_DIR / "llm_narratives.json"

# Scene IDs to use (filtered for quality)
SELECTED_SCENE_IDS = [
    "pg19_001_scene00",   # Alice in Wonderland
    "pg19_002_scene00",   # Frankenstein - Arctic
    "pg19_002_scene01",   # Frankenstein - Inn
    "pg19_000_scene02",   # Pride and Prejudice
    "pg19_003_scene01",   # Sherlock Holmes
]

# ============================================================
# SYSTEM PROMPTS
# ============================================================
PHASE1_SYSTEM = """You are an expert visual director analyzing novel text to extract visual anchors for multi-shot video generation.

Given a sequence of 5 consecutive sentences from a novel, extract TWO anchors:

1. ENTITY: A detailed PHYSICAL description of the main character who appears most across the 5 sentences. Include: approximate age, gender, hair color/style, clothing, distinguishing features. Be specific enough that a video model could consistently render this person across multiple shots.

2. BACKGROUND: A detailed description of the primary SETTING/ENVIRONMENT. Include: location type, lighting conditions, key objects, atmosphere, color palette. This should remain consistent across all 5 shots.

IMPORTANT: Base your descriptions ONLY on what can be reasonably inferred from the text. Do not hallucinate details that contradict the source material.

Output ONLY valid JSON (no markdown, no explanation):
{"entity": "...", "background": "..."}"""

PHASE2_FULL_SYSTEM = """You are an expert visual director translating novel sentences into video generation prompts.

Given an anchor (character description + background) and a raw novel sentence, create a COMPLETE video generation prompt that includes:
1. The character's PHYSICAL APPEARANCE (from the anchor entity)
2. The character's ACTION (from the raw sentence)
3. The SCENE/SETTING (from the anchor background)

Rules:
- Write in present tense, describing what is visually happening
- If the sentence is dialogue or internal thought, infer a logical physical action
- Keep the prompt under 60 words
- End with ", cinematic, high quality"

Output ONLY the translated prompt string, nothing else."""

PHASE2_ACTION_SYSTEM = """You are an expert visual director translating novel sentences into video generation prompts.

Given a background context and a raw novel sentence, create an ACTION-FOCUSED video generation prompt.

CRITICAL RULES:
1. DO NOT describe the main character's physical appearance (hair, clothing, age, etc.) — this is handled by a separate visual system
2. Refer to characters generically: "the girl", "the man", "the captain", etc.
3. FOCUS on: physical actions, body language, camera movement, scene dynamics
4. Include the background/setting context
5. Keep the prompt under 50 words
6. End with ", cinematic, high quality"

Output ONLY the translated prompt string, nothing else."""


# ============================================================
# LLM CLIENT
# ============================================================
def call_gpt4o(system_prompt, user_prompt, retries=3):
    """Call GPT-4o with retry logic. No proxy."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                seed=SEED,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# ============================================================
# PHASE 1: ANCHOR EXTRACTION
# ============================================================
def extract_anchor(sentences):
    """Extract entity and background anchors from a scene's sentences."""
    scene_text = "\n".join(f"Sentence {i+1}: {s}" for i, s in enumerate(sentences))
    user_prompt = f"Novel scene (5 consecutive sentences):\n{scene_text}"

    raw = call_gpt4o(PHASE1_SYSTEM, user_prompt)

    # Parse JSON response
    try:
        # Handle potential markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        anchor = json.loads(raw)
        assert "entity" in anchor and "background" in anchor
        return anchor
    except (json.JSONDecodeError, AssertionError) as e:
        print(f"  WARNING: Failed to parse anchor JSON: {raw[:200]}")
        print(f"  Error: {e}")
        return {"entity": raw, "background": ""}


# ============================================================
# PHASE 2: PROMPT TRANSLATION
# ============================================================
def translate_full(sentence, anchor):
    """Translate sentence to full prompt (identity + action) for text_concat."""
    user_prompt = (
        f"- Anchor Entity: {anchor['entity']}\n"
        f"- Anchor Background: {anchor['background']}\n"
        f"- Raw Sentence: {sentence}\n\n"
        f"Translated Prompt:"
    )
    return call_gpt4o(PHASE2_FULL_SYSTEM, user_prompt)


def translate_action(sentence, anchor_bg):
    """Translate sentence to action-only prompt (no identity) for n_anchor."""
    user_prompt = (
        f"- Background Context: {anchor_bg}\n"
        f"- Raw Sentence: {sentence}\n\n"
        f"Translated Prompt:"
    )
    return call_gpt4o(PHASE2_ACTION_SYSTEM, user_prompt)


# ============================================================
# MAIN PIPELINE
# ============================================================
def process_scenes(force=False):
    """Process all selected scenes through LLM pipeline."""

    # Check cache
    if OUTPUT_FILE.exists() and not force:
        print(f"Loading cached LLM narratives from {OUTPUT_FILE}")
        with open(OUTPUT_FILE) as f:
            return json.load(f)

    # Load extracted scenes
    if not SCENES_FILE.exists():
        print("Novel scenes not found. Run extract_novel_scenes.py first.")
        sys.exit(1)

    with open(SCENES_FILE) as f:
        all_scenes = json.load(f)

    # Filter to selected scenes
    scenes = [s for s in all_scenes if s['id'] in SELECTED_SCENE_IDS]
    print(f"Processing {len(scenes)} scenes through GPT-4o pipeline...")

    narratives = []
    for scene_idx, scene in enumerate(scenes):
        print(f"\n[{scene_idx+1}/{len(scenes)}] {scene['book_title']}")

        # Clean sentences (remove \r, normalize whitespace)
        clean_sentences = [
            s.replace('\r', ' ').replace('  ', ' ').strip()
            for s in scene['sentences']
        ]

        # Phase 1: Extract anchor
        print("  Phase 1: Extracting anchor...")
        anchor = extract_anchor(clean_sentences)
        print(f"    Entity: {anchor['entity'][:80]}...")
        print(f"    Background: {anchor['background'][:80]}...")

        # Phase 2: Translate prompts (both modes)
        full_prompts = []
        action_prompts = []
        for i, sent in enumerate(clean_sentences):
            print(f"  Phase 2: Translating shot {i+1}/5...")

            fp = translate_full(sent, anchor)
            full_prompts.append(fp)
            print(f"    [full] {fp[:80]}...")

            ap = translate_action(sent, anchor['background'])
            action_prompts.append(ap)
            print(f"    [action] {ap[:80]}...")

        narrative = {
            'id': scene['id'].replace('pg19_', 'novel_'),
            'title': scene['book_title'],
            'source': scene['source'],
            'anchor': anchor,
            'raw_sentences': clean_sentences,
            'full_prompts': full_prompts,
            'action_prompts': action_prompts,
            'llm_model': MODEL,
            'llm_temperature': TEMPERATURE,
            'llm_seed': SEED,
        }
        narratives.append(narrative)

    # Save to cache
    DATA_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(narratives, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(narratives)} LLM-generated narratives to {OUTPUT_FILE}")

    return narratives


def main():
    parser = argparse.ArgumentParser(
        description="LLM Phase 1+2 for N-Anchor pipeline")
    parser.add_argument('--force', action='store_true',
                        help='Force regenerate even if cache exists')
    args = parser.parse_args()

    narratives = process_scenes(force=args.force)

    # Print summary
    print("\n" + "=" * 70)
    print("LLM-GENERATED NARRATIVES SUMMARY")
    print("=" * 70)
    for n in narratives:
        print(f"\n[{n['id']}] {n['title']}")
        print(f"  Anchor Entity: {n['anchor']['entity'][:100]}...")
        print(f"  Anchor Background: {n['anchor']['background'][:100]}...")
        for i in range(len(n['raw_sentences'])):
            print(f"  Shot {i+1}:")
            print(f"    Raw:    {n['raw_sentences'][i][:80]}...")
            print(f"    Full:   {n['full_prompts'][i][:80]}...")
            print(f"    Action: {n['action_prompts'][i][:80]}...")
    print("=" * 70)


if __name__ == '__main__':
    main()
