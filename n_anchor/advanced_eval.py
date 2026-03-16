#!/usr/bin/env python3
"""
Advanced Evaluation Metrics for N-Anchor Experiments
=====================================================
Three metrics beyond standard CLIP-I/CLIP-TI:

1. VQA Score (GPT-4o Vision) — narrative fidelity via Yes/No questions
2. Optical Flow Magnitude — motion dynamism measurement
3. Conditional CLIP-I — entity consistency only where character should appear

Reproducibility:
  - All GPT-4o responses cached to JSON
  - Optical Flow computed locally (no API)
  - Fixed evaluation parameters

Usage:
  python advanced_eval.py --exp_dir ./outputs/llm_exp --narratives llm
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# CONFIG
# ============================================================
API_KEY = os.environ.get("OPENAI_API_KEY", "")
CACHE_DIR = Path(__file__).parent / "data" / "eval_cache"

# ============================================================
# 1. OPTICAL FLOW (Local — no API)
# ============================================================
def compute_optical_flow_magnitude(video_path):
    """
    Compute average optical flow magnitude across all frame pairs.
    Higher value = more dynamic motion in the video.

    Uses Farneback dense optical flow (OpenCV).
    Returns: float (average flow magnitude)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_magnitudes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitudes.append(magnitude.mean())
        prev_gray = gray

    cap.release()
    return float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0


def evaluate_optical_flow(exp_dir, narratives, conditions):
    """Compute optical flow for all videos."""
    cache_path = CACHE_DIR / "optical_flow.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
    else:
        cached = {}

    results = {}
    for condition in conditions:
        flows = []
        for story in narratives:
            for shot_idx in range(len(story['raw_sentences'])):
                key = f"{condition}/{story['id']}/shot_{shot_idx:02d}"
                if key in cached:
                    flow_val = cached[key]
                else:
                    video_path = exp_dir / condition / story['id'] / f"shot_{shot_idx:02d}.mp4"
                    if video_path.exists():
                        flow_val = compute_optical_flow_magnitude(video_path)
                        cached[key] = flow_val
                        logging.info(f"  OF {key}: {flow_val:.4f}")
                    else:
                        flow_val = 0.0
                flows.append(flow_val)

        results[condition] = {
            'mean': float(np.mean(flows)),
            'std': float(np.std(flows)),
            'n': len(flows),
        }

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cached, f, indent=2)

    return results


# ============================================================
# 2. VQA SCORE (GPT-4o Vision)
# ============================================================
def encode_image_base64(image_path):
    """Encode image to base64 for GPT-4o Vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_vqa_questions(narratives):
    """
    Generate VQA questions for each shot using GPT-4o.
    Returns dict: {story_id: [{question, expected_answer}, ...]}
    Cached to JSON.
    """
    cache_path = CACHE_DIR / "vqa_questions.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    system_prompt = """You are generating Yes/No visual verification questions for video frames.

Given a novel sentence and the main character's description, generate exactly 2 questions:

1. ACTION_Q: A question about whether the specific ACTION described in the sentence is visually depicted.
2. ENTITY_Q: A question about whether a specific ENTITY (character/object) mentioned in the sentence is visible.

Also provide the expected correct answer (Yes/No) for each.

Output ONLY valid JSON (no markdown):
{"action_q": "...", "action_expected": "Yes/No", "entity_q": "...", "entity_expected": "Yes/No"}"""

    all_questions = {}
    for story in narratives:
        story_qs = []
        entity_desc = story['anchor']['entity']

        for i, sent in enumerate(story['raw_sentences']):
            user_prompt = (
                f"Main character: {entity_desc}\n"
                f"Novel sentence (Shot {i+1}): {sent}\n"
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0, seed=42, max_tokens=200,
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                qa = json.loads(raw)
                story_qs.append(qa)
                logging.info(f"  VQA Q generated: {story['id']}/shot{i}")
            except Exception as e:
                logging.warning(f"  VQA Q error: {e}")
                story_qs.append({
                    "action_q": f"Does the frame show the scene described: {sent[:50]}?",
                    "action_expected": "Yes",
                    "entity_q": f"Is the main character visible?",
                    "entity_expected": "Yes",
                })
            time.sleep(0.5)

        all_questions[story['id']] = story_qs

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(all_questions, f, indent=2)

    return all_questions


def ask_gpt4o_vision(image_path, question):
    """Ask GPT-4o Vision a Yes/No question about an image."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    img_b64 = encode_image_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"Answer ONLY 'Yes' or 'No' to this question about the image:\n"
                    f"{question}")},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "low"}}
            ]
        }],
        temperature=0, seed=42, max_tokens=5,
    )
    answer = response.choices[0].message.content.strip()
    return "Yes" if answer.lower().startswith("yes") else "No"


def evaluate_vqa(exp_dir, narratives, conditions):
    """Run VQA evaluation using GPT-4o Vision."""
    cache_path = CACHE_DIR / "vqa_answers.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
    else:
        cached = {}

    # Generate questions first
    questions = generate_vqa_questions(narratives)

    results = {}
    for condition in conditions:
        correct_action = 0
        correct_entity = 0
        total = 0

        for story in narratives:
            story_qs = questions.get(story['id'], [])
            for shot_idx, qa in enumerate(story_qs):
                key = f"{condition}/{story['id']}/shot_{shot_idx:02d}"
                frame_path = exp_dir / condition / story['id'] / f"shot_{shot_idx:02d}_mid.png"

                if not frame_path.exists():
                    continue

                # Check cache
                if key in cached:
                    answers = cached[key]
                else:
                    try:
                        action_ans = ask_gpt4o_vision(frame_path, qa['action_q'])
                        entity_ans = ask_gpt4o_vision(frame_path, qa['entity_q'])
                        answers = {
                            'action_answer': action_ans,
                            'entity_answer': entity_ans,
                        }
                        cached[key] = answers
                        logging.info(f"  VQA {key}: action={action_ans}, entity={entity_ans}")
                        time.sleep(0.3)
                    except Exception as e:
                        logging.warning(f"  VQA error {key}: {e}")
                        continue

                # Score
                if answers['action_answer'] == qa.get('action_expected', 'Yes'):
                    correct_action += 1
                if answers['entity_answer'] == qa.get('entity_expected', 'Yes'):
                    correct_entity += 1
                total += 1

        results[condition] = {
            'action_accuracy': correct_action / max(total, 1),
            'entity_accuracy': correct_entity / max(total, 1),
            'combined_accuracy': (correct_action + correct_entity) / max(2 * total, 1),
            'total': total,
        }

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cached, f, indent=2)

    return results


# ============================================================
# 3. CONDITIONAL CLIP-I
# ============================================================
def detect_character_presence(narratives):
    """
    Determine which shots should have the main character visible.
    Uses GPT-4o to classify each sentence.
    Cached to JSON.
    """
    cache_path = CACHE_DIR / "character_presence.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    system_prompt = """Given a sentence from a novel and the main character's description, determine:
Is the main character PHYSICALLY PRESENT and VISIBLE in this scene?

Answer ONLY "Yes" or "No".
- "Yes" if the character is performing actions, being described, or physically present
- "No" if the scene focuses on other characters/objects, or the character is absent/sleeping/offscreen"""

    presence_map = {}
    for story in narratives:
        shots = []
        for i, sent in enumerate(story['raw_sentences']):
            user_prompt = (
                f"Main character: {story['anchor']['entity']}\n"
                f"Sentence: {sent}"
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0, seed=42, max_tokens=5,
                )
                ans = response.choices[0].message.content.strip()
                present = ans.lower().startswith("yes")
                shots.append(present)
                logging.info(f"  Presence {story['id']}/shot{i}: {present}")
            except Exception as e:
                shots.append(True)  # Default to present
                logging.warning(f"  Presence error: {e}")
            time.sleep(0.3)

        presence_map[story['id']] = shots

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(presence_map, f, indent=2)

    return presence_map


@torch.no_grad()
def evaluate_conditional_clip_i(exp_dir, narratives, conditions):
    """
    Compute CLIP-I only between consecutive shots where BOTH have
    the main character present. Ignores cutaway transitions.
    """
    import clip as clip_module

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip_module.load("ViT-B/32", device=device)
    clip_model.eval()

    presence_map = detect_character_presence(narratives)

    results = {}
    for condition in conditions:
        all_sims = []

        for story in narratives:
            presence = presence_map.get(story['id'], [True] * 5)

            # Load frames
            frames = []
            for shot_idx in range(len(story['raw_sentences'])):
                frame_path = exp_dir / condition / story['id'] / f"shot_{shot_idx:02d}_mid.png"
                if frame_path.exists():
                    frames.append(Image.open(frame_path).convert('RGB'))
                else:
                    frames.append(None)

            # Encode all frames
            embeddings = []
            for frame in frames:
                if frame is not None:
                    img = preprocess(frame).unsqueeze(0).to(device)
                    emb = clip_model.encode_image(img)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embeddings.append(emb)
                else:
                    embeddings.append(None)

            # Compute conditional similarity (only where both shots have character)
            for i in range(len(embeddings) - 1):
                if (i < len(presence) and i + 1 < len(presence)
                        and presence[i] and presence[i + 1]
                        and embeddings[i] is not None
                        and embeddings[i + 1] is not None):
                    sim = F.cosine_similarity(embeddings[i], embeddings[i + 1])
                    all_sims.append(sim.item())

        results[condition] = {
            'mean': float(np.mean(all_sims)) if all_sims else 0,
            'std': float(np.std(all_sims)) if all_sims else 0,
            'n': len(all_sims),
        }

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Advanced N-Anchor Evaluation")
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--narratives', type=str, default='llm',
                        choices=['synthetic', 'novel', 'llm'])
    parser.add_argument('--conditions', type=str, nargs='+',
                        default=['baseline', 'text_concat', 'n_anchor_concat',
                                 'n_anchor_dual_cosine', 'n_anchor_dual_linear',
                                 'n_anchor_dual_step'])
    parser.add_argument('--skip_vqa', action='store_true',
                        help='Skip VQA (requires API key)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    exp_dir = Path(args.exp_dir)

    # Load narratives
    if args.narratives == 'llm':
        data_path = Path(__file__).parent / "data" / "llm_narratives.json"
        with open(data_path) as f:
            narratives = json.load(f)
    elif args.narratives == 'synthetic':
        from run_experiment import NARRATIVES
        narratives = NARRATIVES
    elif args.narratives == 'novel':
        from novel_narratives import NOVEL_NARRATIVES
        narratives = NOVEL_NARRATIVES

    conditions = args.conditions
    all_results = {}

    # ── Metric 1: Optical Flow ──
    logging.info("=" * 60)
    logging.info("Computing Optical Flow (motion dynamism)...")
    of_results = evaluate_optical_flow(exp_dir, narratives, conditions)
    all_results['optical_flow'] = of_results

    # ── Metric 2: VQA Score ──
    if not args.skip_vqa and API_KEY:
        logging.info("=" * 60)
        logging.info("Computing VQA Score (GPT-4o Vision)...")
        vqa_results = evaluate_vqa(exp_dir, narratives, conditions)
        all_results['vqa_score'] = vqa_results
    else:
        logging.info("Skipping VQA (no API key or --skip_vqa)")

    # ── Metric 3: Conditional CLIP-I ──
    logging.info("=" * 60)
    logging.info("Computing Conditional CLIP-I...")
    if API_KEY:
        cclip_results = evaluate_conditional_clip_i(
            exp_dir, narratives, conditions)
    else:
        # Without API, use all shots (no character presence filtering)
        logging.info("  No API key — computing standard CLIP-I instead")
        cclip_results = evaluate_conditional_clip_i(
            exp_dir, narratives, conditions)
    all_results['conditional_clip_i'] = cclip_results

    # ── Save Results ──
    results_path = exp_dir / "advanced_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nResults saved to {results_path}")

    # ── Print Summary ──
    print("\n" + "=" * 70)
    print("ADVANCED EVALUATION RESULTS")
    print("=" * 70)

    print("\n--- Optical Flow (motion dynamism, ↑ higher = more dynamic) ---")
    print(f"{'Condition':<25} {'Mean':>8} {'Std':>8} {'N':>5}")
    print("-" * 50)
    for cond, vals in of_results.items():
        print(f"{cond:<25} {vals['mean']:>8.4f} {vals['std']:>8.4f} {vals['n']:>5}")

    if 'vqa_score' in all_results:
        print("\n--- VQA Score (GPT-4o Vision, ↑ higher = better narrative fidelity) ---")
        print(f"{'Condition':<25} {'Action':>8} {'Entity':>8} {'Combined':>8} {'N':>5}")
        print("-" * 60)
        for cond, vals in all_results['vqa_score'].items():
            print(f"{cond:<25} {vals['action_accuracy']:>8.4f} "
                  f"{vals['entity_accuracy']:>8.4f} "
                  f"{vals['combined_accuracy']:>8.4f} {vals['total']:>5}")

    print("\n--- Conditional CLIP-I (character-present shots only, ↑ higher) ---")
    print(f"{'Condition':<25} {'Mean':>8} {'Std':>8} {'N':>5}")
    print("-" * 50)
    for cond, vals in cclip_results.items():
        print(f"{cond:<25} {vals['mean']:>8.4f} {vals['std']:>8.4f} {vals['n']:>5}")

    print("=" * 70)


if __name__ == '__main__':
    main()
