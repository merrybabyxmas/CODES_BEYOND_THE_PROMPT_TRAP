#!/usr/bin/env python3
"""
Save shot gallery: organize all mid-frames + raw text + prompts
into a human-readable directory structure for paper figures.

Output structure:
  shot_gallery/
    {story_id}/
      raw_text.txt          ← original novel sentences
      anchor.txt            ← extracted anchor descriptions
      prompts.txt           ← all prompt variants (full, action)
      shots_comparison.txt  ← side-by-side summary
      baseline/             ← mid-frame PNGs
      text_concat/
      n_anchor_concat/
      n_anchor_dual_cosine/
      n_anchor_dual_linear/
      n_anchor_dual_step/
"""

import json
import shutil
from pathlib import Path

EXP_DIRS = {
    'llm_exp': Path(__file__).parent / 'outputs' / 'llm_exp',
    'main_exp': Path(__file__).parent / 'outputs' / 'main_exp',
}
GALLERY_DIR = Path(__file__).parent / 'outputs' / 'shot_gallery'
CONDITIONS = ['baseline', 'text_concat', 'n_anchor_concat',
              'n_anchor_dual_cosine', 'n_anchor_dual_linear',
              'n_anchor_dual_step']


def load_narratives():
    """Load all narrative data."""
    all_narratives = []

    # LLM narratives
    llm_path = Path(__file__).parent / 'data' / 'llm_narratives.json'
    if llm_path.exists():
        with open(llm_path) as f:
            all_narratives.extend(json.load(f))

    # Synthetic narratives (from run_experiment.py)
    try:
        from run_experiment import NARRATIVES
        all_narratives.extend(NARRATIVES)
    except ImportError:
        pass

    return all_narratives


def save_gallery():
    narratives = load_narratives()
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    for story in narratives:
        story_id = story['id']
        story_dir = GALLERY_DIR / story_id
        story_dir.mkdir(exist_ok=True)

        # 1. Save raw text
        with open(story_dir / 'raw_text.txt', 'w') as f:
            f.write(f"Title: {story.get('title', story_id)}\n")
            f.write(f"Source: {story.get('source', 'N/A')}\n")
            f.write("=" * 60 + "\n\n")
            for i, sent in enumerate(story['raw_sentences']):
                f.write(f"Shot {i+1}: {sent}\n\n")

        # 2. Save anchor
        with open(story_dir / 'anchor.txt', 'w') as f:
            f.write(f"Entity Anchor:\n{story['anchor']['entity']}\n\n")
            f.write(f"Background Anchor:\n{story['anchor']['background']}\n")

        # 3. Save all prompt variants
        with open(story_dir / 'prompts.txt', 'w') as f:
            for i in range(len(story['raw_sentences'])):
                f.write(f"{'='*60}\nShot {i+1}\n{'='*60}\n\n")
                f.write(f"[RAW NARRATIVE]\n{story['raw_sentences'][i]}\n\n")
                if 'full_prompts' in story:
                    f.write(f"[FULL PROMPT (text_concat)]\n{story['full_prompts'][i]}\n\n")
                if 'action_prompts' in story:
                    f.write(f"[ACTION PROMPT (n_anchor)]\n{story['action_prompts'][i]}\n\n")
                if 'translated_prompts' in story:
                    f.write(f"[TRANSLATED PROMPT]\n{story['translated_prompts'][i]}\n\n")

        # 4. Copy mid-frame PNGs from each condition
        for exp_name, exp_dir in EXP_DIRS.items():
            for cond in CONDITIONS:
                src_dir = exp_dir / cond / story_id
                if not src_dir.exists():
                    continue

                dst_dir = story_dir / cond
                dst_dir.mkdir(exist_ok=True)

                for shot_idx in range(len(story['raw_sentences'])):
                    # Copy mid-frame PNG
                    src_png = src_dir / f'shot_{shot_idx:02d}_mid.png'
                    if src_png.exists():
                        shutil.copy2(src_png, dst_dir / f'shot_{shot_idx:02d}.png')

                    # Copy video
                    src_mp4 = src_dir / f'shot_{shot_idx:02d}.mp4'
                    if src_mp4.exists():
                        shutil.copy2(src_mp4, dst_dir / f'shot_{shot_idx:02d}.mp4')

        # 5. Summary comparison
        with open(story_dir / 'shots_comparison.txt', 'w') as f:
            f.write(f"SHOT COMPARISON: {story.get('title', story_id)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Anchor Entity: {story['anchor']['entity']}\n")
            f.write(f"Anchor Background: {story['anchor']['background']}\n\n")

            for i in range(len(story['raw_sentences'])):
                f.write(f"{'─'*80}\n")
                f.write(f"SHOT {i+1}\n")
                f.write(f"{'─'*80}\n")
                f.write(f"Raw:    {story['raw_sentences'][i]}\n")
                if 'action_prompts' in story:
                    f.write(f"Action: {story['action_prompts'][i]}\n")
                f.write(f"\nFrames saved in: {story_id}/{{condition}}/shot_{i:02d}.png\n\n")

        print(f"Saved gallery for {story_id}")

    print(f"\nGallery saved to {GALLERY_DIR}")
    print(f"Total stories: {len(narratives)}")


if __name__ == '__main__':
    save_gallery()
