#!/usr/bin/env python3
"""
PG-19 Real Novel Scene Extraction for N-Anchor Experiments
==========================================================
Extracts 5-shot scene chunks from real published novels (Project Gutenberg)
for the "Wild Novel Text" benchmark.

Reproducibility:
  - Fixed random seed for scene selection
  - Deterministic filtering criteria
  - All extracted scenes saved to JSON for experiment use

Output:
  - novel_scenes.json: Extracted scenes ready for N-Anchor experiment
"""

import json
import random
import sys
from pathlib import Path

import spacy

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)

# Scene extraction parameters
NUM_SHOTS = 5
MIN_WORDS = 8       # Filter out very short sentences ("Yes." "No.")
MAX_WORDS = 35      # Filter out overly long paragraphs
NUM_BOOKS = 20      # How many books to scan
SCENES_PER_BOOK = 3 # Max scenes per book
TOTAL_SCENES = 10   # Target number of scenes for experiment

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
# Increase max length for novels
nlp.max_length = 100000


def extract_valid_scenes(text, book_title="", num_shots=NUM_SHOTS):
    """
    Extract scene chunks from novel text.
    Each scene is NUM_SHOTS consecutive sentences suitable for video generation.

    Filtering criteria:
    - Each sentence has 8-35 words (not too short/long)
    - No sentences that are pure dialogue (starting with quotes)
    - At least 2 sentences should contain action verbs or visual descriptions
    - Consecutive sentences (narrative continuity)
    """
    # Parse first 50K chars (memory efficient)
    text_chunk = text[:50000]
    # Skip header (Project Gutenberg boilerplate)
    start_markers = ["CHAPTER", "Chapter", "PART", "I.", "I\n"]
    for marker in start_markers:
        idx = text_chunk.find(marker)
        if 500 < idx < 5000:
            text_chunk = text_chunk[idx:]
            break

    doc = nlp(text_chunk)
    sentences = [sent.text.strip().replace('\n', ' ').replace('  ', ' ')
                 for sent in doc.sents]

    valid_chunks = []

    for i in range(len(sentences) - num_shots):
        chunk = sentences[i:i + num_shots]

        # Filter 1: Word count bounds
        word_counts = [len(s.split()) for s in chunk]
        if not all(MIN_WORDS <= wc <= MAX_WORDS for wc in word_counts):
            continue

        # Filter 2: Skip chunks that are mostly dialogue
        dialogue_count = sum(1 for s in chunk
                           if s.startswith('"') or s.startswith("'")
                           or s.startswith('\u201c'))
        if dialogue_count > 2:
            continue

        # Filter 3: At least some visual/action content
        visual_keywords = ['walked', 'stood', 'sat', 'looked', 'turned',
                          'opened', 'closed', 'ran', 'moved', 'held',
                          'wore', 'carried', 'entered', 'left', 'crossed',
                          'eyes', 'face', 'hand', 'door', 'room', 'light',
                          'dark', 'sun', 'sky', 'road', 'house', 'window',
                          'garden', 'forest', 'river', 'morning', 'night']
        visual_count = sum(1 for s in chunk
                         if any(kw in s.lower() for kw in visual_keywords))
        if visual_count < 2:
            continue

        valid_chunks.append({
            'sentences': chunk,
            'start_idx': i,
            'visual_score': visual_count,
        })

    # Sort by visual score (most visual scenes first)
    valid_chunks.sort(key=lambda x: x['visual_score'], reverse=True)
    return valid_chunks[:SCENES_PER_BOOK]


def main():
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)

    # PG-19 loading script deprecated; download directly from Project Gutenberg
    # These are all PUBLIC DOMAIN novels (copyright expired)
    GUTENBERG_BOOKS = [
        (1342, "Pride and Prejudice - Jane Austen"),
        (11, "Alice's Adventures in Wonderland - Lewis Carroll"),
        (84, "Frankenstein - Mary Shelley"),
        (1661, "The Adventures of Sherlock Holmes - Arthur Conan Doyle"),
        (98, "A Tale of Two Cities - Charles Dickens"),
        (2701, "Moby Dick - Herman Melville"),
        (174, "The Picture of Dorian Gray - Oscar Wilde"),
        (1952, "The Yellow Wallpaper - Charlotte Perkins Gilman"),
        (345, "Dracula - Bram Stoker"),
        (1232, "The Prince - Niccolo Machiavelli"),
        (46, "A Christmas Carol - Charles Dickens"),
        (76, "Adventures of Huckleberry Finn - Mark Twain"),
        (120, "Treasure Island - Robert Louis Stevenson"),
        (16328, "Beowulf"),
        (5200, "Metamorphosis - Franz Kafka"),
        (2591, "Grimms Fairy Tales"),
        (1400, "Great Expectations - Charles Dickens"),
        (219, "Heart of Darkness - Joseph Conrad"),
        (244, "A Study in Scarlet - Arthur Conan Doyle"),
        (43, "The Strange Case of Dr Jekyll and Mr Hyde - Stevenson"),
    ]

    print(f"Downloading {len(GUTENBERG_BOOKS)} classic novels from "
          f"Project Gutenberg (public domain)...")
    import urllib.request

    books = []
    for book_id, title in GUTENBERG_BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'NanchorResearch/1.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode('utf-8', errors='ignore')
            if len(text) > 5000:
                books.append({'text': text, 'title': title, 'id': book_id})
                print(f"  [{len(books)}/{len(GUTENBERG_BOOKS)}] {title} "
                      f"({len(text)//1000}K chars)")
        except Exception as e:
            print(f"  SKIP {title}: {e}")
        if len(books) >= NUM_BOOKS:
            break

    print(f"  Downloaded {len(books)} books successfully")

    all_scenes = []
    books_scanned = 0

    for book_idx, book in enumerate(books):
        book_text = book['text']
        book_title = book['title']

        if len(book_text) < 10000:
            continue

        scenes = extract_valid_scenes(book_text, book_title)
        books_scanned += 1

        for scene_idx, scene in enumerate(scenes):
            scene_id = f"pg19_{book_idx:03d}_scene{scene_idx:02d}"
            all_scenes.append({
                'id': scene_id,
                'source': 'PG-19',
                'book_idx': book_idx,
                'book_title': str(book_title),
                'sentences': scene['sentences'],
                'visual_score': scene['visual_score'],
            })

        if scenes:
            print(f"  Book {book_idx} ({book_title}): "
                  f"{len(scenes)} scenes extracted")

        if len(all_scenes) >= TOTAL_SCENES:
            break

    print(f"\nScanned {books_scanned} books, "
          f"extracted {len(all_scenes)} scenes total")

    # Select final scenes (top by visual score, then random sample)
    if len(all_scenes) > TOTAL_SCENES:
        all_scenes.sort(key=lambda x: x['visual_score'], reverse=True)
        all_scenes = all_scenes[:TOTAL_SCENES]

    # Print extracted scenes
    print("\n" + "=" * 70)
    print("EXTRACTED REAL NOVEL SCENES (Wild Text)")
    print("=" * 70)
    for scene in all_scenes:
        print(f"\n[{scene['id']}] from '{scene['book_title']}'")
        for i, sent in enumerate(scene['sentences']):
            print(f"  Shot {i+1}: {sent}")

    # Save to JSON
    output_path = output_dir / 'novel_scenes.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_scenes, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_scenes)} scenes to {output_path}")

    return all_scenes


if __name__ == '__main__':
    main()
