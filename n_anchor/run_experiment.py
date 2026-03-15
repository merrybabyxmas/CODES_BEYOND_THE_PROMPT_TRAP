#!/usr/bin/env python3
"""
N-Anchor: Training-Free Narrative-to-Video Generation via Global Context Anchoring
==================================================================================
Reproducible Experiment Script

Backbone: Wan2.1-T2V-1.3B
Conditions:
  1. baseline:        Raw narrative sentence → Wan2.1 (no anchor)
  2. text_concat:     "[Anchor]. [Prompt]" concatenated as single text → Wan2.1
  3. n_anchor_concat: Anchor K/V injection in cross-attention → Wan2.1

Evaluation:
  - Entity Consistency: CLIP-I cosine similarity between consecutive shot frames
  - Narrative Alignment: CLIP text-image similarity between prompt and generated frame

Reproducibility:
  - Fixed seed per (story, shot): seed = base_seed + story_idx * 1000 + shot_idx
  - All narratives/anchors/prompts pre-defined (NO API calls, NO proxy)
  - Deterministic noise generation with fixed torch.Generator

Reference Code:
  - Wan2.1: backbone model and video generation pipeline
  - VGoT (evaluate/code/calculate_clip.py): CLIP similarity computation
  - StoryMem (extract_keyframes.py): frame similarity function
  - Wan2.1 (utils/utils.py): cache_video for video saving

Usage:
  python run_experiment.py [--output_dir OUTPUT_DIR] [--num_shots NUM_SHOTS]
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
import types
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent.parent
WAN_DIR = str(BASE_DIR / 'related_git_repos' / 'Wan2.1')
VGOT_DIR = str(BASE_DIR / 'related_git_repos' / 'VideoGen-of-Thought')
STORYMEM_DIR = str(BASE_DIR / 'related_git_repos' / 'StoryMem')

# Add Wan2.1 to path
sys.path.insert(0, WAN_DIR)

# ============================================================
# FLASH ATTENTION FALLBACK
# ============================================================
# Wan2.1 model directly imports flash_attention. If flash_attn
# is not installed, we patch it with PyTorch's native SDPA.
import wan.modules.attention as _attn_module
import wan.modules.model as _model_module

if not (_attn_module.FLASH_ATTN_2_AVAILABLE or _attn_module.FLASH_ATTN_3_AVAILABLE):
    logging.warning("flash_attn not available, using PyTorch SDPA fallback")

    def _sdpa_fallback(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                       softmax_scale=None, q_scale=None, causal=False,
                       window_size=(-1, -1), deterministic=False,
                       dtype=torch.bfloat16, version=None):
        """Fallback using torch.nn.functional.scaled_dot_product_attention."""
        # q: [B, Lq, N, C], k: [B, Lk, N, C], v: [B, Lk, N, C]
        q = q.transpose(1, 2).to(dtype)  # [B, N, Lq, C]
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=dropout_p,
            scale=softmax_scale)
        return out.transpose(1, 2).contiguous()

    _attn_module.flash_attention = _sdpa_fallback
    _model_module.flash_attention = _sdpa_fallback

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.attention import attention as wan_attention
from wan.utils.utils import cache_video

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
DEFAULT_CONFIG = {
    'model_path': '/home/dongwoo43/papers/paper_DIAGONAL/codes/multi_shot_eval_repos/EchoShot_official/models/Wan2.1-T2V-1.3B',
    'task': 't2v-1.3B',
    'size': '832*480',
    'frame_num': 17,          # 4n+1, min reasonable
    'sampling_steps': 30,
    'guide_scale': 5.0,
    'shift': 5.0,
    'base_seed': 42,
    'num_shots': 5,           # shots per story
    'conditions': ['baseline', 'text_concat', 'n_anchor_concat'],
}

# ============================================================
# NARRATIVE DATA (Pre-computed — NO API, NO proxy)
# ============================================================
NARRATIVES = [
    {
        'id': 'story_01_painter',
        'title': 'The Painter',
        'anchor': {
            'entity': (
                'A young woman in her late 20s with long auburn hair tied in a messy bun, '
                'light freckles on her cheeks, wearing a paint-stained white cotton apron '
                'over a blue denim shirt, rolled-up sleeves'
            ),
            'background': (
                'A sunlit artist studio with large arched windows, a wooden easel at center, '
                'shelves lined with paint tubes and brushes, warm golden morning light, '
                'hardwood floor with colorful paint splatters'
            ),
        },
        'raw_sentences': [
            "Sarah opened her art studio early in the morning.",
            "She mixed colors on her palette with great concentration.",
            "The woman stepped back to examine her canvas critically.",
            "She smiled as the landscape painting started to come together.",
            "Sarah carefully signed her name in the corner of the finished work.",
        ],
        'translated_prompts': [
            ("A young woman with long auburn hair in a messy bun, wearing a paint-stained "
             "white apron over a blue denim shirt, pushes open the door of a sunlit artist "
             "studio with large arched windows, wooden easel, shelves of paint tubes, warm "
             "golden morning light streaming in, cinematic, high quality"),
            ("A young woman with long auburn hair in a messy bun, wearing a paint-stained "
             "white apron over a blue denim shirt, mixes vibrant oil colors on a wooden "
             "palette with intense concentration in a sunlit artist studio, natural light "
             "from large arched windows, cinematic, high quality"),
            ("A young woman with long auburn hair in a messy bun, wearing a paint-stained "
             "white apron over a blue denim shirt, steps back from a wooden easel and "
             "examines a landscape canvas with a critical eye in a sunlit artist studio, "
             "paint tubes on shelves behind her, cinematic, high quality"),
            ("A young woman with long auburn hair in a messy bun, wearing a paint-stained "
             "white apron over a blue denim shirt, smiles warmly while adding brushstrokes "
             "to a landscape painting on a wooden easel in a sunlit studio with large arched "
             "windows, cinematic, high quality"),
            ("A young woman with long auburn hair in a messy bun, wearing a paint-stained "
             "white apron over a blue denim shirt, carefully signs her name in the lower "
             "corner of a finished landscape painting on a wooden easel in a sunlit artist "
             "studio, cinematic, high quality"),
        ],
    },
    {
        'id': 'story_02_explorer',
        'title': 'The Explorer',
        'anchor': {
            'entity': (
                'A rugged man in his mid-30s with short dark brown hair and a neatly trimmed '
                'beard, wearing a khaki safari jacket with many pockets, brown leather hiking '
                'boots, carrying a worn leather backpack'
            ),
            'background': (
                'A lush green forest with tall ancient oak trees, dappled golden sunlight '
                'filtering through the canopy, moss-covered rocks, a narrow winding dirt '
                'trail, ferns and wildflowers along the path'
            ),
        },
        'raw_sentences': [
            "Marcus adjusted his backpack before entering the dense forest.",
            "He carefully crossed a narrow wooden bridge over a stream.",
            "The man discovered ancient stone ruins hidden among the trees.",
            "He took detailed notes in his leather journal about the discovery.",
            "Marcus set up camp near the ruins as the sun began to set.",
        ],
        'translated_prompts': [
            ("A rugged man with short dark brown hair and trimmed beard, wearing a khaki "
             "safari jacket with many pockets and brown hiking boots, adjusts his worn "
             "leather backpack at the entrance of a lush green forest with tall ancient oak "
             "trees and dappled golden sunlight, cinematic, high quality"),
            ("A rugged man with short dark brown hair and trimmed beard, wearing a khaki "
             "safari jacket with many pockets, carefully walks across a narrow wooden bridge "
             "over a flowing stream in a lush green forest with moss-covered rocks and ferns, "
             "dappled sunlight, cinematic, high quality"),
            ("A rugged man with short dark brown hair and trimmed beard, wearing a khaki "
             "safari jacket with many pockets, stands amazed before ancient stone ruins "
             "covered in moss and vines, hidden among tall oak trees in a lush green forest, "
             "dappled golden sunlight, cinematic, high quality"),
            ("A rugged man with short dark brown hair and trimmed beard, wearing a khaki "
             "safari jacket, sits on a mossy stone and writes detailed notes in a leather "
             "journal near ancient stone ruins in a green forest, warm dappled sunlight, "
             "cinematic, high quality"),
            ("A rugged man with short dark brown hair and trimmed beard, wearing a khaki "
             "safari jacket, sets up a small campsite with a tent near ancient stone ruins "
             "in a lush forest as warm golden sunset light filters through the canopy, "
             "cinematic, high quality"),
        ],
    },
    {
        'id': 'story_03_chef',
        'title': 'The Chef',
        'anchor': {
            'entity': (
                'A woman in her early 30s with black hair pulled back in a tight ponytail, '
                'wearing a traditional white double-breasted chef coat, a black apron tied '
                'at the waist, and a tall white chef hat (toque)'
            ),
            'background': (
                'A professional restaurant kitchen with polished stainless steel countertops, '
                'hanging copper pots and pans, bright overhead fluorescent lighting, organized '
                'spice racks on the wall, gas burners with blue flames'
            ),
        },
        'raw_sentences': [
            "Elena arrived at the restaurant kitchen before dawn.",
            "She began preparing the day's special sauce with fresh herbs.",
            "The chef tasted the dish and nodded with satisfaction.",
            "She arranged the plates with artistic precision for the lunch service.",
            "Elena watched proudly as guests enjoyed her culinary creation.",
        ],
        'translated_prompts': [
            ("A woman with black hair in a tight ponytail, wearing a white double-breasted "
             "chef coat, black apron and tall white chef hat, walks into a professional "
             "restaurant kitchen with stainless steel countertops, hanging copper pots, "
             "bright overhead lighting, cinematic, high quality"),
            ("A woman with black hair in a tight ponytail, wearing a white double-breasted "
             "chef coat, black apron and tall white chef hat, chops fresh green herbs and "
             "stirs a bubbling sauce in a stainless steel pot on a gas burner with blue flame, "
             "professional kitchen, cinematic, high quality"),
            ("A woman with black hair in a tight ponytail, wearing a white double-breasted "
             "chef coat, black apron and tall white chef hat, carefully tastes from a silver "
             "spoon and nods with satisfaction in a professional kitchen with copper pots "
             "hanging above, cinematic, high quality"),
            ("A woman with black hair in a tight ponytail, wearing a white double-breasted "
             "chef coat, black apron and tall white chef hat, meticulously arranges colorful "
             "food on white plates with artistic precision on a stainless steel counter, "
             "professional kitchen, cinematic, high quality"),
            ("A woman with black hair in a tight ponytail, wearing a white double-breasted "
             "chef coat, black apron and tall white chef hat, stands at the pass window "
             "watching guests enjoy food in a warmly lit restaurant dining area, "
             "cinematic, high quality"),
        ],
    },
]


# ============================================================
# LAMBDA SCHEDULING FUNCTIONS
# ============================================================
def lambda_cosine(normalized_t, lambda_max=0.5):
    """
    Cosine λ schedule: high at start (noisy), smooth decay to 0 at end (clean).
    normalized_t: 1.0 = noisy (start), 0.0 = clean (end)
    """
    return lambda_max * 0.5 * (1 + math.cos(math.pi * (1 - normalized_t)))


def lambda_linear(normalized_t, lambda_max=0.5):
    """Linear λ decay from lambda_max (noisy) to 0 (clean)."""
    return lambda_max * normalized_t


def lambda_step(normalized_t, lambda_max=0.5, threshold=0.3):
    """Step function: full anchor early, zero anchor late."""
    return lambda_max if normalized_t > threshold else 0.0


LAMBDA_SCHEDULES = {
    'cosine': lambda_cosine,
    'linear': lambda_linear,
    'step': lambda_step,
}


# ============================================================
# UNIFIED CROSS-ATTENTION (supports concat + dual modes)
# ============================================================
def _unified_cross_attn_forward(self, x, context, context_lens):
    """
    Unified cross-attention supporting three modes:

    1. Standard (no anchor): normal cross-attention
    2. Concat mode: K̃ = [K_prompt || K_anchor], Ṽ = [V_prompt || V_anchor]
    3. Dual mode:  Attn = Softmax(Q·K_p^T/√d)·V_p + λ(t)·Softmax(Q·K_a^T/√d)·V_a

    Controlled by:
      self._anchor_context: None or tensor
      self._anchor_mode:    None, 'concat', or 'dual'
      self._current_lambda: float (for dual mode, set by model forward wrapper)
    """
    anchor_ctx = getattr(self, '_anchor_context', None)
    mode = getattr(self, '_anchor_mode', None)

    b, n, d = x.size(0), self.num_heads, self.head_dim
    q = self.norm_q(self.q(x)).view(b, -1, n, d)

    if anchor_ctx is None or mode is None:
        # Standard cross-attention (baseline, text_concat)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = wan_attention(q, k, v, k_lens=context_lens)

    elif mode == 'concat':
        # N-Anchor Concat: K̃ = [K_prompt || K_anchor]
        combined = torch.cat([context, anchor_ctx], dim=1)
        k = self.norm_k(self.k(combined)).view(b, -1, n, d)
        v = self.v(combined).view(b, -1, n, d)
        x = wan_attention(q, k, v, k_lens=None)

    elif mode == 'dual':
        # Dual Attention with λ(t) scheduling
        lam = getattr(self, '_current_lambda', 0.3)

        # Prompt attention
        k_p = self.norm_k(self.k(context)).view(b, -1, n, d)
        v_p = self.v(context).view(b, -1, n, d)
        x_p = wan_attention(q, k_p, v_p, k_lens=context_lens)

        # Anchor attention
        k_a = self.norm_k(self.k(anchor_ctx)).view(b, -1, n, d)
        v_a = self.v(anchor_ctx).view(b, -1, n, d)
        x_a = wan_attention(q, k_a, v_a, k_lens=None)

        # Weighted combination: prompt + λ(t) * anchor
        x = x_p + lam * x_a

    x = x.flatten(2)
    x = self.o(x)
    return x


def _model_forward_with_lambda(self, x, t, context, seq_len,
                                clip_fea=None, y=None):
    """
    Wrapped WanModel.forward() that sets λ on cross-attention blocks
    based on current timestep before each forward pass.
    """
    schedule_fn = getattr(self, '_lambda_schedule_fn', None)
    if schedule_fn is not None:
        # Normalize timestep: ~1.0 at start (noisy), ~0.0 at end (clean)
        normalized_t = min(t[0].item() / 1000.0, 1.0)
        lam = schedule_fn(normalized_t)
        for block in self.blocks:
            block.cross_attn._current_lambda = lam
    # Call original class forward (not the instance override)
    from wan.modules.model import WanModel
    return WanModel.forward(self, x, t, context, seq_len,
                           clip_fea=clip_fea, y=y)


# ============================================================
# MODEL PATCHING HELPERS
# ============================================================
def patch_cross_attention(model):
    """Monkey-patch all cross-attention blocks with unified forward."""
    for block in model.blocks:
        block.cross_attn._original_forward = block.cross_attn.forward
        block.cross_attn._anchor_context = None
        block.cross_attn._anchor_mode = None
        block.cross_attn._current_lambda = 0.0
        block.cross_attn.forward = types.MethodType(
            _unified_cross_attn_forward, block.cross_attn)


def patch_model_forward(model, schedule_name):
    """Wrap model forward for timestep-based λ scheduling."""
    model._lambda_schedule_fn = LAMBDA_SCHEDULES[schedule_name]
    model.forward = types.MethodType(_model_forward_with_lambda, model)


def unpatch_model_forward(model):
    """Restore original model forward."""
    if hasattr(model, '_lambda_schedule_fn'):
        from wan.modules.model import WanModel
        model.forward = types.MethodType(WanModel.forward, model)
        model._lambda_schedule_fn = None


def unpatch_cross_attention(model):
    """Restore original cross-attention."""
    for block in model.blocks:
        if hasattr(block.cross_attn, '_original_forward'):
            block.cross_attn.forward = block.cross_attn._original_forward


def set_anchor_context(model, anchor_ctx):
    """Set anchor context tensor on all cross-attention blocks."""
    for block in model.blocks:
        block.cross_attn._anchor_context = anchor_ctx


def set_anchor_mode(model, mode):
    """Set anchor mode ('concat', 'dual', or None) on all blocks."""
    for block in model.blocks:
        block.cross_attn._anchor_mode = mode


def clear_anchor_context(model):
    """Clear anchor context from all cross-attention blocks."""
    for block in model.blocks:
        block.cross_attn._anchor_context = None
        block.cross_attn._anchor_mode = None


# ============================================================
# EVALUATION METRICS
# (Adapted from VGoT: evaluate/code/calculate_clip.py)
# (Adapted from StoryMem: extract_keyframes.py)
# ============================================================
def extract_middle_frame(video_tensor):
    """
    Extract middle frame from video tensor.
    video_tensor: [C, T, H, W] in [-1, 1]
    Returns: PIL Image
    """
    t = video_tensor.shape[1]
    mid = t // 2
    frame = video_tensor[:, mid, :, :]  # [C, H, W]
    frame = (frame + 1.0) / 2.0  # [-1,1] → [0,1]
    frame = frame.clamp(0, 1)
    frame = (frame * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(frame)


def extract_video_frames_from_tensor(video_tensor, num_frames=8):
    """
    Extract uniformly sampled frames from video tensor.
    Adapted from VGoT: evaluate/code/calculate_clip.py
    video_tensor: [C, T, H, W] in [-1, 1]
    Returns: list of PIL Images
    """
    t = video_tensor.shape[1]
    if t <= num_frames:
        indices = list(range(t))
    else:
        step = t / num_frames
        indices = [int(i * step) for i in range(num_frames)]

    frames = []
    for idx in indices:
        frame = video_tensor[:, idx, :, :]
        frame = (frame + 1.0) / 2.0
        frame = frame.clamp(0, 1)
        frame = (frame * 255).byte().permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray(frame))
    return frames


@torch.no_grad()
def compute_clip_image_similarity(images, clip_model, preprocess, device):
    """
    Compute pairwise CLIP-I cosine similarity between consecutive images.
    Adapted from VGoT evaluation code and StoryMem extract_keyframes.py

    Args:
        images: list of PIL Images
        clip_model: CLIP model
        preprocess: CLIP preprocessing transform
        device: torch device

    Returns:
        list of cosine similarities between consecutive pairs
    """
    if len(images) < 2:
        return []

    # Encode all images
    image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
    features = clip_model.encode_image(image_tensors)
    features = features / features.norm(dim=-1, keepdim=True)

    # Compute consecutive similarities
    similarities = []
    for i in range(len(features) - 1):
        sim = F.cosine_similarity(
            features[i].unsqueeze(0), features[i + 1].unsqueeze(0))
        similarities.append(sim.item())

    return similarities


@torch.no_grad()
def compute_clip_text_image_similarity(image, text, clip_model, preprocess,
                                       tokenizer, device):
    """
    Compute CLIP similarity between an image and text.
    Adapted from VGoT: evaluate/code/calculate_clip.py

    Returns: float similarity score
    """
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tokens = tokenizer([text]).to(device)

    img_feat = clip_model.encode_image(img_tensor)
    text_feat = clip_model.encode_text(text_tokens)

    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sim = F.cosine_similarity(img_feat, text_feat)
    return sim.item()


# ============================================================
# VIDEO GENERATION ENGINE
# ============================================================
class NarrativeVideoGenerator:
    """
    Generates multi-shot videos from narratives using Wan2.1-T2V-1.3B
    with optional N-Anchor cross-attention injection.
    """

    def __init__(self, config, narratives=None):
        self.config = config
        self.narratives = narratives or NARRATIVES
        self.device = torch.device('cuda:0')

        logging.info("Loading Wan2.1-T2V-1.3B pipeline...")
        cfg = WAN_CONFIGS[config['task']]
        self.wan_cfg = cfg

        self.pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=config['model_path'],
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
        )

        # Monkey-patch cross-attention with unified forward (concat + dual)
        patch_cross_attention(self.pipeline.model)
        logging.info("Cross-attention patched (unified: concat + dual modes)")

        self._current_condition = None

        # Pre-encode anchor texts
        self.anchor_embeddings = {}
        self._precompute_anchors()

    def _precompute_anchors(self):
        """Pre-encode all anchor texts through T5 + text_embedding."""
        logging.info("Pre-encoding anchor texts through T5...")
        self.pipeline.text_encoder.model.to(self.device)

        for story in self.narratives:
            anchor_text = (
                f"{story['anchor']['entity']}. {story['anchor']['background']}"
            )
            # Encode through T5
            raw_ctx = self.pipeline.text_encoder([anchor_text], self.device)
            # Process through model's text_embedding (same as in WanModel.forward)
            # Must use autocast to match dtype (T5 outputs bf16, model may be fp32)
            text_len = self.pipeline.model.text_len
            text_dim = raw_ctx[0].size(1)
            padded = torch.cat([
                raw_ctx[0],
                raw_ctx[0].new_zeros(text_len - raw_ctx[0].size(0), text_dim)
            ]).unsqueeze(0)  # [1, text_len, text_dim]
            with amp.autocast(dtype=self.pipeline.param_dtype):
                anchor_emb = self.pipeline.model.text_embedding(padded)
            self.anchor_embeddings[story['id']] = anchor_emb.detach()
            logging.info(f"  Anchor encoded for {story['id']}: "
                         f"shape={anchor_emb.shape}, dtype={anchor_emb.dtype}")

        # Keep T5 on GPU for generation (offload_model=False)
        logging.info("Anchor pre-encoding complete")

    def _setup_condition(self, condition):
        """
        Configure model for a specific condition.
        Handles cross-attention mode and model forward wrapping.
        """
        if self._current_condition == condition:
            return

        model = self.pipeline.model

        # Reset: clear anchor, unpatch model forward
        clear_anchor_context(model)
        unpatch_model_forward(model)

        if condition in ('baseline', 'text_concat'):
            # No anchor injection needed
            pass
        elif condition == 'n_anchor_concat':
            set_anchor_mode(model, 'concat')
        elif condition.startswith('n_anchor_dual_'):
            # e.g. 'n_anchor_dual_cosine', 'n_anchor_dual_linear', 'n_anchor_dual_step'
            schedule_name = condition.split('n_anchor_dual_')[1]
            set_anchor_mode(model, 'dual')
            patch_model_forward(model, schedule_name)
            logging.info(f"  Dual attention enabled with λ schedule: {schedule_name}")

        self._current_condition = condition

    def get_prompt(self, story, shot_idx, condition):
        """Get the appropriate prompt for a given condition."""
        if condition == 'baseline':
            return story['raw_sentences'][shot_idx]
        elif condition == 'text_concat':
            anchor_text = (
                f"{story['anchor']['entity']}. "
                f"{story['anchor']['background']}. "
            )
            return anchor_text + story['translated_prompts'][shot_idx]
        else:
            # All n_anchor variants use translated prompts
            return story['translated_prompts'][shot_idx]

    def generate_shot(self, story, shot_idx, condition, seed):
        """
        Generate a single shot video.

        Args:
            story: narrative dict
            shot_idx: shot index (0-4)
            condition: any supported condition string
            seed: random seed for this shot

        Returns:
            video tensor [C, T, H, W] in [-1, 1]
        """
        # Setup condition (patches model if needed)
        self._setup_condition(condition)

        prompt = self.get_prompt(story, shot_idx, condition)

        # Set anchor context for anchor-based conditions
        if condition.startswith('n_anchor'):
            anchor_emb = self.anchor_embeddings[story['id']]
            set_anchor_context(self.pipeline.model, anchor_emb)
        else:
            clear_anchor_context(self.pipeline.model)

        # Generate video
        video = self.pipeline.generate(
            input_prompt=prompt,
            size=SIZE_CONFIGS[self.config['size']],
            frame_num=self.config['frame_num'],
            shift=self.config['shift'],
            sample_solver='unipc',
            sampling_steps=self.config['sampling_steps'],
            guide_scale=self.config['guide_scale'],
            seed=seed,
            offload_model=False,
        )

        # Clear anchor after generation
        clear_anchor_context(self.pipeline.model)

        return video

    def cleanup(self):
        """Free GPU memory."""
        clear_anchor_context(self.pipeline.model)
        unpatch_model_forward(self.pipeline.model)
        unpatch_cross_attention(self.pipeline.model)
        self.pipeline.model.cpu()
        self.pipeline.text_encoder.model.cpu()
        self.pipeline.vae.model.cpu()
        self.anchor_embeddings.clear()
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(config):
    """Run the complete N-Anchor experiment."""
    output_dir = Path(config.get('output_dir',
                                  str(Path(__file__).parent / 'outputs')))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config for reproducibility
    config_path = output_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    logging.info(f"Config saved to {config_path}")

    # Initialize generator
    narratives = config.get('_narratives', NARRATIVES)
    generator = NarrativeVideoGenerator(config, narratives=narratives)

    # Generation phase
    num_shots = config['num_shots']
    conditions = config['conditions']
    base_seed = config['base_seed']
    generation_log = []

    total = len(narratives) * num_shots * len(conditions)
    counter = 0

    for story_idx, story in enumerate(narratives):
        for condition in conditions:
            cond_dir = output_dir / condition / story['id']
            cond_dir.mkdir(parents=True, exist_ok=True)

            for shot_idx in range(num_shots):
                # Deterministic seed: same (story, shot) → same noise across conditions
                seed = base_seed + story_idx * 1000 + shot_idx
                counter += 1

                video_path = cond_dir / f'shot_{shot_idx:02d}.mp4'
                if video_path.exists():
                    logging.info(f"[{counter}/{total}] SKIP (exists): {video_path}")
                    continue

                logging.info(
                    f"[{counter}/{total}] Generating: "
                    f"{story['id']}/{condition}/shot_{shot_idx:02d} "
                    f"(seed={seed})")
                prompt = generator.get_prompt(story, shot_idx, condition)
                logging.info(f"  Prompt: {prompt[:100]}...")

                t0 = time.time()
                video = generator.generate_shot(
                    story, shot_idx, condition, seed)
                elapsed = time.time() - t0

                # Save video
                cache_video(
                    tensor=video[None],
                    save_file=str(video_path),
                    fps=generator.wan_cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))

                # Save middle frame as reference
                mid_frame = extract_middle_frame(video)
                mid_frame.save(str(cond_dir / f'shot_{shot_idx:02d}_mid.png'))

                gen_entry = {
                    'story': story['id'],
                    'condition': condition,
                    'shot_idx': shot_idx,
                    'seed': seed,
                    'prompt': prompt,
                    'video_path': str(video_path),
                    'elapsed_sec': round(elapsed, 2),
                }
                generation_log.append(gen_entry)
                logging.info(f"  Saved: {video_path} ({elapsed:.1f}s)")

                # Free video tensor
                del video
                torch.cuda.empty_cache()

    # Save generation log
    log_path = output_dir / 'generation_log.json'
    with open(log_path, 'w') as f:
        json.dump(generation_log, f, indent=2)
    logging.info(f"Generation log saved to {log_path}")

    # Cleanup generator
    generator.cleanup()

    # ---- Evaluation Phase ----
    logging.info("=" * 60)
    logging.info("Starting evaluation...")
    results = evaluate_all(output_dir, config)

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")

    # Print summary
    print_results_summary(results)

    return results


def evaluate_all(output_dir, config):
    """
    Evaluate all generated videos.
    Metrics:
      - Entity Consistency: CLIP-I cosine similarity between consecutive shots
      - Narrative Alignment: CLIP text-image similarity
    """
    import clip as clip_module

    device = torch.device('cuda:0')
    clip_model, preprocess = clip_module.load("ViT-B/32", device=device)
    clip_model.eval()
    tokenizer = clip_module.tokenize

    conditions = config['conditions']
    num_shots = config['num_shots']
    results = {
        'entity_consistency': {},
        'narrative_alignment': {},
        'per_story': {},
    }

    narratives = config.get('_narratives', NARRATIVES)

    for condition in conditions:
        all_consistencies = []
        all_alignments = []

        for story in narratives:
            cond_dir = output_dir / condition / story['id']

            # Load middle frames for all shots
            frames = []
            for shot_idx in range(num_shots):
                frame_path = cond_dir / f'shot_{shot_idx:02d}_mid.png'
                if frame_path.exists():
                    frames.append(Image.open(frame_path).convert('RGB'))
                else:
                    logging.warning(f"Missing frame: {frame_path}")

            if len(frames) == 0:
                continue

            # Entity Consistency: CLIP-I similarity between consecutive shots
            consistencies = []
            if len(frames) >= 2:
                consistencies = compute_clip_image_similarity(
                    frames, clip_model, preprocess, device)
                all_consistencies.extend(consistencies)

            # Narrative Alignment: CLIP text-image similarity
            story_alignments = []
            for shot_idx, frame in enumerate(frames):
                # Use translated prompt for all conditions (fair comparison)
                prompt = story['translated_prompts'][shot_idx]
                sim = compute_clip_text_image_similarity(
                    frame, prompt, clip_model, preprocess, tokenizer, device)
                story_alignments.append(sim)
            all_alignments.extend(story_alignments)

            # Per-story results
            story_key = f"{condition}/{story['id']}"
            results['per_story'][story_key] = {
                'entity_consistency': {
                    'values': consistencies,
                    'mean': float(np.mean(consistencies)) if consistencies else 0,
                },
                'narrative_alignment': {
                    'values': story_alignments,
                    'mean': float(np.mean(story_alignments)),
                },
            }

        # Aggregate per condition
        results['entity_consistency'][condition] = {
            'mean': float(np.mean(all_consistencies)) if all_consistencies else 0,
            'std': float(np.std(all_consistencies)) if all_consistencies else 0,
            'n': len(all_consistencies),
        }
        results['narrative_alignment'][condition] = {
            'mean': float(np.mean(all_alignments)) if all_alignments else 0,
            'std': float(np.std(all_alignments)) if all_alignments else 0,
            'n': len(all_alignments),
        }

    return results


def print_results_summary(results):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS: N-Anchor Narrative-to-Video Generation")
    print("=" * 70)

    print("\n--- Entity Consistency (CLIP-I, ↑ higher is better) ---")
    print(f"{'Condition':<25} {'Mean':>8} {'Std':>8} {'N':>5}")
    print("-" * 50)
    for cond, vals in results['entity_consistency'].items():
        print(f"{cond:<25} {vals['mean']:>8.4f} {vals['std']:>8.4f} {vals['n']:>5}")

    print("\n--- Narrative Alignment (CLIP T-I, ↑ higher is better) ---")
    print(f"{'Condition':<25} {'Mean':>8} {'Std':>8} {'N':>5}")
    print("-" * 50)
    for cond, vals in results['narrative_alignment'].items():
        print(f"{cond:<25} {vals['mean']:>8.4f} {vals['std']:>8.4f} {vals['n']:>5}")

    print("\n--- Per-Story Breakdown ---")
    for key, vals in results['per_story'].items():
        ec = vals['entity_consistency']['mean']
        na = vals['narrative_alignment']['mean']
        print(f"  {key:<45} EC={ec:.4f}  NA={na:.4f}")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="N-Anchor: Training-Free Narrative-to-Video Experiment")
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for videos and results')
    parser.add_argument('--num_shots', type=int, default=5,
                        help='Number of shots per story (default: 5)')
    parser.add_argument('--frame_num', type=int, default=17,
                        help='Frames per video, must be 4n+1 (default: 17)')
    parser.add_argument('--sampling_steps', type=int, default=30,
                        help='Denoising steps (default: 30)')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--conditions', type=str, nargs='+',
                        default=['baseline', 'text_concat', 'n_anchor_concat'],
                        help='Conditions to run')
    parser.add_argument('--narratives', type=str, default='synthetic',
                        choices=['synthetic', 'novel', 'all'],
                        help='Which narratives to use (default: synthetic)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip generation, only run evaluation')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])

    # Build config
    config = dict(DEFAULT_CONFIG)
    if args.output_dir:
        config['output_dir'] = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config['output_dir'] = str(
            Path(__file__).parent / 'outputs' / f'exp_{timestamp}')
    config['num_shots'] = args.num_shots
    config['frame_num'] = args.frame_num
    config['sampling_steps'] = args.sampling_steps
    config['base_seed'] = args.base_seed
    config['conditions'] = args.conditions

    # Load narratives
    if args.narratives == 'novel' or args.narratives == 'all':
        from novel_narratives import NOVEL_NARRATIVES
    if args.narratives == 'synthetic':
        config['_narratives'] = NARRATIVES
    elif args.narratives == 'novel':
        config['_narratives'] = NOVEL_NARRATIVES
    elif args.narratives == 'all':
        config['_narratives'] = NARRATIVES + NOVEL_NARRATIVES

    logging.info("=" * 60)
    logging.info("N-Anchor Experiment")
    logging.info(f"  Stories: {len(config['_narratives'])}")
    logging.info(f"  Shots/story: {config['num_shots']}")
    logging.info(f"  Conditions: {config['conditions']}")
    logging.info(f"  Total videos: "
                 f"{len(NARRATIVES) * config['num_shots'] * len(config['conditions'])}")
    logging.info(f"  Resolution: {config['size']}")
    logging.info(f"  Frames: {config['frame_num']}")
    logging.info(f"  Steps: {config['sampling_steps']}")
    logging.info(f"  Base seed: {config['base_seed']}")
    logging.info(f"  Output: {config['output_dir']}")
    logging.info("=" * 60)

    if args.eval_only:
        output_dir = Path(config['output_dir'])
        results = evaluate_all(output_dir, config)
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print_results_summary(results)
    else:
        run_experiment(config)


if __name__ == '__main__':
    main()
