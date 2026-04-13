"""
Generate per-segment visual embeddings for TTM annotations.

Input:
  - starter_code/data/result_TTM/<uid>.json
  - starter_code/data/json_original/<uid>/*.json
  - /DATA/G17/Data/video/<uid>/img_*.jpg

Output:
  - one JSON file per UID containing segment records with:
	  uid, person_id, label, tag, start_frame, end_frame, embedding

The embedding is a 512-d vector derived from the VideoSwinV2TTM temporal head.
If the model hidden size differs from 512, the vector is padded or truncated
to keep the output dimension stable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T


warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s - %(message)s",
	datefmt="%H:%M:%S",
)
logger = logging.getLogger("embedding_generation")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_val_transform(img_size: int = 224) -> T.Compose:
	return T.Compose([
		T.ToImage(),
		T.ToDtype(torch.float32, scale=True),
		T.Resize(int(img_size * 256 / 224), antialias=True),
		T.CenterCrop(img_size),
		T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])


def load_json(path: Path):
	with open(path) as f:
		return json.load(f)


def load_uid_list(path: Path) -> list[str]:
	with open(path) as f:
		return [line.strip() for line in f if line.strip()]


def load_frame(path: Path) -> Image.Image:
	return Image.open(path).convert("RGB")


def load_face_bboxes(uid: str, json_root: Path) -> dict[str, tuple[int, int, int, int]]:
	uid_dir = json_root / uid
	if not uid_dir.is_dir():
		return {}

	out: dict[str, tuple[int, int, int, int]] = {}
	for track_file in sorted(uid_dir.glob("*.json")):
		try:
			track = load_json(track_file)
		except Exception:
			continue
		if not isinstance(track, list):
			continue

		for frame in track:
			try:
				person_id = str(frame.get("Person ID", "")).replace("'", "")
				if not person_id:
					continue

				frame_number = int(frame["frameNumber"])
				if frame_number <= 0:
					continue

				x = float(frame["x"])
				y = float(frame["y"])
				w = float(frame["width"])
				h = float(frame["height"])
				if w <= 0 or h <= 0:
					continue

				x1 = max(0, int(x))
				y1 = max(0, int(y))
				x2 = max(x1 + 1, int(x + w))
				y2 = max(y1 + 1, int(y + h))
				out[f"{frame_number}:{person_id}"] = (x1, y1, x2, y2)
			except (KeyError, TypeError, ValueError):
				continue

	return out


def crop_face_or_black(img: Image.Image, bbox: tuple[int, int, int, int] | None) -> Image.Image:
	if bbox is None:
		return Image.new("RGB", img.size, (0, 0, 0))

	x1, y1, x2, y2 = bbox
	width, height = img.size
	x1 = max(0, min(x1, width - 1))
	y1 = max(0, min(y1, height - 1))
	x2 = max(x1 + 1, min(x2, width))
	y2 = max(y1 + 1, min(y2, height))
	if x2 <= x1 or y2 <= y1:
		return Image.new("RGB", img.size, (0, 0, 0))
	return img.crop((x1, y1, x2, y2))


def resolve_frame_path(frame_dir: Path, frame_number: int) -> Path | None:
	for name in (
		f"img_{frame_number:05d}.jpg",
		f"img_{frame_number:06d}.jpg",
		f"{frame_number:06d}.jpg",
		f"{frame_number:05d}.jpg",
		f"{frame_number}.jpg",
		f"img_{frame_number:05d}.png",
		f"{frame_number:06d}.png",
	):
		candidate = frame_dir / name
		if candidate.is_file():
			return candidate
	return None


def collect_frame_items(
	frame_dir: Path,
	bbox_map: dict[str, tuple[int, int, int, int]],
	person_id: int,
	start_frame: int,
	end_frame: int,
	stride: int,
) -> list[dict]:
	if not frame_dir.is_dir():
		return []

	all_frames = sorted(
		int(path.stem.replace("img_", ""))
		for path in frame_dir.iterdir()
		if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
	)
	if not all_frames:
		return []

	frame_set = set(all_frames)
	seg_frames = [frame for frame in range(start_frame, end_frame + 1, stride) if frame in frame_set]

	if not seg_frames:
		seg_frames = [frame for frame in all_frames if start_frame <= frame <= end_frame][::stride]

	items: list[dict] = []
	for frame_number in seg_frames:
		frame_path = resolve_frame_path(frame_dir, frame_number)
		if frame_path is None:
			continue
		items.append(
			{
				"frame_number": frame_number,
				"path": str(frame_path),
				"bbox": bbox_map.get(f"{frame_number}:{person_id}"),
			}
		)

	return items


def sample_items(items: list[dict], clip_frames: int) -> list[dict]:
	if not items:
		return []

	if len(items) >= clip_frames:
		indices = np.linspace(0, len(items) - 1, clip_frames, dtype=int)
		return [items[index] for index in indices]

	repeated = items * ((clip_frames // len(items)) + 1)
	return repeated[:clip_frames]


def force_embedding_dim(embedding: torch.Tensor, embedding_dim: int = 512) -> torch.Tensor:
	current_dim = embedding.shape[-1]
	if current_dim == embedding_dim:
		return embedding
	if current_dim > embedding_dim:
		return embedding[..., :embedding_dim]

	pad_width = embedding_dim - current_dim
	return torch.nn.functional.pad(embedding, (0, pad_width))


@torch.no_grad()
def encode_clip(model, clip: torch.Tensor, embedding_dim: int = 512) -> torch.Tensor:
	feats = model.backbone(clip)
	d_expected = model.temporal_head.cls_token.shape[-1]

	if feats.shape[-1] == d_expected:
		tokens = feats.mean(dim=(2, 3)).contiguous()
	elif feats.shape[1] == d_expected:
		tokens = feats.mean(dim=(3, 4)).transpose(1, 2).contiguous()
	else:
		raise RuntimeError(
			f"Unexpected backbone output shape {tuple(feats.shape)} for expected dim {d_expected}"
		)

	batch_size = tokens.size(0)
	cls = model.temporal_head.cls_token.expand(batch_size, -1, -1)
	norm_tokens = model.temporal_head.norm1(tokens)
	attended, _ = model.temporal_head.cross_attn(
		query=cls,
		key=norm_tokens,
		value=norm_tokens,
		need_weights=False,
	)
	cls = cls + attended
	cls = cls + model.temporal_head.mlp(model.temporal_head.norm2(cls))

	hidden = model.temporal_head.head[:-1](cls.squeeze(1))
	hidden = force_embedding_dim(hidden, embedding_dim=embedding_dim)
	return hidden


def load_model(args, device: torch.device):
	model = build_model(
		"VideoSwinV2TTM",
		variant=args.variant,
		pretrained=args.pretrained,
		num_classes=2,
		dropout=args.dropout,
		freeze_stages=args.freeze_stages,
	)

	if args.checkpoint:
		ckpt_path = Path(args.checkpoint)
		if ckpt_path.is_file():
			logger.info("Loading checkpoint: %s", ckpt_path)
			ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
			state = ckpt.get("state_dict", ckpt.get("model", ckpt))
			state = {key.replace("module.", ""): value for key, value in state.items()}
			missing, unexpected = model.load_state_dict(state, strict=False)
			logger.info("Checkpoint loaded. missing=%d unexpected=%d", len(missing), len(unexpected))
		else:
			logger.warning("Checkpoint not found: %s", ckpt_path)

	model.eval().to(device)
	return model


def choose_uids(args) -> list[str]:
	ttm_dir = Path(args.ttm_dir)
	if args.split in {"train", "val"}:
		split_file = Path(args.starter_code) / "data" / "split" / f"{args.split}.list"
		return load_uid_list(split_file)
	return sorted(path.stem for path in ttm_dir.glob("*.json"))


def build_segment_embeddings(args):
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	logger.info("Device: %s", device)

	ttm_dir = Path(args.ttm_dir)
	json_root = Path(args.json_path)
	frames_root = Path(args.frames_root)
	out_root = Path(args.output) / args.split
	out_root.mkdir(parents=True, exist_ok=True)

	transform = build_val_transform(args.img_size)
	model = load_model(args, device)

	uids = choose_uids(args)
	logger.info("Processing %d UIDs for split=%s", len(uids), args.split)

	saved = 0
	skipped = 0
	pos = 0
	neg = 0
	all_segments: list[dict] = []

	for uid in uids:
		ttm_path = ttm_dir / f"{uid}.json"
		frame_dir = frames_root / uid
		if not ttm_path.is_file() or not frame_dir.is_dir():
			skipped += 1
			continue

		try:
			segments = load_json(ttm_path)
		except Exception as exc:
			logger.error("Failed to read %s: %s", ttm_path, exc)
			skipped += 1
			continue

		if not isinstance(segments, list) or not segments:
			skipped += 1
			continue

		bbox_map = load_face_bboxes(uid, json_root)
		out_segments = []

		for seg in segments:
			try:
				person_id = int(seg.get("label", 0))
			except (TypeError, ValueError):
				person_id = 0

			try:
				start_frame = int(seg["start_frame"])
				end_frame = int(seg["end_frame"])
			except (KeyError, TypeError, ValueError):
				continue

			if end_frame < start_frame:
				continue
			if person_id == 0:
				continue

			tag_value = seg.get("tags", seg.get("tag", 0))
			try:
				tag_value = int(tag_value) if tag_value is not None else 0
			except (TypeError, ValueError):
				tag_value = 0
			label = 1 if tag_value == 1 else 0

			frame_items = collect_frame_items(
				frame_dir=frame_dir,
				bbox_map=bbox_map,
				person_id=person_id,
				start_frame=start_frame,
				end_frame=end_frame,
				stride=args.frame_stride,
			)
			if not frame_items:
				continue

			sampled_items = sample_items(frame_items, args.clip_frames)
			if not sampled_items:
				continue

			processed_frames = []
			for item in sampled_items:
				img = load_frame(Path(item["path"]))
				img = crop_face_or_black(img, item.get("bbox"))
				processed_frames.append(transform(img))

			clip = torch.stack(processed_frames, dim=1).unsqueeze(0).to(device)
			try:
				embedding = encode_clip(model, clip, embedding_dim=args.embedding_dim)
			except Exception as exc:
				logger.error(
					"Embedding failed for %s %d-%d: %s",
					uid,
					start_frame,
					end_frame,
					exc,
				)
				continue

			out_segments.append(
				{
					"uid": uid,
					"person_id": person_id,
					"label": label,
					"tag": tag_value,
					"start_frame": start_frame,
					"end_frame": end_frame,
					"embedding": embedding.squeeze(0).cpu().tolist(),
				}
			)

			if label == 1:
				pos += 1
			else:
				neg += 1

		if not out_segments:
			skipped += 1
			continue

		all_segments.extend(out_segments)
		saved += 1

	out_file = out_root / "visual_embedding.json"
	if out_file.exists() and not args.overwrite:
		logger.info("Output already exists and --overwrite is not set: %s", out_file)
	else:
		with open(out_file, "w") as f:
			json.dump({"segments": all_segments}, f, indent=2)
		logger.info("Wrote %d segments to %s", len(all_segments), out_file)

	logger.info(
		"Done. saved=%d skipped=%d pos=%d neg=%d pos_ratio=%.1f%%",
		saved,
		skipped,
		pos,
		neg,
		100.0 * pos / max(pos + neg, 1),
	)


def parse_args():
	parser = argparse.ArgumentParser(description="Generate visual TTM segment embeddings")
	parser.add_argument(
		"--ttm_dir",
		default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/result_TTM",
		help="Directory containing <uid>.json TTM segment annotations",
	)
	parser.add_argument(
		"--json_path",
		default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/json_original",
		help="Directory containing face track JSON folders",
	)
	parser.add_argument(
		"--frames_root",
		default="/DATA/G17/Data/video",
		help="Directory containing per-UID frame folders",
	)
	parser.add_argument(
		"--starter_code",
		default="/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code",
		help="Starter code root, used to find train/val split files",
	)
	parser.add_argument(
		"--output",
		default="/DATA/G17/outputs/visual_segment_embeddings",
		help="Output directory for per-UID embedding JSON files",
	)
	parser.add_argument(
		"--checkpoint",
		default="/DATA/G17/Group17_Who_is_Talking_To_Me/new_visual/experiments/fresh_phase2b/best_model.pth",
		help="Optional checkpoint to load into the visual model",
	)
	parser.add_argument(
		"--split",
		default="train",
		choices=["train", "val", "all"],
		help="Which UID list to process",
	)
	parser.add_argument(
		"--variant",
		default="swin3d_s",
		help="Backbone variant used by VideoSwinV2TTM",
	)
	parser.add_argument(
		"--clip_frames",
		type=int,
		default=8,
		help="Number of frames per segment clip",
	)
	parser.add_argument(
		"--frame_stride",
		type=int,
		default=1,
		help="Subsample stride applied inside each segment range",
	)
	parser.add_argument(
		"--img_size",
		type=int,
		default=224,
		help="Input image size for the visual encoder",
	)
	parser.add_argument(
		"--dropout",
		type=float,
		default=0.4,
		help="Model dropout used when building VideoSwinV2TTM",
	)
	parser.add_argument(
		"--freeze_stages",
		type=int,
		default=3,
		help="Optional stage freezing when loading the backbone",
	)
	parser.add_argument(
		"--embedding_dim",
		type=int,
		default=512,
		help="Final embedding dimension written to JSON",
	)
	parser.add_argument(
		"--pretrained",
		action="store_true",
		default=True,
		help="Use pretrained VideoSwin weights when available",
	)
	parser.add_argument(
		"--no_pretrained",
		action="store_false",
		dest="pretrained",
		help="Disable pretrained VideoSwin weights",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing output JSON files",
	)
	parser.add_argument(
		"--cpu",
		action="store_true",
		help="Force CPU even if CUDA is available",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	build_segment_embeddings(args)


if __name__ == "__main__":
	main()
