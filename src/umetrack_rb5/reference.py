from dataclasses import fields
import hashlib
from itertools import islice
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/UmeTrack"))
from lib.data_utils import bundles
from lib.models.model_loader import load_pretrained_model
from lib.tracker.tracker import HandTracker, HandTrackerOpts
from lib.tracker.video_pose_data import SyncedImagePoseStream
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from .encoder_provider import EncoderProvider


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_cpu(threads=2):
    if torch.cuda.is_available():
        raise RuntimeError("CPU reference required")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def setup(clip):
    if clip["start_frame"] != 0:
        raise ValueError(
            "Only sequence-start clips supported; mid-sequence history must not be silently omitted"
        )
    stream = SyncedImagePoseStream(str(ROOT / clip["local_path"]))
    model = (
        load_pretrained_model(
            str(
                ROOT / "third_party/UmeTrack/pretrained_models/pretrained_weights.torch"
            )
        )
        .cpu()
        .eval()
    )
    return stream, model


class ContextTracker(HandTracker):
    def _make_inputs(self, *args):
        data, desc, skeleton = super()._make_inputs(*args)
        metadata = {}
        for prefix, obj in (("frame", data), ("desc", desc), ("skeleton", skeleton)):
            for field in fields(obj):
                metadata[prefix + "." + field.name] = (
                    getattr(obj, field.name).detach().cpu().tolist()
                )
        self.provider.set_context(self.frame_key, self.crop_ids, metadata)
        return data, desc, skeleton


def run_clip(clip, mode, records=None):
    stream, model = setup(clip)
    provider = EncoderProvider(model._feature_extractor._image_backbone, mode, records)
    model._feature_extractor._image_backbone = provider
    tracker = ContextTracker(model, HandTrackerOpts())
    tracker.provider = provider
    count = clip["frames"]
    predicted = np.zeros((2, count, 21, 3))
    truth = np.zeros_like(predicted)
    valid = np.zeros((2, count), dtype=bool)
    frames = []
    hand_model = stream._hand_pose_labels.hand_model
    try:
        for index, (frame, gt) in enumerate(islice(stream, count)):
            crops = tracker.gen_crop_cameras(
                [v.camera for v in frame.views],
                stream._hand_pose_labels.camera_angles,
                hand_model,
                gt,
                min_num_crops=1,
            )
            key = f'{clip["id"]}:{index:06d}'
            ids = [
                f"{key}:hand{hand}:camera{cam}"
                for hand, views in crops.items()
                for cam in views
            ]
            tracker.frame_key, tracker.crop_ids = key, ids
            result = tracker.track_frame(frame, hand_model, crops)
            frames.append(
                {
                    "frame_index": index,
                    "key": key,
                    "crop_ids": ids,
                    "crop_count": len(ids),
                }
            )
            for hand, pose in result.hand_poses.items():
                predicted[hand, index] = landmarks_from_hand_pose(
                    hand_model, pose, hand
                )
                truth[hand, index] = landmarks_from_hand_pose(
                    hand_model, gt[hand], hand
                )
                valid[hand, index] = True
        if len(frames) != count:
            raise RuntimeError("Clip ended early")
        provider.assert_consumed()
        return (
            {"predicted": predicted, "truth": truth, "valid": valid},
            provider.records,
            frames,
        )
    finally:
        model._feature_extractor._image_backbone = provider.backbone


def profile_clip(clip, warmup=10):
    stream, model = setup(clip)
    tracker = HandTracker(model, HandTrackerOpts())
    encoder_times = []
    stamp = [None]

    def before(module, args):
        stamp[0] = time.perf_counter()

    def after(module, args, output):
        encoder_times.append(time.perf_counter() - stamp[0])

    backbone = model._feature_extractor._image_backbone
    hooks = [
        backbone.register_forward_pre_hook(before),
        backbone.register_forward_hook(after),
    ]
    rows = []
    iterator = iter(stream)
    hand_model = stream._hand_pose_labels.hand_model
    try:
        for index in range(clip["frames"]):
            start = time.perf_counter()
            frame, gt = next(iterator)
            decoded = time.perf_counter()
            crops = tracker.gen_crop_cameras(
                [v.camera for v in frame.views],
                stream._hand_pose_labels.camera_angles,
                hand_model,
                gt,
                min_num_crops=1,
            )
            selected = time.perf_counter()
            encoder_times.clear()
            crop_count = sum(map(len, crops.values()))
            if crops:
                data, desc, skeleton = tracker._make_inputs(frame, hand_model, crops)
                prepared = time.perf_counter()
                with torch.no_grad():
                    output = bundles.to_device(
                        model.regress_pose_use_skeleton(data, desc, skeleton),
                        torch.device("cpu"),
                    )
                inferred = time.perf_counter()
                result = tracker._gen_tracking_result(
                    output, desc.hand_idx.cpu().numpy(), crops
                )
                postprocessed = time.perf_counter()
                for hand, pose in result.hand_poses.items():
                    landmarks_from_hand_pose(hand_model, pose, hand)
                    landmarks_from_hand_pose(hand_model, gt[hand], hand)
            else:
                tracker.reset_history()
                prepared = inferred = postprocessed = time.perf_counter()
            end = time.perf_counter()
            encoder = sum(encoder_times)
            row = {
                "clip": clip["id"],
                "frame": index,
                "warmup": index < warmup,
                "crops": crop_count,
                "decode_s": decoded - start,
                "crop_selection_s": selected - decoded,
                "prepare_s": prepared - selected,
                "encoder_s": encoder,
                "remaining_model_s": inferred - prepared - encoder,
                "postprocess_s": postprocessed - inferred,
                "landmarks_s": end - postprocessed,
                "total_s": end - start,
            }
            rows.append(row)
        return rows
    finally:
        for hook in hooks:
            hook.remove()
