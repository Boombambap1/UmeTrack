import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/UmeTrack"))
from umetrack_rb5.encoder_provider import EncoderProvider, save_records, load_records


class SimpleBackbone(nn.Module):
    def forward(self, crops):
        return (
            crops.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1).expand(-1, 72, 6, 6).clone()
        )


class ProviderChecks(unittest.TestCase):
    def capture(self, count=2):
        crops = (
            torch.arange(count * 96 * 96, dtype=torch.float32).reshape(count, 1, 96, 96)
            / 100000
        )
        provider = EncoderProvider(SimpleBackbone(), "capture")
        ids = [f"clip:0:hand0:camera{i}" for i in range(count)]
        metadata = {"sample_range": [[0, count]], "use_memory": [False]}
        provider.set_context("clip:0", ids, metadata)
        with torch.no_grad():
            expected = provider(crops)
        return crops, ids, metadata, provider.records, expected

    def test_single_and_multi_view(self):
        for count in (1, 2, 4):
            crops, ids, meta, records, expected = self.capture(count)
            replay = EncoderProvider(SimpleBackbone(), "replay", records)
            replay.set_context("clip:0", ids, meta)
            torch.testing.assert_close(replay(crops), expected, atol=0, rtol=0)
            replay.assert_consumed()

    def test_reject_input_metadata_order_and_missing(self):
        crops, ids, meta, records, _ = self.capture()
        cases = [
            (crops + 1, ids, meta, records),
            (crops, ids[::-1], meta, records),
            (crops, ids, {"use_memory": [True]}, records),
            (crops, ids, meta, {}),
        ]
        for value, order, metadata, data in cases:
            provider = EncoderProvider(SimpleBackbone(), "replay", data)
            provider.set_context("clip:0", order, metadata)
            with self.assertRaises((KeyError, ValueError)):
                provider(value)

    def test_corrupt_wrong_shape_and_unused_features(self):
        crops, ids, meta, records, _ = self.capture()
        for mode in ("nonfinite", "shape", "digest"):
            changed = copy.deepcopy(records)
            if mode == "nonfinite":
                changed["clip:0"]["features"][0, 0, 0, 0] = np.nan
            elif mode == "shape":
                changed["clip:0"]["features"] = np.zeros((2, 72, 5, 6), np.float32)
            else:
                changed["clip:0"]["features"][0, 0, 0, 0] += 1
            provider = EncoderProvider(SimpleBackbone(), "replay", changed)
            provider.set_context("clip:0", ids, meta)
            with self.assertRaises(ValueError):
                provider(crops)
        provider = EncoderProvider(SimpleBackbone(), "replay", records)
        with self.assertRaises(RuntimeError):
            provider.assert_consumed()

    def test_bundle_checksums_provenance_and_file_order(self):
        crops, ids, meta, records, _ = self.capture()
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            provenance = {"checkpoint": "test"}
            records["clip:1"] = copy.deepcopy(records["clip:0"])
            records["clip:1"]["crop_ids"] = [x.replace("clip:0", "clip:1") for x in ids]
            save_records(records, bundle, provenance)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["records"].reverse()
            manifest_path.write_text(json.dumps(manifest))
            loaded = load_records(bundle, provenance)
            provider = EncoderProvider(SimpleBackbone(), "replay", loaded)
            for key in ("clip:0", "clip:1"):
                provider.set_context(key, loaded[key]["crop_ids"], meta)
                provider(crops)
            provider.assert_consumed()
            np.testing.assert_array_equal(loaded["clip:0"]["inputs"], crops.numpy())
            with self.assertRaises(ValueError):
                load_records(bundle, {"checkpoint": "wrong"})
            path = bundle / "features-0000.npz"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                load_records(bundle, provenance)

    def test_duplicate_calls(self):
        crops, ids, meta, records, _ = self.capture()
        provider = EncoderProvider(SimpleBackbone(), "replay", records)
        provider.set_context("clip:0", ids, meta)
        provider(crops)
        provider.set_context("clip:0", ids, meta)
        with self.assertRaises(RuntimeError):
            provider(crops)

    def test_no_hand_clears_history_without_encoder_call(self):
        from lib.models.model_loader import load_pretrained_model
        from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame

        model = load_pretrained_model(
            str(
                ROOT / "third_party/UmeTrack/pretrained_models/pretrained_weights.torch"
            )
        ).eval()
        provider = EncoderProvider(
            model._feature_extractor._image_backbone, "replay", {}
        )
        model._feature_extractor._image_backbone = provider
        tracker = HandTracker(model, HandTrackerOpts())
        tracker._valid_tracking_history[:] = True
        result = tracker.track_frame(InputFrame(views=[]), None, {})
        self.assertFalse(tracker._valid_tracking_history.any())
        self.assertFalse(result.hand_poses)
        self.assertFalse(provider.seen)
        from lib.tracker.video_pose_data import SyncedImagePoseStream

        stream = SyncedImagePoseStream(
            str(ROOT / "artifacts/data/smoke/recording_00.mp4")
        )
        frame, gt = next(iter(stream))
        hand_model = stream._hand_pose_labels.hand_model
        crops = tracker.gen_crop_cameras(
            [v.camera for v in frame.views],
            stream._hand_pose_labels.camera_angles,
            hand_model,
            gt,
            min_num_crops=1,
        )
        model._feature_extractor._image_backbone = provider.backbone
        first = tracker.track_frame(frame, hand_model, crops)
        self.assertTrue(tracker._valid_tracking_history.any())
        tracker.track_frame(InputFrame(views=[]), None, {})
        reappeared = tracker.track_frame(frame, hand_model, crops)
        for hand in first.hand_poses:
            np.testing.assert_array_equal(
                first.hand_poses[hand].joint_angles,
                reappeared.hand_poses[hand].joint_angles,
            )
            np.testing.assert_array_equal(
                first.hand_poses[hand].wrist_xform,
                reappeared.hand_poses[hand].wrist_xform,
            )


if __name__ == "__main__":
    unittest.main()
