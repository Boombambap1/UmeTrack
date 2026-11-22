import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umetrack_rb5.encoder_provider import (
    save_records,
    load_records,
    tensor_digest,
    metadata_digest,
)
from umetrack_rb5.raw_exchange import export_inputs, import_outputs
from raw_fixtures import write_fixture


class RawExchangeChecks(unittest.TestCase):
    def reference(self, root):
        records = {}
        rng = np.random.default_rng(42)
        for index, count in enumerate((2, 1)):
            inputs = rng.random((count, 1, 96, 96), dtype=np.float32)
            features = rng.standard_normal((count, 72, 6, 6)).astype(np.float32)
            metadata = {"frame": index}
            records[f"clip:{index}"] = {
                "crop_ids": [f"clip:{index}:hand0:camera{c}" for c in range(count)],
                "inputs": inputs,
                "features": features,
                "metadata": metadata,
                "metadata_digest": metadata_digest(metadata),
                "input_digest": tensor_digest(inputs),
                "feature_digest": tensor_digest(features),
            }
        reference = root / "reference"
        provenance = {"checkpoint_sha256": "a" * 64}
        save_records(records, reference / "features", provenance)
        return reference, records

    def test_both_layouts_reassemble_exactly(self):
        for layout in ("NCHW", "NHWC"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                reference, records = self.reference(root)
                package = root / "package"
                returned = root / "returned"
                imported = root / "imported"
                export_inputs(reference, package, layout)
                write_fixture(reference, package, returned, layout)
                summary = import_outputs(reference, package, returned, imported)
                self.assertEqual(summary["feature_max_absolute_error"], 0)
                manifest = json.loads((imported / "features/manifest.json").read_text())
                loaded = load_records(imported / "features", manifest["provenance"])
                for key in records:
                    np.testing.assert_array_equal(
                        records[key]["features"], loaded[key]["features"]
                    )

    def test_reject_missing_extra_truncated_nonfinite_outputs(self):
        for change in ("missing", "extra", "truncated", "nonfinite"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                reference, _ = self.reference(root)
                package = root / "package"
                returned = root / "returned"
                export_inputs(reference, package, "NHWC")
                write_fixture(reference, package, returned, "NHWC")
                path = returned / "Result_0/features.raw"
                if change == "missing":
                    path.unlink()
                elif change == "extra":
                    (returned / "extra.raw").write_bytes(b"junk")
                elif change == "truncated":
                    path.write_bytes(path.read_bytes()[:-4])
                else:
                    array = np.fromfile(path, dtype="<f4")
                    array[0] = np.nan
                    array.tofile(path)
                with self.assertRaises(ValueError):
                    import_outputs(reference, package, returned, root / "imported")

    def test_reject_changed_input_list_and_crop_map(self):
        for change in ("list", "map", "input"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                reference, _ = self.reference(root)
                package = root / "package"
                returned = root / "returned"
                export_inputs(reference, package, "NHWC")
                write_fixture(reference, package, returned, "NHWC")
                if change == "list":
                    (package / "input_list.txt").write_text("inputs/wrong.raw\n")
                elif change == "input":
                    (package / "inputs/crop-00000.raw").write_bytes(b"wrong")
                else:
                    path = package / "package.json"
                    manifest = json.loads(path.read_text())
                    manifest["crops"].reverse()
                    path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    import_outputs(reference, package, returned, root / "imported")

    def test_board_provenance_requires_matching_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reference, _ = self.reference(root)
            package = root / "package"
            returned = root / "returned"
            export_inputs(reference, package, "NHWC")
            write_fixture(reference, package, returned, "NHWC")
            path = returned / "execution.json"
            execution = json.loads(path.read_text())
            execution.update(
                execution_kind="board",
                model_format="dlc",
                sdk_version="test-runtime",
                conversion_sdk_version="wrong",
                source_checkpoint_sha256="a" * 64,
            )
            path.write_text(json.dumps(execution))
            with self.assertRaises(ValueError):
                import_outputs(reference, package, returned, root / "imported")
            execution["conversion_sdk_version"] = "test-runtime"
            execution["source_checkpoint_sha256"] = "b" * 64
            path.write_text(json.dumps(execution))
            with self.assertRaises(ValueError):
                import_outputs(reference, package, returned, root / "imported")

    def test_reject_provenance_layout_and_evidence_mismatch(self):
        for key, value in [
            ("package_sha256", "0" * 64),
            ("output_layout", "guess"),
            ("output_dtype", "int8"),
            ("sdk_version", ""),
        ]:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                reference, _ = self.reference(root)
                package = root / "package"
                returned = root / "returned"
                export_inputs(reference, package, "NHWC")
                write_fixture(reference, package, returned, "NHWC")
                path = returned / "execution.json"
                execution = json.loads(path.read_text())
                execution[key] = value
                path.write_text(json.dumps(execution))
                with self.assertRaises(ValueError):
                    import_outputs(reference, package, returned, root / "imported")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reference, _ = self.reference(root)
            package = root / "package"
            returned = root / "returned"
            export_inputs(reference, package, "NHWC")
            write_fixture(reference, package, returned, "NHWC")
            (returned / "fixture.txt").write_text("changed")
            with self.assertRaises(ValueError):
                import_outputs(reference, package, returned, root / "imported")


if __name__ == "__main__":
    unittest.main()
