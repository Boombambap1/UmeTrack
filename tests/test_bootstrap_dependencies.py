import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap_dependencies import ensure_repository, git, verify_checkpoint


class BootstrapChecks(unittest.TestCase):
    def source(self, root):
        source = root / "fixture-source"
        source.mkdir()
        git(source, "init", "--quiet")
        (source / "labels.json").write_text('{"fixture": true}\n')
        git(source, "add", "labels.json")
        git(source, "-c", "user.name=Bootstrap fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "--no-gpg-sign", "--quiet", "-m", "Temporary test fixture")
        return source, git(source, "rev-parse", "HEAD")

    def test_check_missing_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                ensure_repository(target, "unused", "0" * 40, check_only=True)
            self.assertFalse(target.exists())

    def test_restore_worktree_and_bare_metadata_without_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, revision = self.source(root)
            for metadata in (False, True):
                target = root / ("metadata" if metadata else "code")
                report = ensure_repository(target, str(source), revision, metadata_only=metadata)
                self.assertEqual(report["bare"], metadata)
                self.assertEqual(git(target, "show", f"{revision}:labels.json"), '{"fixture": true}')
                self.assertEqual((target / "labels.json").exists(), not metadata)
                ensure_repository(target, str(source), revision, metadata_only=metadata, check_only=True)

    def test_wrong_revision_origin_and_modified_source_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, revision = self.source(root)
            target = root / "code"
            ensure_repository(target, str(source), revision)
            with self.assertRaises(ValueError):
                ensure_repository(target, str(source), "0" * 40)
            with self.assertRaises(ValueError):
                ensure_repository(target, "unexpected-origin", revision)
            (target / "labels.json").write_text("local changes")
            with self.assertRaises(ValueError):
                ensure_repository(target, str(source), revision)
            self.assertEqual((target / "labels.json").read_text(), "local changes")
            self.assertEqual(git(target, "rev-parse", "HEAD"), revision)

    def test_reject_parent_repository_instead_of_using_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, revision = self.source(root)
            target = source / "not-a-clone"
            target.mkdir()
            (target / "keep").write_text("preserve")
            with self.assertRaises(ValueError):
                ensure_repository(target, str(source), revision)
            self.assertEqual((target / "keep").read_text(), "preserve")

    def test_checkpoint_hash_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            path.write_bytes(b"fixture")
            digest = hashlib.sha256(b"fixture").hexdigest()
            self.assertEqual(verify_checkpoint(path, digest, 7), digest)
            for expected_hash, size in (("0" * 64, 7), (digest, 8)):
                with self.assertRaises(ValueError):
                    verify_checkpoint(path, expected_hash, size)


if __name__ == "__main__":
    unittest.main()
