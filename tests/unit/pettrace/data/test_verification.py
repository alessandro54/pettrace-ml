import hashlib
import os

import pytest

from pettrace.data import verification


def test_nested_sessions_share_one_verified_hash(tmp_path):
    photo = tmp_path / 'photo.jpg'
    photo.write_bytes(b'photo bytes')
    expected = hashlib.sha256(photo.read_bytes()).hexdigest()
    with verification.verification_session() as index:
        assert verification.sha256(photo) == expected
        with verification.verification_session() as nested:
            assert nested is index
            assert verification.sha256(photo) == expected
            assert verification.sha256(photo) == expected
            assert verification.sha256(photo) == expected
        assert index.stats == {'files': 1, 'hash_reads': 1, 'hash_reuses': 3, 'bytes_read': photo.stat().st_size}
    assert verification._ACTIVE.get() is None
    with verification.verification_session() as fresh:
        assert verification.sha256(photo) == expected
        assert fresh.reads == 1 and fresh.hits == 0


def test_same_size_overwrite_with_restored_mtime_invalidates_hash(tmp_path):
    photo = tmp_path / 'photo.jpg'
    photo.write_bytes(b'first')
    with verification.verification_session() as index:
        before = photo.stat()
        original = verification.sha256(photo)
        photo.write_bytes(b'other')
        os.utime(photo, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert photo.stat().st_size == before.st_size
        assert photo.stat().st_mtime_ns == before.st_mtime_ns
        assert verification.sha256(photo) == hashlib.sha256(b'other').hexdigest() != original
        assert index.reads == 2 and index.hits == 0


def test_atomic_replacement_invalidates_hash_and_missing_path_is_not_a_cache_hit(tmp_path):
    photo, replacement = tmp_path / 'photo', tmp_path / 'replacement'
    photo.write_bytes(b'first')
    with verification.verification_session() as index:
        before = photo.stat()
        verification.sha256(photo)
        replacement.write_bytes(b'other')
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(photo)
        assert verification.sha256(photo) == hashlib.sha256(b'other').hexdigest()
        assert index.reads == 2
        photo.unlink()
        with pytest.raises(FileNotFoundError):
            verification.sha256(photo)


def test_concurrent_change_is_rejected_and_exception_discards_index(tmp_path, monkeypatch):
    photo = tmp_path / 'photo'
    photo.write_bytes(b'first')
    digest = hashlib.file_digest
    def change_during_read(stream, algorithm):
        result = digest(stream, algorithm)
        photo.write_bytes(b'changed during read')
        return result
    monkeypatch.setattr(hashlib, 'file_digest', change_during_read)
    with pytest.raises(ValueError, match='changed while hashing'):
        with verification.verification_session() as index:
            verification.sha256(photo)
    assert not index.files and verification._ACTIVE.get() is None


def test_no_session_means_every_call_reads_actual_bytes(tmp_path, monkeypatch):
    photo = tmp_path / 'photo'
    photo.write_bytes(b'photo')
    reads = []
    digest = hashlib.file_digest
    def count(stream, algorithm):
        reads.append(stream.name)
        return digest(stream, algorithm)
    monkeypatch.setattr(hashlib, 'file_digest', count)
    for _ in range(3):
        verification.sha256(photo)
    assert reads == [str(photo)] * 3
