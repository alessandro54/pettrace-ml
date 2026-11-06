"""Reuse verified hashes within one preparation on local disk, never across kernels."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os

_ACTIVE = ContextVar('pettrace_file_verification', default=None)


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class _Index:
    def __init__(self):
        self.files = {}
        self.reads = self.hits = self.bytes_read = 0

    @property
    def stats(self):
        return {'files': len(self.files), 'hash_reads': self.reads,
                'hash_reuses': self.hits, 'bytes_read': self.bytes_read}


@contextmanager
def verification_session():
    """Share a hash map across preparation stages; nested calls share the same index.

    Each path is read once per unchanged device/inode/size/mtime/ctime signature.
    This assumes ordinary local-filesystem metadata and no concurrent writers;
    silent storage corruption without a metadata change needs a fresh hash scan.
    External archives/Drive transfers retain independent full-hash verification.
    """
    current = _ACTIVE.get()
    if current is not None:
        yield current
        return
    current = _Index()
    token = _ACTIVE.set(current)
    try:
        yield current
    finally:
        _ACTIVE.reset(token)
        if current.reads or current.hits:
            print(f'SHA256: {current.reads} reads · {current.hits} reused · '
                  f'{current.bytes_read / 2**30:.2f} GiB read.', flush=True)
        current.files.clear()


def sha256(path):
    """Hash actual bytes, or reuse this session's digest only while its signature is unchanged."""
    path = os.path.abspath(os.fspath(path))
    before = _signature(os.stat(path))
    index = _ACTIVE.get()
    saved = index.files.get(path) if index is not None else None
    if saved is not None and saved[0] == before:
        index.hits += 1
        return saved[1]
    with open(path, 'rb') as source:
        if _signature(os.fstat(source.fileno())) != before:
            raise ValueError('File changed before hashing')
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if _signature(os.fstat(source.fileno())) != before:
            raise ValueError('File changed while hashing')
    if _signature(os.stat(path)) != before:
        raise ValueError('File changed while hashing')
    if index is not None:
        index.files[path] = before, digest
        index.reads += 1
        index.bytes_read += before[2]
    return digest
