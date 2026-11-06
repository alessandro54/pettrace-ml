"""Immutable incremental archives on mounted Drive; computation stays on local disk."""

import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pettrace.data.verification import sha256 as verified_sha256


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _safe(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or len(path.parts) < 2:
        raise ValueError('Unsafe checkpoint path')
    return path


class BackupNotFoundError(FileNotFoundError):
    """No confirmed backup exists for the requested files."""


class DriveBackup:
    """One writer per backup directory. Only confirmed archives can be restored.

    Call periodically at completed work units, and use as a context manager to
    save on normal exit or Python exceptions. Abrupt VM loss retains the last
    confirmed checkpoint, not the unfinished work since that checkpoint.
    """

    def __init__(self, roots, destination, *, interval=300, block_bytes=512 * 2**20):
        self.roots = {name: Path(path).resolve() for name, path in roots.items()}
        if not self.roots or any('/' in name or name in {'', '.', '..'} for name in self.roots):
            raise ValueError('Use simple named checkpoint roots')
        self.destination = Path(destination).resolve()
        if interval < 0 or block_bytes <= 0:
            raise ValueError('Invalid checkpoint interval or block size')
        if any(self.destination.is_relative_to(p) or p.is_relative_to(self.destination)
               for p in self.roots.values()):
            raise ValueError('Backups and working directories must be separate')
        self.interval, self.block_bytes = interval, block_bytes
        self._saved = {}
        self._pending_paths = set()
        self._last = time.monotonic()

    def _target(self, name):
        relative = _safe(name)
        root = self.roots.get(relative.parts[0])
        if root is None:
            raise ValueError('Unknown checkpoint root')
        target = root.joinpath(*relative.parts[1:])
        if not target.resolve().is_relative_to(root):
            raise ValueError('Checkpoint path escapes working directory')
        return target

    def inventory(self):
        """List latest confirmed members without reading archives or local photo bytes."""
        latest = {}
        for path in sorted(self.destination.glob('*.json')):
            receipt = json.loads(path.read_text())
            if receipt.get('schema') != 1 or receipt.get('archive') != path.with_suffix('.tar').name:
                raise ValueError('Invalid checkpoint receipt')
            for name, info in receipt['files'].items():
                if _safe(name).parts[0] not in self.roots:
                    raise ValueError('Unknown checkpoint root')
                latest[name] = info
        return latest

    def restore(self, *, only=None, files=None):
        """Restore selected roots or exact archive paths, verifying confirmed packs."""
        if only is not None and not set(only) <= self.roots.keys():
            raise ValueError("Unknown restore root")
        if files is not None:
            files = set(files)
            for name in files:
                self._target(name)
                if only is not None and _safe(name).parts[0] not in only:
                    raise ValueError("Requested file is outside the selected restore roots")
        latest, receipts = {}, {}
        for receipt_path in sorted(self.destination.glob('*.json')):
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('schema') != 1 or receipt.get('archive') != receipt_path.with_suffix('.tar').name:
                raise ValueError('Invalid checkpoint receipt')
            for name, info in receipt['files'].items():
                if files is not None and name not in files:
                    continue
                label = _safe(name).parts[0]
                if label not in self.roots:
                    raise ValueError('Unknown checkpoint root')
                if only is None or label in only:
                    self._target(name)
                    receipts[receipt_path.name] = receipt
                    latest[name] = (receipt_path.name, info)
        if files is not None and files - latest.keys():
            raise BackupNotFoundError(f"No confirmed Drive backup for: {', '.join(sorted(files - latest.keys()))}")
        needed = {}
        for name, (receipt_name, info) in latest.items():
            target = self._target(name)
            if not target.is_file() or target.stat().st_size != info['size'] or verified_sha256(target) != info['sha256']:
                needed.setdefault(receipt_name, set()).add(name)
        for receipt_name, names in needed.items():
            receipt = receipts[receipt_name]
            with tempfile.TemporaryDirectory(prefix='pettrace-restore-') as folder:
                local = Path(folder) / 'block.tar'
                shutil.copyfile(self.destination / receipt['archive'], local)
                if _sha(local) != receipt['sha256']:
                    raise ValueError('Checkpoint archive checksum mismatch')
                with tarfile.open(local) as archive:
                    members = archive.getmembers()
                    if len({m.name for m in members}) != len(members) or {m.name for m in members} != set(receipt['files']):
                        raise ValueError('Checkpoint inventory mismatch')
                    for member in members:
                        self._target(member.name)
                        if not member.isfile() or member.size != receipt['files'][member.name]['size']:
                            raise ValueError('Unsafe checkpoint member')
                    for member in members:
                        if member.name not in names:
                            continue
                        target = self._target(member.name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                            partial = Path(stream.name)
                            with archive.extractfile(member) as source:
                                shutil.copyfileobj(source, stream)
                        try:
                            if _sha(partial) != receipt['files'][member.name]['sha256']:
                                raise ValueError('Checkpoint member checksum mismatch')
                            partial.replace(target)
                        finally:
                            partial.unlink(missing_ok=True)
                print(f'Drive: restored block ({len(names)} files).', flush=True)
        for name in latest:
            self._saved[name] = self._signature(self._target(name))
        print(f'Drive: verified {len(latest)} files; backup ready.', flush=True)

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def __call__(self, *, force=False, paths=None):
        """Back up supplied files/subtrees, or scan all roots when paths is omitted.

        Paths accumulate until the next checkpoint; context exit always scans all
        roots so exceptions and callers without change tracking remain recoverable.
        """
        if paths is not None:
            for path in paths:
                path = Path(path).absolute()
                match = next(((label, root) for label, root in self.roots.items() if path.is_relative_to(root)), None)
                if match is None:
                    raise ValueError('Checkpoint path is outside its roots')
                label, root = match
                if path != root:
                    self._target(f'{label}/{path.relative_to(root).as_posix()}')
                if path.is_symlink():
                    raise ValueError('Checkpoint roots must contain regular files, not symlinks')
                self._pending_paths.add(path)
        if not force and time.monotonic() - self._last < self.interval:
            return
        def candidates():
            for label, root in self.roots.items():
                if not root.exists():
                    continue
                selected = {root} if paths is None else {p for p in self._pending_paths if p.is_relative_to(root)}
                for path in sorted(selected):
                    if any(parent in selected for parent in path.parents):
                        continue
                    yield from ((label, p) for p in (path.rglob('*') if path.is_dir() else [path]))

        pending, size = [], 0
        for label, path in candidates():
            relative = path.relative_to(self.roots[label])
            # Incomplete outputs, Hub internals and SQLite journals are not snapshots.
            if any(p.startswith('.') for p in relative.parts) or path.name.endswith(('.partial', '-wal', '-shm', '-journal')):
                continue
            name = f'{label}/{relative.as_posix()}'
            if path.is_symlink():
                raise ValueError('Checkpoint roots must contain regular files, not symlinks')
            if not path.is_file():
                continue
            signature = self._signature(path)
            if self._saved.get(name) == signature:
                continue
            pending.append((name, path, signature))
            size += signature[0]
            if size >= self.block_bytes:
                self._save_block(pending)
                pending, size = [], 0
        if pending:
            self._save_block(pending)
        self._pending_paths.clear()
        self._last = time.monotonic()

    def _save_block(self, files):
        self.destination.mkdir(parents=True, exist_ok=True)
        key = f'{time.time_ns():020d}-{uuid4().hex}'
        with tempfile.TemporaryDirectory(prefix='pettrace-checkpoint-') as folder:
            folder = Path(folder)
            pack = folder / f'{key}.tar'
            inventory = {}
            with tarfile.open(pack, 'w') as archive:
                for name, path, signature in files:
                    source = path
                    if path.suffix == '.sqlite3':
                        source = folder / 'snapshot.sqlite3'
                        source.unlink(missing_ok=True)
                        with sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True) as db, sqlite3.connect(source) as copy:
                            db.backup(copy)
                    inventory[name] = {'size': source.stat().st_size, 'sha256': _sha(source)}
                    archive.add(source, arcname=name, recursive=False)
                    if self._signature(path) != signature:
                        raise RuntimeError('File changed during backup; retry at a completed work unit')
            digest = _sha(pack)
            remote = self.destination / pack.name
            partial = remote.with_suffix('.tar.partial')
            shutil.copyfile(pack, partial)
            if _sha(partial) != digest:
                raise IOError('Drive backup verification failed')
            partial.replace(remote)
            receipt = {'schema': 1, 'archive': remote.name, 'sha256': digest, 'files': inventory}
            temporary = remote.with_suffix('.json.partial')
            temporary.write_text(json.dumps(receipt, sort_keys=True))
            if json.loads(temporary.read_text()) != receipt:
                raise IOError('Drive receipt verification failed')
            temporary.replace(remote.with_suffix('.json'))
        for name, _, signature in files:
            self._saved[name] = signature
        print(f'Drive: backup confirmed ({len(files)} files, {pack.name}).', flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self(force=True)
        return False


def stage_archive(source, destination):
    """Copy a Drive ZIP to local scratch; acquisition verifies its pinned index and CRCs."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('ZIP staging must use a separate local destination')
    stat = source.stat()
    signature = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    receipt = destination.with_suffix(destination.suffix + '.json')
    if destination.is_file() and receipt.is_file() and json.loads(receipt.read_text()) == signature:
        if destination.stat().st_size == stat.st_size:
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < stat.st_size + 64 * 2**20:
        raise OSError("Insufficient local disk space to stage the source ZIP")
    partial = destination.with_suffix(destination.suffix + '.partial')
    with source.open('rb') as incoming, partial.open('wb') as outgoing:
        copied, last = 0, time.monotonic()
        while block := incoming.read(8 * 2**20):
            outgoing.write(block)
            copied += len(block)
            if time.monotonic() - last >= 5:
                print(f'\rZIP {source.name}: {copied / 2**30:.1f}/{stat.st_size / 2**30:.1f} GiB', end='', flush=True)
                last = time.monotonic()
    if source.stat().st_mtime_ns != stat.st_mtime_ns or copied != stat.st_size:
        raise IOError('Source ZIP changed during transfer')
    partial.replace(destination)
    receipt.write_text(json.dumps(signature))
    print(f'\rZIP {source.name}: local copy ready; index and CRC will be verified during extraction.', flush=True)
    return destination
