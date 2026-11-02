import json
import sqlite3

import pytest

from pettrace.data.checkpoints import DriveBackup, stage_archive


def test_incremental_restore_changed_files_and_ignore_uncommitted_blocks(tmp_path):
    root, drive = tmp_path / 'local', tmp_path / 'drive'
    root.mkdir()
    (root / 'photo.jpg').write_bytes(b'photo')
    (root / 'receipt.json').write_text('old')
    (root / '.env').write_text('never back up')
    (root / 'unfinished.partial').write_bytes(b'partial')
    backup = DriveBackup({'data': root}, drive, interval=0, block_bytes=5)
    backup(force=True)
    initial = len(list(drive.glob('*.json')))
    backup(force=True)
    assert len(list(drive.glob('*.json'))) == initial
    (root / 'receipt.json').write_text('new')
    backup(force=True)
    (drive / 'unconfirmed.tar').write_bytes(b'interrupted')
    restored = tmp_path / 'restored'
    fresh = DriveBackup({'data': restored}, drive)
    fresh.restore()
    assert (restored / 'photo.jpg').read_bytes() == b'photo'
    assert (restored / 'receipt.json').read_text() == 'new'
    assert sorted(p.name for p in restored.iterdir()) == ['photo.jpg', 'receipt.json']
    count = len(list(drive.glob('*.json')))
    fresh(force=True)
    assert len(list(drive.glob('*.json'))) == count


def test_sqlite_snapshot_is_consistent_with_open_connection(tmp_path):
    root = tmp_path / 'local'
    root.mkdir()
    with sqlite3.connect(root / 'features.sqlite3') as db:
        db.execute('CREATE TABLE features (value TEXT)')
        db.execute("INSERT INTO features VALUES ('saved')")
        db.commit()
        DriveBackup({'data': root}, tmp_path / 'drive')(force=True)
    restored = tmp_path / 'restored'
    DriveBackup({'data': restored}, tmp_path / 'drive').restore()
    with sqlite3.connect(restored / 'features.sqlite3') as db:
        assert db.execute('SELECT value FROM features').fetchall() == [('saved',)]


def test_corruption_and_unsafe_receipt_fail_before_restoration(tmp_path):
    root, drive = tmp_path / 'local', tmp_path / 'drive'
    root.mkdir()
    (root / 'photo').write_bytes(b'photo')
    DriveBackup({'data': root}, drive)(force=True)
    pack = next(drive.glob('*.tar'))
    pack.write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        DriveBackup({'data': tmp_path / 'restored'}, drive).restore()
    receipt_path = next(drive.glob('*.json'))
    receipt = json.loads(receipt_path.read_text())
    receipt['files']['data/../../escape'] = receipt['files'].pop('data/photo')
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='Unsafe'):
        DriveBackup({'data': tmp_path / 'restored'}, drive).restore()


def test_exception_checkpoints_completed_files(tmp_path):
    root = tmp_path / 'local'
    root.mkdir()
    with pytest.raises(RuntimeError, match='processing'):
        with DriveBackup({'data': root}, tmp_path / 'drive'):
            (root / 'complete').write_text('ok')
            raise RuntimeError('processing failed')
    assert list((tmp_path / 'drive').glob('*.json'))


def test_stage_archive_reuses_copy_and_detects_changed_source(tmp_path):
    source = tmp_path / 'drive.zip'
    source.write_bytes(b'first')
    target = tmp_path / 'scratch' / 'data.zip'
    assert stage_archive(source, target) == target
    before = target.stat().st_mtime_ns
    stage_archive(source, target)
    assert target.stat().st_mtime_ns == before
    source.write_bytes(b'changed')
    stage_archive(source, target)
    assert target.read_bytes() == b'changed'


def test_scoped_checkpoints_accumulate_until_due_and_full_exit_catches_other_files(tmp_path, monkeypatch):
    from pettrace.data import checkpoints

    root, drive = tmp_path / 'local', tmp_path / 'drive'
    root.mkdir()
    tracked, untracked = root / 'photo', root / 'review.csv'
    tracked.write_bytes(b'photo')
    untracked.write_text('review')
    now = [0]
    monkeypatch.setattr(checkpoints.time, 'monotonic', lambda: now[0])
    backup = DriveBackup({'data': root}, drive, interval=300)
    backup(paths=[tracked])
    assert not list(drive.glob('*.json'))
    now[0] = 301
    backup(paths=[])
    receipt = json.loads(next(drive.glob('*.json')).read_text())
    assert set(receipt['files']) == {'data/photo'}
    backup(force=True)
    restored = tmp_path / 'restored'
    DriveBackup({'data': restored}, drive).restore()
    assert (restored / 'review.csv').read_text() == 'review'
    assert (restored / 'photo').read_bytes() == b'photo'


def test_scoped_checkpoint_does_not_scan_raw_and_retries_failed_backup(tmp_path, monkeypatch):
    from pathlib import Path
    root, drive = tmp_path / 'local', tmp_path / 'drive'
    cache = root / 'cache'
    cache.mkdir(parents=True)
    (cache / 'result.json').write_text('{}')
    (root / 'raw').mkdir()
    backup = DriveBackup({'data': root}, drive, interval=0)
    walk = Path.rglob
    scanned = []

    def scan(path, pattern):
        scanned.append(path)
        return walk(path, pattern)

    monkeypatch.setattr(Path, 'rglob', scan)
    save = backup._save_block
    with monkeypatch.context() as patch:
        patch.setattr(backup, '_save_block', lambda _: (_ for _ in ()).throw(OSError('Drive unavailable')))
        with pytest.raises(OSError, match='Drive unavailable'):
            backup(paths=[cache])
    assert not list(drive.glob('*.json'))
    backup(paths=[])
    assert scanned == [cache, cache]
    assert list(drive.glob('*.json'))
    assert backup._save_block == save
    with pytest.raises(ValueError, match='outside'):
        backup(paths=[tmp_path / 'outside'])
    (cache / '.env').write_text('secret')
    (cache / 'escape').symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError, match='escapes|symlinks'):
        backup(paths=[cache])


def test_prepared_only_restore_does_not_resolve_raw_paths(tmp_path, monkeypatch):
    data, prepared, drive = (tmp_path / name for name in ('data', 'prepared', 'drive'))
    data.mkdir()
    prepared.mkdir()
    (data / 'photo.jpg').write_bytes(b'raw')
    (prepared / 'manifest.csv').write_text('refined')
    DriveBackup({'data': data, 'prepared': prepared}, drive, block_bytes=1)(force=True)
    fresh = DriveBackup({'data': tmp_path / 'new-data', 'prepared': tmp_path / 'new-prepared'}, drive)
    resolved = []
    target = fresh._target

    def resolve(name):
        resolved.append(name)
        return target(name)

    monkeypatch.setattr(fresh, '_target', resolve)
    fresh.restore(only={'prepared'})
    assert resolved and all(name.startswith('prepared/') for name in resolved)
    assert not (tmp_path / 'new-data').exists()
    assert (tmp_path / 'new-prepared/manifest.csv').read_text() == 'refined'


def test_exact_file_restore_uses_latest_copy_and_skips_other_packs(tmp_path):
    root, drive = tmp_path / 'data', tmp_path / 'drive'
    root.mkdir()
    review = root / 'source-100-seed42/review.csv'
    review.parent.mkdir()
    review.write_text('old inventory')
    (root / 'photo.jpg').write_bytes(b'photo')
    backup = DriveBackup({'data': root}, drive, block_bytes=1)
    backup(force=True)
    review.write_text('complete inventory')
    backup(force=True, paths=[review])
    receipts = sorted(drive.glob('*.json'))
    # Only the last pack is needed: neither raw photos nor the older CSV is read.
    for path in receipts[:-1]:
        path.with_suffix('.tar').write_bytes(b'unavailable old pack')
    restored = tmp_path / 'restored'
    fresh = DriveBackup({'data': restored}, drive)
    member = 'data/source-100-seed42/review.csv'
    fresh.restore(files={member})
    assert (restored / 'source-100-seed42/review.csv').read_text() == 'complete inventory'
    assert not (restored / 'photo.jpg').exists()
    receipts[-1].with_suffix('.tar').write_bytes(b'corrupted requested pack')
    fresh.restore(files={member})  # Verified local copy needs no TAR transfer.
    (restored / 'source-100-seed42/review.csv').unlink()
    with pytest.raises(ValueError, match='checksum'):
        fresh.restore(files={member})


def test_exact_file_restore_rejects_missing_volume_and_unsafe_path(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'review.csv').write_text('other inventory')
    drive = tmp_path / 'drive'
    DriveBackup({'data': root}, drive)(force=True)
    target = tmp_path / 'new-data'
    fresh = DriveBackup({'data': target, 'prepared': tmp_path / 'prepared'}, drive)
    with pytest.raises(FileNotFoundError, match='source-100'):
        fresh.restore(files={'data/source-100-seed42/review.csv'})
    assert not target.exists()
    with pytest.raises(ValueError, match='Unsafe'):
        fresh.restore(files={'data/../escape'})
    with pytest.raises(ValueError, match='outside'):
        fresh.restore(only={'prepared'}, files={'data/review.csv'})
