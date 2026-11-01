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
