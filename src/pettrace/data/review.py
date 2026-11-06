"""An explicit, resumable photo review for the E3 public identity pilot."""

import base64
import csv
import html
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from pettrace.data.sources import REVIEW_FIELDS
from pettrace.data.curation import sha256


def _read(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != REVIEW_FIELDS:
            raise ValueError("Unexpected review columns; regenerate the review from the original selection")
        rows = list(reader)
    if not rows or any(set(row) != set(REVIEW_FIELDS) or None in row.values() for row in rows):
        raise ValueError("Empty or malformed review CSV")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Review contains duplicate paths")
    return rows


def review_status(path):
    rows = _read(path)
    counts = Counter(row["decision"] for row in rows)
    return {"total": len(rows), "keep": counts["keep"], "drop": counts["drop"],
            "pending": sum(row["decision"] not in {"keep", "drop"} for row in rows)}


def exclusion_summary(snapshot):
    """Count recorded exclusions without decoding images or rerunning any filter."""
    snapshot = Path(snapshot)
    audit = json.loads((snapshot / "audit.json").read_text())
    corrupt = {row["path"] for row in audit["corrupt"]}
    groups, reasons = defaultdict(list), Counter()
    with (snapshot / "excluded.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            reasons[row["reason"]] += 1
            # Older snapshots use insufficient_distinct_photos as a fallback for decode failures.
            if row["reason"] == "insufficient_distinct_photos" and row["path"] not in corrupt:
                groups[(row["species"], row["animal"])].append(row["path"])
    histogram = {species: dict(sorted(Counter(len(photos) for (sp, _), photos in groups.items()
                                             if sp == species).items())) for species in ("cat", "dog")}
    minimum = audit["minimum_photos"]
    if any(len(photos) >= minimum for photos in groups.values()):
        raise ValueError("Exclusion counts contradict the recorded minimum")
    recoverable = [photos for photos in groups.values() if 2 <= len(photos) < minimum]
    return {"exclusion_reasons": dict(reasons), "minimum_photos": minimum,
            "excluded_identities_by_distinct_photos": histogram,
            "additional_identities_at_minimum_2": len(recoverable),
            "additional_photos_at_minimum_2": sum(map(len, recoverable)),
            "quarantined_identities": len(audit["conflicting_identities_quarantined"]),
            "conflicting_hashes": len(audit["conflicting_hashes"]),
            "note": "Counterfactual from recorded filters, not a new validated dataset or verified identity labels."}


def quarantine_groups(snapshot):
    """Group labels connected by exact images or proposed near copies, never merge them."""
    audit = json.loads((Path(snapshot) / "audit.json").read_text())
    conflicts = audit["conflicting_hashes"]
    by_animal, labels = defaultdict(set), {}
    for digest, rows in conflicts.items():
        animals = {row["animal"] for row in rows}
        if len(animals) < 2:
            raise ValueError("A conflicting hash must involve at least two identity labels")
        labels[digest] = animals
        for animal in animals:
            by_animal[animal].add(digest)
    near = {}
    for pair in audit.get("near_copy_pairs", []):
        left, right = pair["left"], pair["right"]
        evidence = audit.get("near_copy_labels", {})
        if left == right or not evidence.get(left) or not evidence.get(right):
            raise ValueError("Near-copy link requires two recorded images and their labels")
        link = (left, right)
        near[link] = pair
        labels[link] = set(evidence[left]) | set(evidence[right])
        for animal in labels[link]:
            by_animal[animal].add(link)
    unseen, groups = set(by_animal), []
    while unseen:
        pending = [unseen.pop()]
        animals, hashes = set(pending), set()
        while pending:
            animal = pending.pop()
            for digest in by_animal[animal] - hashes:
                hashes.add(digest)
                neighbors = labels[digest] & unseen
                unseen.difference_update(neighbors)
                animals.update(neighbors)
                pending.extend(neighbors)
        group = {"animals": sorted(animals),
                 "shared_hashes": {digest: sorted(labels[digest]) for digest in sorted(h for h in hashes if isinstance(h, str))}}
        links = sorted(h for h in hashes if isinstance(h, tuple))
        if links:
            group["near_pairs"] = [near[link] for link in links]
            group["near_hashes"] = {digest: sorted(audit["near_copy_labels"][digest])
                                    for digest in sorted({h for link in links for h in link})}
        groups.append(group)
    return sorted(groups, key=lambda group: min(set(group["shared_hashes"]) | set(group.get("near_hashes", {}))))


def quarantine_gallery(snapshot, output, *, start=0, count=8, representatives_only=False):
    """Export a bounded, offline HTML review batch; never approve or change a snapshot.

    Requires the original source images at their recorded paths. Every photo of each
    involved identity is shown once per hash, with all its labels and exclusions.
    Connected labels form one case; indirect links do not establish animal identity.
    representatives_only verifies one available original per recorded hash, useful
    after selective recovery. Other copies' labels come from the inventory only.
    """
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if output.is_relative_to(snapshot):
        raise ValueError("Write the review outside the immutable snapshot")
    if output.suffix.lower() != ".html":
        raise ValueError("Use an .html output file")
    if start < 0 or count < 1 or count > 50:
        raise ValueError("Use start >= 0 and 1–50 groups per batch")
    cases = quarantine_groups(snapshot)
    chosen = cases[start:start + count]
    if not chosen:
        raise ValueError("No conflicts in this batch")
    animals = {animal for case in chosen for animal in case["animals"]}
    groups = defaultdict(list)
    with (snapshot / "review.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["animal"] in animals:
                groups[row["animal"]].append(row)
    reasons = {}
    if (snapshot / "excluded.csv").is_file():
        with (snapshot / "excluded.csv").open(newline="") as stream:
            reasons = {row["path"]: row["reason"] for row in csv.DictReader(stream) if row["animal"] in animals}
    content = ['<!doctype html><html lang="en"><meta charset="utf-8">',
               '<title>PetTrace · Identity conflicts</title>',
               '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 16px}'
               '.photos{display:flex;flex-wrap:wrap;gap:12px}figure{width:260px;margin:0 0 16px;overflow-wrap:anywhere}'
               'img{max-width:260px;max-height:240px}figcaption{font-size:13px}.shared{outline:4px solid #d97706}'
               'section{border-top:2px solid #aaa;margin-top:28px}code{overflow-wrap:anywhere}</style>',
               '<h1>Identity conflict review</h1>',
               f'<p>Inventory: <code>{html.escape(snapshot.name)}</code>. '
               f'Groups {start + 1}–{start + len(chosen)} of {len(cases)} '
               '(ordered by the first linked SHA256).</p>',
               '<p>The orange border marks a shared image or a possible near copy. '
               'Each image appears once with all its labels. Indirect links do not prove that '
               'the animal is the same: some photos may be mislabeled. '
               'All cases remain pending; opening this viewer approves nothing.</p>']
    if representatives_only:
        content.append('<p>One original per SHA256 was verified; labels for the other copies '
                       'come from the inventory and their files are not read again.</p>')
    for number, case in enumerate(chosen, start + 1):
        photos_by_hash = defaultdict(list)
        linked_hashes = case["shared_hashes"] | case.get("near_hashes", {})
        for animal in case["animals"]:
            photos = groups[animal]
            if not photos or not any(row["sha256"] in linked_hashes for row in photos):
                raise ValueError("Conflict does not match the recorded review")
            for row in photos:
                photos_by_hash[row["sha256"]].append(row)
        for digest, expected_animals in linked_hashes.items():
            if not set(expected_animals) <= {row["animal"] for row in photos_by_hash.get(digest, [])}:
                raise ValueError("Conflict does not match the recorded review")
        total = sum(map(len, photos_by_hash.values()))
        content.append(f'<section><h2>Group {number} · pending</h2>'
                       f'<p>{len(case["animals"])} labels · {total} files · '
                       f'{len(photos_by_hash)} distinct images by SHA256.</p>'
                       f'<p>Labels: {html.escape(", ".join(case["animals"]))}</p>')
        if case.get("near_pairs"):
            content.append('<p>Possible resized/recompressed copies pending review:</p><ul>')
            for pair in case["near_pairs"]:
                content.append(f'<li>{html.escape(pair["left"])} ↔ {html.escape(pair["right"])}'
                               f' · dHash distance {int(pair["hash_distance"])} · RGB MAE {float(pair["rgb_mae"]):.3f}</li>')
            content.append('</ul>')
        content.append('<div class="photos">')
        for digest, photos in sorted(photos_by_hash.items()):
            shown = next((row for row in photos if Path(row["path"]).is_file()), photos[0]) if representatives_only else photos[0]
            for row in [shown] if representatives_only else photos:
                path = Path(row["path"])
                if sha256(path) != digest:
                    raise ValueError(f"Source photo changed: {path.name}")
            try:
                with Image.open(shown["path"]) as image:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (520, 480))
                    stream = io.BytesIO()
                    thumb.save(stream, format="JPEG", quality=80)
                preview = '<img loading="lazy" alt="Photo to review" src="data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode() + '">'
            except (OSError, ValueError, Image.DecompressionBombError):
                preview = '<p>Unreadable image; inspect the original.</p>'
            shared = ' class="shared"' if digest in linked_hashes else ''
            content.append(f'<figure{shared}>{preview}<figcaption><ul>')
            for row in photos:
                content.append(f'<li>{html.escape(row["animal"])} · {html.escape(Path(row["path"]).name)}'
                               f' · {html.escape(reasons.get(row["path"], "no recorded exclusion"))}</li>')
            content.append(f'</ul>SHA256: {html.escape(digest)}</figcaption></figure>')
        content.append('</div></section>')
    content.append('<p>No photos, labels, or decisions were changed. Corrections require a new version.</p></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(content))
    return output


class ReviewPending(ValueError):
    """Show actionable review instructions instead of an implementation traceback."""

    def _render_traceback_(self):
        return [str(self)]


def require_review_complete(path):
    status = review_status(path)
    if status["pending"]:
        raise ReviewPending(
            f"Review pending: {status['pending']}/{status['total']} photos have no keep/drop decision. "
            "Use the viewer in section 1.1, save the decisions, and run this cell again. "
            "Training has not started."
        )
    return status


def save_decisions(path, updates, *, expected_sha256):
    """Save explicit choices, rejecting stale CSVs, changed bytes and label edits."""
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError("The review changed in another window. Reopen the viewer before saving")
    rows = _read(path)
    by_path = {row["path"]: row for row in rows}
    if not updates.keys() <= by_path.keys():
        raise ValueError("Unknown photo in review update")
    for image_path, update in updates.items():
        if set(update) != {"decision", "note"} or update["decision"] not in {"", "keep", "drop"}:
            raise ValueError("Only explicit decision/note updates are allowed")
        if not isinstance(update["note"], str):
            raise ValueError("Review note must be text")
        if sha256(image_path) != by_path[image_path]["sha256"]:
            raise ValueError("The photo changed after selection; a new review is required")
        by_path[image_path].update(update)
    # Recheck before replacement so another viewer's saved choices are not overwritten.
    if sha256(path) != expected_sha256:
        raise ValueError("The review changed while it was being saved. Reopen the viewer")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256(path)


def review_with_prompts(path):
    """Review using standard notebook images/stdin when widget rendering is unavailable."""
    from IPython.display import clear_output, display

    path = Path(path)
    rows = _read(path)
    digest = sha256(path)
    groups = defaultdict(list)
    for row in rows:
        groups[row["animal"]].append(row)
    choices = {"k": "keep", "keep": "keep", "d": "drop", "drop": "drop"}
    for number, (animal, photos) in enumerate(groups.items(), 1):
        if all(row["decision"] in {"keep", "drop"} for row in photos):
            continue
        clear_output(wait=True)
        print(f"{number}/{len(groups)} · {animal}")
        print("Check identity, unambiguous subject, body coverage, quality, and variety. "
              "At least 4 distinct photos must remain after curation.")
        columns, width, height = 4, 300, 320
        sheet = Image.new("RGB", (columns * width, ((len(photos) + columns - 1) // columns) * height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(photos):
            x, y = index % columns * width, index // columns * height
            try:
                with Image.open(row["path"]) as image:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (290, 280))
                sheet.paste(thumb, (x, y))
            except (OSError, ValueError, Image.DecompressionBombError):
                draw.rectangle((x, y, x + 290, y + 280), fill="#ffe5e5")
                draw.text((x + 8, y + 120), "UNREADABLE: mark d", fill="#8b0000")
            draw.text((x + 4, y + 282), f"{index + 1}. {Path(row['path']).name[:32]}", fill="black")
            draw.text((x + 4, y + 299), f"Current: {row['decision'] or 'pending'}", fill="black")
        display(sheet)
        print("One decision per photo, in order: k=keep, d=drop. Example: k d k k.")
        print("q: exit while preserving saved work. s: postpone this identity without changing it.")
        while True:
            response = input(f"Decisions for {len(photos)} photos: ").strip().lower()
            if response == "q":
                print("Review paused. Previous decisions are saved in review.csv.")
                return review_status(path)
            if response == "s":
                break
            tokens = response.split()
            if len(tokens) != len(photos) or any(token not in choices for token in tokens):
                print(f"Enter {len(photos)} space-separated k/d decisions, or q/s.")
                continue
            updates = {row["path"]: {"decision": choices[token], "note": row["note"]}
                       for row, token in zip(photos, tokens)}
            digest = save_decisions(path, updates, expected_sha256=digest)
            for row in photos:
                row.update(updates[row["path"]])
            print("Saved to review.csv.")
            break
    status = review_status(path)
    print(f"Review saved: {status['total'] - status['pending']}/{status['total']}; "
          f"pending: {status['pending']}.")
    return status


def photo_reviewer(path):
    """Display one identity at a time. No photo is approved by default."""
    import ipywidgets as widgets

    path = Path(path)
    rows = _read(path)
    groups = defaultdict(list)
    for row in rows:
        groups[row["animal"]].append(row)
    animals = list(groups)
    first = next((i for i, animal in enumerate(animals)
                  if any(row["decision"] not in {"keep", "drop"} for row in groups[animal])), 0)
    state = {"index": first, "digest": sha256(path)}
    heading, progress, message = widgets.HTML(), widgets.HTML(), widgets.HTML()
    cards = widgets.Box(layout=widgets.Layout(display="flex", flex_flow="row wrap"))
    previous = widgets.Button(description="Save and previous", layout=widgets.Layout(width="180px"))
    save = widgets.Button(description="Save", button_style="info")
    following = widgets.Button(description="Save and next", layout=widgets.Layout(width="180px"))
    controls = []

    def update_progress():
        pending = sum(row["decision"] not in {"keep", "drop"} for row in rows)
        keep = sum(row["decision"] == "keep" for row in groups[animals[state["index"]]])
        progress.value = (f"<p><b>{len(rows) - pending}/{len(rows)} photos reviewed</b> · {pending} pending. "
                          f"This identity: {keep} kept; at least 4 distinct photos after deduplication.</p>")

    def close_tree(widget):
        for child in getattr(widget, "children", ()):
            close_tree(child)
        widget.close()

    def render():
        controls.clear()
        old_cards = cards.children
        cards.children = ()
        for card in old_cards:
            close_tree(card)
        animal = animals[state["index"]]
        heading.value = f"<h3>{state['index'] + 1}/{len(animals)} · {html.escape(animal)}</h3>"
        tiles = []
        for index, row in enumerate(groups[animal], 1):
            try:
                with Image.open(row["path"]) as image:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (320, 280))
                    output = io.BytesIO()
                    thumb.save(output, format="JPEG", quality=90)
                preview = widgets.Image(value=output.getvalue(), format="jpeg")
            except (OSError, ValueError, Image.DecompressionBombError):
                preview = widgets.HTML("<p><b>Unreadable file.</b> Mark as drop.</p>")
            choice = widgets.Dropdown(options=[("Pending", ""), ("Keep", "keep"),
                                               ("Drop", "drop")],
                                      value=row["decision"] if row["decision"] in {"keep", "drop"} else "",
                                      layout=widgets.Layout(width="100%"))
            note = widgets.Text(value=row["note"], placeholder="Reason / note", layout=widgets.Layout(width="100%"))
            controls.append((row, choice, note))
            label = widgets.HTML(f"<p>{index}. {html.escape(Path(row['path']).name)}</p>")
            tiles.append(widgets.VBox([label, preview, choice, note],
                                      layout=widgets.Layout(width="330px", border="1px solid #888", padding="10px", margin="8px")))
        cards.children = tuple(tiles)
        previous.disabled = state["index"] == 0
        following.disabled = state["index"] == len(animals) - 1
        update_progress()

    def persist(step=0):
        try:
            updates = {row["path"]: {"decision": choice.value, "note": note.value}
                       for row, choice, note in controls}
            state["digest"] = save_decisions(path, updates, expected_sha256=state["digest"])
            for row, choice, note in controls:
                row.update(decision=choice.value, note=note.value)
            state["index"] += step
            if step:
                render()
            else:
                update_progress()
            message.value = "<p>Decisions saved to review.csv.</p>"
        except (OSError, ValueError) as exc:
            message.value = f"<p><b>Not saved:</b> {html.escape(str(exc))}</p>"

    previous.on_click(lambda _: persist(-1))
    save.on_click(lambda _: persist())
    following.on_click(lambda _: persist(1))
    render()
    help_text = widgets.HTML("<p>Check consistent identity, an unambiguous subject, useful body coverage, "
                             "quality, and variety. Photos with multiple animals require a verified subject. "
                             "Save before closing; reopening resumes at the first pending identity.</p>")
    return widgets.VBox([heading, progress, help_text, cards,
                         widgets.HBox([previous, save, following]), message])
