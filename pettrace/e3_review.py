"""An explicit, resumable photo review for the E3 public identity pilot."""

import csv
import html
import io
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from pettrace.e3_data import REVIEW_FIELDS
from pettrace.public_data import sha256


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


class ReviewPending(ValueError):
    """Show actionable review instructions instead of an implementation traceback."""

    def _render_traceback_(self):
        return [str(self)]


def require_review_complete(path):
    status = review_status(path)
    if status["pending"]:
        raise ReviewPending(
            f"Revisión pendiente: {status['pending']}/{status['total']} fotos sin keep/drop. "
            "Usa el visor de la sección 1.1, guarda las decisiones y vuelve a ejecutar esta celda. "
            "El entrenamiento todavía no comenzó."
        )
    return status


def save_decisions(path, updates, *, expected_sha256):
    """Save explicit choices, rejecting stale CSVs, changed bytes and label edits."""
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError("La revisión cambió en otra ventana. Vuelve a abrir el visor antes de guardar")
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
            raise ValueError("La foto cambió desde la selección; requiere una revisión nueva")
        by_path[image_path].update(update)
    # Recheck before replacement so another viewer's saved choices are not overwritten.
    if sha256(path) != expected_sha256:
        raise ValueError("La revisión cambió mientras se guardaba. Vuelve a abrir el visor")
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
        print("Revisar identidad, objetivo inequívoco, cuerpo, calidad y variedad. Mínimo 4 fotos distintas tras curación.")
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
                draw.text((x + 8, y + 120), "ILEGIBLE: marcar d", fill="#8b0000")
            draw.text((x + 4, y + 282), f"{index + 1}. {Path(row['path']).name[:32]}", fill="black")
            draw.text((x + 4, y + 299), f"Actual: {row['decision'] or 'pendiente'}", fill="black")
        display(sheet)
        print("Una decisión por foto, en orden: k=conservar, d=descartar. Ejemplo: k d k k.")
        print("q: salir conservando lo ya guardado. s: posponer esta identidad sin modificarla.")
        while True:
            response = input(f"Decisiones para {len(photos)} fotos: ").strip().lower()
            if response == "q":
                print("Revisión pausada. Las decisiones anteriores están guardadas en review.csv.")
                return review_status(path)
            if response == "s":
                break
            tokens = response.split()
            if len(tokens) != len(photos) or any(token not in choices for token in tokens):
                print(f"Se requieren {len(photos)} decisiones k/d separadas por espacios, o q/s.")
                continue
            updates = {row["path"]: {"decision": choices[token], "note": row["note"]}
                       for row, token in zip(photos, tokens)}
            digest = save_decisions(path, updates, expected_sha256=digest)
            for row in photos:
                row.update(updates[row["path"]])
            print("Guardado en review.csv.")
            break
    status = review_status(path)
    print(f"Revisión guardada: {status['total'] - status['pending']}/{status['total']}; pendientes: {status['pending']}.")
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
    previous = widgets.Button(description="Guardar y anterior", layout=widgets.Layout(width="180px"))
    save = widgets.Button(description="Guardar", button_style="info")
    following = widgets.Button(description="Guardar y siguiente", layout=widgets.Layout(width="180px"))
    controls = []

    def update_progress():
        pending = sum(row["decision"] not in {"keep", "drop"} for row in rows)
        keep = sum(row["decision"] == "keep" for row in groups[animals[state["index"]]])
        progress.value = (f"<p><b>{len(rows) - pending}/{len(rows)} fotos revisadas</b> · {pending} pendientes. "
                          f"Esta identidad: {keep} conservadas; mínimo 4 distintas tras eliminar duplicados.</p>")

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
                preview = widgets.HTML("<p><b>Archivo ilegible.</b> Marcar como descartar (drop).</p>")
            choice = widgets.Dropdown(options=[("Pendiente", ""), ("Conservar (keep)", "keep"),
                                               ("Descartar (drop)", "drop")],
                                      value=row["decision"] if row["decision"] in {"keep", "drop"} else "",
                                      layout=widgets.Layout(width="100%"))
            note = widgets.Text(value=row["note"], placeholder="Motivo / observación", layout=widgets.Layout(width="100%"))
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
            message.value = "<p>Decisiones guardadas en review.csv.</p>"
        except (OSError, ValueError) as exc:
            message.value = f"<p><b>No se guardó:</b> {html.escape(str(exc))}</p>"

    previous.on_click(lambda _: persist(-1))
    save.on_click(lambda _: persist())
    following.on_click(lambda _: persist(1))
    render()
    help_text = widgets.HTML("<p>Comprobar identidad consistente, objetivo inequívoco, cuerpo útil, calidad y variedad. "
                             "Las fotos con varios animales requieren un objetivo verificado. "
                             "Guardar antes de cerrar; al reabrir se retoma la primera identidad pendiente.</p>")
    return widgets.VBox([heading, progress, help_text, cards,
                         widgets.HBox([previous, save, following]), message])
