"""Profile pictures: any JPEG, PNG or WebP, cropped to the square its owner framed and re-encoded as a 256 × 256 WebP,
stored as `<user id>-<hash>.webp` under DATA_DIR/avatars. Re-encoding leaves nothing of the upload but its pixels
(no EXIF, no location, no hidden payload)."""
from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import get_settings
from .db import connection

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 40_000_000
SIZE = 256
FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}  # MPO: a phone JPEG with a second image (an HDR gain map); its first is kept
NAME_RE = re.compile(r"[0-9a-f]{16}\.webp")


Crop = tuple[float, float, float]  # x, y and side of a square, in the upright picture's pixels


def crop_box(width: int, height: int, crop: Crop | None) -> tuple[int, int, int, int] | None:
    """The square to keep, clamped into the picture (the browser's sums can run a pixel over), or None to keep the
    centred square. A square that isn't one, or lies mostly outside the picture, is refused."""
    if crop is None:
        return None
    x, y, side = crop
    slack = 2
    if (not side >= 1 or x < -slack or y < -slack
            or x + side > width + slack or y + side > height + slack):
        raise HTTPException(422, "That framing doesn't fit the picture; choose it again.")
    side = min(round(side), width, height)
    left = min(max(round(x), 0), width - side)
    top = min(max(round(y), 0), height - side)
    return left, top, left + side, top + side


def _encode(data: bytes, crop: Crop | None = None) -> bytes:
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in FORMATS:
                raise HTTPException(400, "Use a JPEG, PNG or WebP picture.")
            if img.width * img.height > MAX_PIXELS:
                raise HTTPException(400, "That picture is too large; use one under 40 megapixels.")
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") and "transparency" in img.info or
                              img.mode in ("RGBA", "LA") else "RGB")
            box = crop_box(img.width, img.height, crop)
            square = (img.resize((SIZE, SIZE), Image.Resampling.LANCZOS, box=box) if box
                      else ImageOps.fit(img, (SIZE, SIZE), Image.Resampling.LANCZOS))
            out = io.BytesIO()
            square.save(out, "WEBP", quality=85)
            return out.getvalue()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        if isinstance(exc, Image.DecompressionBombError):
            raise HTTPException(400, "That picture is too large; use one under 40 megapixels.") from None
        raise HTTPException(400, "That file isn't a picture we can read; use a JPEG, PNG or WebP.") from None


def path_for(user_id: int, name: str) -> Path | None:
    """The stored file a picture URL names, or None: the name must be a hash we wrote, inside the avatars folder."""
    if not NAME_RE.fullmatch(name):
        return None
    folder = get_settings().avatars_dir.resolve()
    path = (folder / f"{user_id}-{name}").resolve()
    return path if path.parent == folder and path.is_file() else None


def store(user_id: int, data: bytes, crop: Crop | None = None) -> str:
    """Save a new picture for `user_id` and remove the old one; returns its hash. `crop` is the square the person framed
    (else the centred one)."""
    webp = _encode(data, crop)
    digest = hashlib.sha256(webp).hexdigest()[:16]
    folder = get_settings().avatars_dir
    folder.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".part")
    with os.fdopen(fd, "wb") as f:
        f.write(webp)
    os.replace(tmp, folder / f"{user_id}-{digest}.webp")
    with connection() as conn:
        old = conn.execute("SELECT avatar FROM meta.users WHERE id = %s FOR UPDATE", (user_id,)).fetchone()
        conn.execute("UPDATE meta.users SET avatar = %s WHERE id = %s", (digest, user_id))
    if old and old["avatar"] and old["avatar"] != digest:
        (folder / f"{user_id}-{old['avatar']}.webp").unlink(missing_ok=True)
    return digest


def remove(user_id: int) -> None:
    with connection() as conn:
        old = conn.execute("UPDATE meta.users u SET avatar = NULL FROM meta.users o WHERE u.id = o.id AND u.id = %s "
                           "RETURNING o.avatar", (user_id,)).fetchone()
    if old and old["avatar"]:
        (get_settings().avatars_dir / f"{user_id}-{old['avatar']}.webp").unlink(missing_ok=True)
