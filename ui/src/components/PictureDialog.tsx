import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { MAX_ZOOM, centred, clampFrame, coverScale, cropSquare, zoomAt, type Frame, type Size } from "../crop";
import { Avatar } from "./Avatar";
import { ErrorAlert } from "./ui";
import { Banner, Dialog, Icon } from "./kit";

const TYPES = ["image/jpeg", "image/png", "image/webp"];
const VIEW = 240;

/** Drag, pinch, scroll or use the keyboard to frame `src` in a circle; reports the frame as it changes. */
function Framer({ src, img, frame, onFrame }: { src: string; img: Size; frame: Frame; onFrame: (f: Frame) => void }) {
  const box = useRef<HTMLDivElement>(null);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const latest = useRef(frame);
  latest.current = frame;
  const set = (f: Frame) => onFrame(clampFrame(f, img, VIEW));
  const zoom = (z: number, px?: number, py?: number) => onFrame(zoomAt(latest.current, img, VIEW, z, px, py));
  const local = (e: { clientX: number; clientY: number }) => {
    const r = box.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  // Wheel zoom needs a non-passive listener to keep the page from scrolling.
  useEffect(() => {
    const el = box.current!;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const p = local(e);
      onFrame(zoomAt(latest.current, img, VIEW, latest.current.zoom * Math.exp(-e.deltaY / 500), p.x, p.y));
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, [img, onFrame]);

  const down = (e: PointerEvent) => {
    box.current!.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, local(e));
  };
  const move = (e: PointerEvent) => {
    const ps = pointers.current;
    const before = ps.get(e.pointerId);
    if (!before) return;
    const now = local(e);
    const others = [...ps.entries()].filter(([id]) => id !== e.pointerId).map(([, p]) => p);
    ps.set(e.pointerId, now);
    const f = latest.current;
    if (others.length === 0) {
      set({ ...f, x: f.x + now.x - before.x, y: f.y + now.y - before.y });
    } else {
      // Pinch: zoom by how far the two fingers spread, around the point between them, and follow that point.
      const o = others[0];
      const spread = Math.hypot(now.x - o.x, now.y - o.y) / Math.max(Math.hypot(before.x - o.x, before.y - o.y), 1);
      const mid = { x: (now.x + o.x) / 2, y: (now.y + o.y) / 2 };
      const was = { x: (before.x + o.x) / 2, y: (before.y + o.y) / 2 };
      const moved = clampFrame({ ...f, x: f.x + mid.x - was.x, y: f.y + mid.y - was.y }, img, VIEW);
      onFrame(zoomAt(moved, img, VIEW, f.zoom * spread, mid.x, mid.y));
    }
  };
  const up = (e: PointerEvent) => pointers.current.delete(e.pointerId);
  const key = (e: ReactKeyboardEvent) => {
    const step = e.shiftKey ? 40 : 10;
    const f = latest.current;
    const moves: Record<string, [number, number]> = { ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
    if (moves[e.key]) set({ ...f, x: f.x + moves[e.key][0], y: f.y + moves[e.key][1] });
    else if (e.key === "+" || e.key === "=") zoom(f.zoom * 1.1);
    else if (e.key === "-" || e.key === "_") zoom(f.zoom / 1.1);
    else return;
    e.preventDefault();
  };

  const s = coverScale(img, VIEW) * frame.zoom;
  return (
    <div className="stack tools-framer">
      <div ref={box} className="framer" style={{ width: VIEW, height: VIEW }} tabIndex={0} role="group"
        aria-label="Picture framing: drag or use the arrow keys to move it, scroll, pinch or + and − to zoom"
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key}>
        <img src={src} alt="" draggable={false}
          style={{ width: img.w * s, height: img.h * s, transform: `translate(${frame.x}px, ${frame.y}px)` }} />
        <div className="framer-ring" aria-hidden="true" />
      </div>
      <label className="row small-text muted framer-zoom">
        <span aria-hidden="true">−</span>
        <input type="range" min={1} max={MAX_ZOOM} step={0.01} value={frame.zoom} aria-label="Zoom"
          onChange={(e) => zoom(Number(e.target.value))} />
        <span aria-hidden="true">+</span>
      </label>
    </div>
  );
}

/** Choose a picture, frame it, and save it; or remove the current one. */
export function PictureDialog({ user, onClose }: {
  user: { id: number; name: string; avatar_url?: string | null }; onClose: () => void;
}) {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [img, setImg] = useState<Size | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  // The browser turns phone pictures upright, as the server does, so the frame is in the same pixels.
  useEffect(() => {
    if (!preview) return;
    const probe = new Image();
    probe.onload = () => {
      const size = { w: probe.naturalWidth, h: probe.naturalHeight };
      setImg(size);
      setFrame(centred(size, VIEW));
    };
    probe.onerror = () => setProblem("That file isn't a picture this browser can show; use a JPEG, PNG or WebP.");
    probe.src = preview;
  }, [preview]);
  const done = () => {
    for (const k of ["me", "users", "pickUsers"]) qc.invalidateQueries({ queryKey: [k] });
    onClose();
  };
  const save = useMutation({
    mutationFn: () => api.setAvatar(user.id, file!, img && frame ? cropSquare(frame, img, VIEW) : undefined),
    onSuccess: done,
  });
  const remove = useMutation({ mutationFn: () => api.removeAvatar(user.id), onSuccess: done });
  const choose = (f: File | undefined) => {
    if (!f) return;
    if (!TYPES.includes(f.type)) return setProblem("Use a JPEG, PNG or WebP picture.");
    if (f.size > 5 * 1024 * 1024) return setProblem("Pictures can be 5 MB at most.");
    setProblem(null);
    setFile(f);
    setImg(null);
    setFrame(null);
    setPreview(URL.createObjectURL(f));
  };
  const framing = preview && img && frame;
  return (
    <Dialog title={`${user.name}'s picture`} onClose={onClose} labelledBy="picture-title"
      actions={<>
        {user.avatar_url && (
          <button type="button" className="ghost tools-danger-text" disabled={remove.isPending} onClick={() => remove.mutate()}>Remove picture</button>
        )}
        <span className="spacer" />
        <button type="button" className="ghost" onClick={onClose}>Cancel</button>
        <button type="button" className="primary" disabled={!framing || save.isPending} onClick={() => save.mutate()}>
          {save.isPending ? "Saving…" : "Save"}
        </button>
      </>}>
      <div className="stack picture-dialog" onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); choose(e.dataTransfer.files[0]); }}>
        {framing ? (
          <>
            <Framer src={preview} img={img} frame={frame} onFrame={setFrame} />
            <div className="row tools-center">
              <span className="muted small-text">Drag to move · scroll or pinch to zoom</span>
              <button type="button" className="ghost small" onClick={() => setFrame(centred(img, VIEW))}><Icon name="reset" size={14} />Reset</button>
              <button type="button" className="ghost small" onClick={() => input.current?.click()}><Icon name="upload" size={14} />Choose another…</button>
            </div>
          </>
        ) : (
          <button type="button" className="dropzone picture-drop" data-autofocus onClick={() => input.current?.click()}>
            <Avatar user={user} size={112} />
            <span className="muted small-text">Drop a picture here or click to choose one · JPEG, PNG or WebP, up to 5 MB.
              You can move and zoom it next.</span>
          </button>
        )}
        <input ref={input} type="file" accept={TYPES.join(",")} hidden onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }} />
        {problem && <Banner tone="bad">{problem}</Banner>}
        <ErrorAlert error={save.error ?? remove.error} />
      </div>
    </Dialog>
  );
}
