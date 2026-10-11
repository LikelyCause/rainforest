import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { Avatar } from "./Avatar";
import { ErrorAlert } from "./ui";

const TYPES = ["image/jpeg", "image/png", "image/webp"];

/** Choose, preview and save an account's picture, or remove it. */
export function PictureDialog({ user, onClose }: {
  user: { id: number; name: string; avatar_url?: string | null }; onClose: () => void;
}) {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  const done = () => {
    for (const k of ["me", "users", "pickUsers"]) qc.invalidateQueries({ queryKey: [k] });
    onClose();
  };
  const save = useMutation({ mutationFn: () => api.setAvatar(user.id, file!), onSuccess: done });
  const remove = useMutation({ mutationFn: () => api.removeAvatar(user.id), onSuccess: done });
  const choose = (f: File | undefined) => {
    if (!f) return;
    if (!TYPES.includes(f.type)) return setProblem("Use a JPEG, PNG or WebP picture.");
    if (f.size > 5 * 1024 * 1024) return setProblem("Pictures can be 5 MB at most.");
    setProblem(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", esc);
    return () => document.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <div className="overlay" onClick={onClose}>
      <div className="dialog stack picture-dialog" role="dialog" aria-modal="true" aria-label="Change picture" onClick={(e) => e.stopPropagation()}>
        <h2>{user.name}'s picture</h2>
        <button type="button" className="dropzone picture-drop" autoFocus onClick={() => input.current?.click()}
          onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); choose(e.dataTransfer.files[0]); }}>
          {preview ? <img className="avatar" src={preview} alt="New picture" /> : <Avatar user={user} size={112} />}
          <span className="muted small-text">Drop a picture here or click to choose one · JPEG, PNG or WebP, up to 5 MB.
            It's cropped to a centred square.</span>
        </button>
        <input ref={input} type="file" accept={TYPES.join(",")} hidden onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }} />
        {problem && <span className="small-text" role="alert" style={{ color: "var(--bad)" }}>{problem}</span>}
        <ErrorAlert error={save.error ?? remove.error} />
        <div className="row" style={{ justifyContent: "space-between" }}>
          {user.avatar_url ? (
            <button type="button" className="ghost" disabled={remove.isPending} onClick={() => remove.mutate()}>Remove picture</button>
          ) : <span />}
          <div className="row">
            <button type="button" className="ghost" onClick={onClose}>Cancel</button>
            <button type="button" className="primary" disabled={!file || save.isPending} onClick={() => save.mutate()}>
              {save.isPending ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
