import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useMe } from "../hooks";
import { Avatar } from "../components/Avatar";
import { PictureDialog } from "../components/PictureDialog";
import { ErrorAlert } from "../components/ui";
import { fmtAgo, roleLabel } from "../format";
import { showError, toast } from "../toast";
import type { Role, TeamUser } from "../types";

function AddUser() {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<Role>("member");
  const add = useMutation({
    mutationFn: () => api.createUser({ name: name.trim(), email: email.trim() || null, role }),
    onSuccess: (u) => {
      qc.invalidateQueries({ queryKey: ["users"] });
      qc.invalidateQueries({ queryKey: ["pickUsers"] });
      toast({ message: `Added ${u.name}` });
      setName("");
      setEmail("");
      setRole("member");
    },
  });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (name.trim() && !add.isPending) add.mutate();
  };
  return (
    <form className="card stack" onSubmit={submit} aria-label="Add a user">
      <h2>Add a user</h2>
      <div className="row" style={{ alignItems: "flex-end" }}>
        <label className="field">Name<input value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} /></label>
        <label className="field">Email (optional)<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} maxLength={320} /></label>
        <label className="field">Role
          <select value={role} onChange={(e) => setRole(e.target.value as Role)}>
            <option value="member">{roleLabel("member")}</option>
            <option value="lead">{roleLabel("lead")}</option>
            <option value="viewer">{roleLabel("viewer")}</option>
          </select>
        </label>
        <button type="submit" className="primary" disabled={!name.trim() || add.isPending}>Add</button>
      </div>
      <ErrorAlert error={add.error} />
    </form>
  );
}

/** An inline text cell that saves on Enter or blur, and reverts on Escape. */
function EditableText({ value, label, onSave, disabled }: {
  value: string; label: string; onSave: (v: string) => void; disabled?: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft !== null && draft.trim() !== value) onSave(draft.trim());
    setDraft(null);
  };
  return (
    <input className="inline-edit" aria-label={label} value={draft ?? value} disabled={disabled}
      onChange={(e) => setDraft(e.target.value)} onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
        if (e.key === "Escape") { setDraft(null); e.currentTarget.blur(); }
      }} />
  );
}

function UserRow({ u, meId, canEdit, sso, lead }: { u: TeamUser; meId: number; canEdit: boolean; sso: boolean; lead: boolean }) {
  const qc = useQueryClient();
  const [picture, setPicture] = useState(false);
  const patch = useMutation({
    mutationFn: (data: Parameters<typeof api.patchUser>[1]) => api.patchUser(u.id, data),
    onSuccess: (saved, data) => {
      qc.invalidateQueries({ queryKey: ["users"] });
      qc.invalidateQueries({ queryKey: ["pickUsers"] });
      if (saved.id === meId) qc.invalidateQueries({ queryKey: ["me"] });
      if (data.active === false) toast({ message: `Deactivated ${saved.name}`, detail: "They are signed out everywhere." });
      else if (data.role) toast({ message: `${saved.name} is now a ${roleLabel(saved.role).toLowerCase()}`, detail: "They need to sign in again." });
    },
    onError: (err) => showError(`Couldn't update ${u.name}`, err),
  });
  // In OIDC mode the provider owns names, emails and roles; leads can only switch users off and on.
  const editDetails = canEdit && !sso && u.active;
  return (
    <tr className={u.active ? undefined : "inactive"}>
      <td>
        <div className="name-cell">
          {(lead || u.id === meId) && u.active ? (
            <button type="button" className="ghost avatar-button" title={`Change ${u.id === meId ? "your" : `${u.name}'s`} picture`}
              onClick={() => setPicture(true)}><Avatar user={u} size={28} /></button>
          ) : <Avatar user={u} size={28} />}
          {picture && <PictureDialog user={u} onClose={() => setPicture(false)} />}
          {editDetails ? <EditableText value={u.name} label={`Name of ${u.name}`} disabled={patch.isPending}
            onSave={(name) => name && patch.mutate({ name })} /> : u.name}
          {u.id === meId && <span className="muted">(you)</span>}
        </div>
      </td>
      <td>
        {editDetails ? <EditableText value={u.email ?? ""} label={`Email of ${u.name}`} disabled={patch.isPending}
          onSave={(email) => patch.mutate({ email: email || null })} /> : u.email ?? <span className="muted">—</span>}
      </td>
      <td>
        {editDetails ? (
          <select aria-label={`Role of ${u.name}`} value={u.role} disabled={patch.isPending}
            onChange={(e) => patch.mutate({ role: e.target.value as Role })}>
            <option value="member">{roleLabel("member")}</option>
            <option value="lead">{roleLabel("lead")}</option>
            <option value="viewer">{roleLabel("viewer")}</option>
          </select>
        ) : <span className={`badge${u.role === "lead" ? " accent" : ""}`}>{roleLabel(u.role)}</span>}
      </td>
      <td>{u.active ? <span className="badge good">Active</span> : <span className="badge">Deactivated</span>}</td>
      <td className="muted" title={u.last_seen_at ?? undefined}>{u.last_seen_at ? fmtAgo(u.last_seen_at) : "never"}</td>
      <td style={{ textAlign: "right" }}>
        {canEdit && (u.active ? (
          <button type="button" className="ghost small" disabled={patch.isPending || u.id === meId}
            title={u.id === meId ? "You can't deactivate yourself" : "Signs them out and hides them from sign-in"}
            onClick={() => patch.mutate({ active: false })}>Deactivate</button>
        ) : (
          <button type="button" className="small" disabled={patch.isPending} onClick={() => patch.mutate({ active: true })}>Reactivate</button>
        ))}
      </td>
    </tr>
  );
}

export function TeamPage() {
  const me = useMe();
  const users = useQuery({ queryKey: ["users"], queryFn: api.users });
  const cfg = useQuery({ queryKey: ["authConfig"], queryFn: api.authConfig, staleTime: Infinity });
  if (!me.data) return null;
  const lead = me.data.role === "lead";
  const sso = me.data.mode === "oidc";
  const fromEnv = Boolean(cfg.data?.accounts_from_env);
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Team</h1>
      </div>
      <p className="secondary" style={{ margin: 0 }}>
        {sso
          ? "People appear here after they first sign in. Names, emails and roles come from the identity provider (its lead and member groups); here a lead can deactivate or reactivate someone."
          : fromEnv
            ? "Everyone listed can be picked on the sign-in page. The accounts and their roles come from RAINFOREST_USERS in .env; change them there and restart. Team Leads start baselines, finalize and delete datasets; Team Members do everything else; Viewers look and change nothing."
            : "Everyone listed can be picked on the sign-in page. Team Leads delete datasets and manage this list; Team Members do everything else; Viewers look and change nothing."}
        {!lead && !fromEnv && " Only a Team Lead can change it."}
        {" "}Click your picture to change it{lead ? "; a Team Lead can change anyone's" : ""}.
      </p>
      {lead && !sso && !fromEnv && <AddUser />}
      <ErrorAlert error={users.error} />
      {users.data && (
        <div className="table-wrap" style={{ maxHeight: "none" }}>
          <table className="data team">
            <thead>
              <tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Last seen</th><th aria-label="Actions" /></tr>
            </thead>
            <tbody>
              {users.data.map((u) => <UserRow key={u.id} u={u} meId={me.data.id} canEdit={lead && !fromEnv} sso={sso} lead={lead} />)}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
