import { useState, type FormEvent, type ReactNode } from "react";
import { Link, NavLink, Navigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useMe } from "../hooks";
import { Avatar } from "../components/Avatar";
import { PictureDialog } from "../components/PictureDialog";
import { ErrorAlert } from "../components/ui";
import { Chip, Icon, Segmented } from "../components/kit";
import { roleLabel } from "../format";
import { getThemePref, setThemePref, type ThemePref } from "../theme";
import { toast } from "../toast";
import { TeamPage } from "./TeamPage";
import type { Me } from "../types";

/** The sections, in the order the side nav lists them. `lead` sections are hidden from everyone else. */
export const SECTIONS = [
  { key: "profile", label: "Profile" },
  { key: "appearance", label: "Appearance" },
  { key: "people", label: "People", lead: true },
  { key: "integrations", label: "Integrations" },
  { key: "application", label: "Application" },
] as const;
type SectionKey = (typeof SECTIONS)[number]["key"];

export function visibleSections(me: Pick<Me, "role">) {
  return SECTIONS.filter((s) => !("lead" in s) || me.role === "lead");
}

function Section({ title, intro, children }: { title: string; intro?: ReactNode; children: ReactNode }) {
  return (
    <section className="stack settings-section tools-gap-16" aria-labelledby="settings-title">
      <div className="tools-title">
        <h1 id="settings-title">{title}</h1>
        {intro && <p className="secondary tools-lead">{intro}</p>}
      </div>
      {children}
    </section>
  );
}

function ComingSoon({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="card stack settings-placeholder">
      <div className="row tools-between">
        <h2>{title}</h2>
        <Chip>Coming soon</Chip>
      </div>
      <div className="secondary">{children}</div>
    </div>
  );
}

function NameForm({ me }: { me: Me }) {
  const qc = useQueryClient();
  const [name, setName] = useState(me.name);
  const save = useMutation({
    mutationFn: () => api.updateMe({ name: name.trim() }),
    onSuccess: (u) => {
      qc.setQueryData(["me"], u);
      qc.invalidateQueries({ queryKey: ["users"] });
      qc.invalidateQueries({ queryKey: ["pickUsers"] });
      toast({ message: "Name saved" });
    },
  });
  const changed = name.trim() !== "" && name.trim() !== me.name;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (changed && !save.isPending) save.mutate();
  };
  if (!me.can_rename) {
    return (
      <div className="field">
        Name
        <span className="settings-value">{me.name}</span>
        <span className="muted small-text">
          {me.mode === "oidc"
            ? "Your name comes from the identity provider; change it there."
            : "Accounts come from RAINFOREST_USERS in .env; a lead changes them there."}
        </span>
      </div>
    );
  }
  return (
    <form className="stack" onSubmit={submit} aria-label="Your name">
      <div className="row tools-form-row">
        <label className="field tools-name-field">
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} autoComplete="name" />
        </label>
        <button type="submit" className="primary" disabled={!changed || save.isPending}>Save</button>
      </div>
      <span className="muted small-text">Shown on the sign-in page, the top bar and everything you change.</span>
      <ErrorAlert error={save.error} />
    </form>
  );
}

function Profile({ me }: { me: Me }) {
  const [picture, setPicture] = useState(false);
  return (
    <Section title="Profile" intro="How the rest of the team sees you.">
      <div className="card stack tools-gap-16">
        <div className="row tools-gap-16">
          <Avatar user={me} size={64} />
          <div className="stack tools-gap-4">
            <button type="button" onClick={() => setPicture(true)}>Change picture…</button>
            <span className="muted small-text">JPEG, PNG or WebP, up to 5 MB.</span>
          </div>
        </div>
        <NameForm key={me.name} me={me} />
        <div className="grid-2 tools-gap-12">
          <div className="field">Email<span className="settings-value">{me.email ?? "—"}</span></div>
          <div className="field">
            Role
            <span className="settings-value"><Chip tone={me.role === "lead" ? "accent" : "plain"} className={`role-${me.role}`}>{roleLabel(me.role)}</Chip></span>
            <span className="muted small-text">
              {me.mode === "oidc" ? "Set by your identity provider's groups." : <>A Team Lead changes roles on the <Link to="/people">People</Link> page.</>}
            </span>
          </div>
        </div>
      </div>
      {picture && <PictureDialog user={me} onClose={() => setPicture(false)} />}
    </Section>
  );
}

const THEMES: { value: ThemePref; label: string }[] = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

function Appearance() {
  const [pref, setPref] = useState<ThemePref>(getThemePref);
  const choose = (p: ThemePref) => {
    setThemePref(p);
    setPref(p);
  };
  return (
    <Section title="Appearance">
      <div className="card stack">
        <h2>Theme</h2>
        <Segmented label="Theme" value={pref} options={THEMES} onChange={choose} className="tools-start" />
        <span className="muted small-text">
          System follows your device's light or dark setting. The choice is kept in this browser.
        </span>
      </div>
    </Section>
  );
}

function Integrations() {
  const system = useQuery({ queryKey: ["system"], queryFn: api.system, staleTime: Infinity });
  return (
    <Section title="Integrations" intro="Connections to the services Rainforest works with.">
      <div className="card stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>MLflow</h2>
          {system.data && (
            <Chip tone={system.data.mlflow ? "good" : "plain"}>{system.data.mlflow ? "Logging on" : "Logging off"}</Chip>
          )}
        </div>
        <p className="secondary tools-lead">
          {system.data?.mlflow
            ? <>Model runs are logged to the built-in MLflow server. <a href="/mlflow/" target="_blank" rel="noopener noreferrer" className="tools-inline-icon">Open MLflow<Icon name="external" size={13} label="opens in a new tab" /></a></>
            : "Model runs aren't logged to MLflow. It is turned off with MLFLOW_TRACKING_URI in .env."}
        </p>
      </div>
      <ComingSoon title="An existing MLflow deployment">
        Log runs to your team's own MLflow tracking server instead of the built-in one.
      </ComingSoon>
      <ComingSoon title="Data storage">
        Pull datasets from S3 or other object storage, or from a database, instead of uploading files.
      </ComingSoon>
    </Section>
  );
}

function Application() {
  return (
    <Section title="Application" intro={<>Settings for the whole app. Today they live in <code>.env</code>; see docs/configuration.md.</>}>
      <ComingSoon title="App settings">
        Change defaults such as export splits, cross-validation folds and background job limits from here, without
        editing <code>.env</code> and restarting.
      </ComingSoon>
    </Section>
  );
}

export function SettingsPage() {
  const me = useMe();
  const { section } = useParams<{ section?: string }>();
  // The section was called Team before People; old links still land.
  if (section === "team") return <Navigate to="/settings/people" replace />;
  if (!me.data) return null;
  const sections = visibleSections(me.data);
  const current = sections.find((s) => s.key === section);
  if (!current) return <Navigate to={`/settings/${sections[0].key}`} replace />;
  const body: Record<SectionKey, ReactNode> = {
    profile: <Profile me={me.data} />,
    appearance: <Appearance />,
    people: (
      <Section title="People" intro="Users and roles.">
        <TeamPage embedded />
      </Section>
    ),
    integrations: <Integrations />,
    application: <Application />,
  };
  return (
    <div className="page settings">
      <nav className="settings-nav" aria-label="Settings">
        <h2 className="settings-nav-title">Settings</h2>
        {sections.map((s) => (
          <NavLink key={s.key} to={`/settings/${s.key}`} className="settings-nav-link">{s.label}</NavLink>
        ))}
      </nav>
      <div className="settings-body">{body[current.key]}</div>
    </div>
  );
}
