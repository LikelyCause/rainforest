import { useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, isServerPath, oidcLoginUrl, safeNext } from "../api";
import { Avatar } from "../components/Avatar";
import { Logo } from "../components/Logo";
import { ErrorAlert } from "../components/ui";
import { Chip } from "../components/kit";
import { roleLabel } from "../format";
import type { AuthConfig } from "../types";

/** Why the API sent the browser back here from the OIDC callback. */
const CALLBACK_ERRORS: Record<string, string> = {
  state: "That sign-in link had expired or came from another browser window. Please sign in again.",
  provider: "The identity provider didn't complete the sign-in.",
  token:
    "The identity provider's answer couldn't be verified. Try again, and ask an admin to check the API log if it keeps happening.",
};

function PickUser({ next }: { next: string }) {
  const nav = useNavigate();
  const qc = useQueryClient();
  const users = useQuery({ queryKey: ["pickUsers"], queryFn: api.pickUsers });
  const pick = useMutation({
    mutationFn: api.pick,
    onSuccess: () => {
      // Drop everything fetched (and failed) while signed out.
      qc.clear();
      // /mlflow/ is served by nginx, not the app, so it needs a full page load.
      if (isServerPath(next)) window.location.assign(next);
      else nav(next, { replace: true });
    },
  });
  return (
    <>
      <ErrorAlert error={users.error} />
      {users.isPending && <span className="skeleton tools-login-skeleton" />}
      <ul className="profile-grid" aria-label="Profiles">
        {users.data?.map((u) => (
          <li key={u.id}>
            <button type="button" className="profile" disabled={pick.isPending} onClick={() => pick.mutate(u.id)}>
              <Avatar user={u} />
              <span className="profile-name">{u.name}</span>
              <Chip tone={u.role === "lead" ? "accent" : "plain"} className={`role-${u.role}`}>
                {roleLabel(u.role)}
              </Chip>
            </button>
          </li>
        ))}
      </ul>
      <ErrorAlert error={pick.error} />
      <p className="muted small-text profile-note">Anyone who can open this page can pick any profile.</p>
    </>
  );
}

function SignInBody({ cfg, next }: { cfg: AuthConfig; next: string }) {
  if (cfg.problem) return <ErrorAlert error={{ message: cfg.problem }} title="Not set up yet." />;
  if (cfg.mode === "pick") return <PickUser next={next} />;
  return (
    <a className="btn primary sign-in" href={oidcLoginUrl(next)}>
      Sign in with {cfg.provider_name}
    </a>
  );
}

export function LoginPage() {
  const [params] = useSearchParams();
  const next = safeNext(params.get("next"));
  const failure = CALLBACK_ERRORS[params.get("error") ?? ""];
  const cfg = useQuery({ queryKey: ["authConfig"], queryFn: api.authConfig, retry: false, staleTime: Infinity });
  const picker = cfg.data?.mode === "pick" && !cfg.data.problem;
  return (
    <div className="login tools-login">
      <div className={`stack ${picker ? "picker" : "card login-card"}`}>
        <div className="login-brand">
          <Logo /> Rainforest
        </div>
        <div className="tools-title">
          <h1>{cfg.data?.mode === "pick" ? "Who are you?" : "Sign in"}</h1>
          {picker && <span className="secondary">Pick your profile to sign in.</span>}
        </div>
        {failure && <ErrorAlert error={{ message: failure }} />}
        {cfg.error && <ErrorAlert error={cfg.error} title="Can't reach the API." />}
        {cfg.data && <SignInBody cfg={cfg.data} next={next} />}
      </div>
    </div>
  );
}
