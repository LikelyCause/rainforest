import { useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, oidcLoginUrl, safeNext } from "../api";
import { Logo } from "../components/Logo";
import { ErrorAlert } from "../components/ui";
import { roleLabel } from "../format";
import type { AuthConfig } from "../types";

/** Why the API sent the browser back here from the OIDC callback. */
const CALLBACK_ERRORS: Record<string, string> = {
  state: "That sign-in link had expired or came from another browser window. Please sign in again.",
  provider: "The identity provider didn't complete the sign-in.",
  token: "The identity provider's answer couldn't be verified. Try again, and ask an admin to check the API log if it keeps happening.",
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
      nav(next, { replace: true });
    },
  });
  return (
    <>
      <div className="alert warn" role="note">
        <span aria-hidden="true">!</span>
        <div className="grow">
          <b>This mode doesn't authenticate.</b> Anyone who can open this page can sign in as anyone. Team deployments
          should use OIDC sign-in.
        </div>
      </div>
      <ErrorAlert error={users.error} />
      {users.isPending && <span className="skeleton" />}
      <div className="stack pick-list" role="list" aria-label="Users">
        {users.data?.map((u) => (
          <button key={u.id} type="button" role="listitem" className="pick-user" disabled={pick.isPending}
            onClick={() => pick.mutate(u.id)}>
            <span className="grow">{u.name}</span>
            <span className={`badge${u.role === "lead" ? " accent" : ""}`}>{roleLabel(u.role)}</span>
          </button>
        ))}
      </div>
      <ErrorAlert error={pick.error} />
    </>
  );
}

function SignInBody({ cfg, next }: { cfg: AuthConfig; next: string }) {
  if (cfg.problem) return <ErrorAlert error={{ message: cfg.problem }} title="Not set up yet." />;
  if (cfg.mode === "pick") return <PickUser next={next} />;
  return (
    <a className="btn primary sign-in" href={oidcLoginUrl(next)}>Sign in with {cfg.provider_name}</a>
  );
}

export function LoginPage() {
  const [params] = useSearchParams();
  const next = safeNext(params.get("next"));
  const failure = CALLBACK_ERRORS[params.get("error") ?? ""];
  const cfg = useQuery({ queryKey: ["authConfig"], queryFn: api.authConfig, retry: false, staleTime: Infinity });
  return (
    <div className="login">
      <div className="card login-card stack">
        <div className="login-brand"><Logo /> Rainforest</div>
        <h1>{cfg.data?.mode === "pick" ? "Who are you?" : "Sign in"}</h1>
        {failure && <ErrorAlert error={{ message: failure }} />}
        {cfg.error && <ErrorAlert error={cfg.error} title="Can't reach the API." />}
        {cfg.data && <SignInBody cfg={cfg.data} next={next} />}
      </div>
    </div>
  );
}
