import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Logo } from "../components/Logo";
import { ErrorAlert } from "../components/ui";

/** Where the OIDC callback sends someone the provider signed in but Rainforest won't let in. */
export function ForbiddenPage() {
  const [params] = useSearchParams();
  const reason = params.get("reason");
  const cfg = useQuery({ queryKey: ["authConfig"], queryFn: api.authConfig, retry: false, staleTime: Infinity });
  const provider = cfg.data?.provider_name ?? "your identity provider";
  // Signing out of the provider lets someone try a different account; otherwise it signs them straight back in.
  const out = useMutation({
    mutationFn: api.logout,
    onSuccess: ({ logout_url }) => window.location.assign(logout_url ?? "/login"),
  });
  return (
    <div className="login tools-login">
      <div className="card login-card stack">
        <div className="login-brand"><Logo /> Rainforest</div>
        <h1>No access</h1>
        {reason === "deactivated" ? (
          <p className="secondary">A lead has deactivated your Rainforest account. Ask a lead to reactivate it.</p>
        ) : (
          <p className="secondary">
            You signed in with {provider}, but your account isn't in
            {cfg.data ? <> the <code>{cfg.data.lead_group}</code> or <code>{cfg.data.member_group}</code> group</> : " a Rainforest group"},
            so it can't use Rainforest. Ask whoever manages {provider} to add you.
          </p>
        )}
        <ErrorAlert error={out.error} />
        <div className="row tools-between">
          <button type="button" className="primary" disabled={out.isPending} onClick={() => out.mutate()}>
            Sign out of {provider}
          </button>
          <Link to="/login">Back to sign-in</Link>
        </div>
      </div>
    </div>
  );
}
