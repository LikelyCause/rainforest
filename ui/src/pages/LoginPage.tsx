import { useState, type FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, safeNext } from "../api";
import { Logo } from "../components/Logo";
import { ErrorAlert } from "../components/ui";

export function LoginPage() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [params] = useSearchParams();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const m = useMutation({
    mutationFn: () => api.login(username, password),
    onSuccess: () => {
      // Drop everything fetched (and failed) while signed out.
      qc.clear();
      nav(safeNext(params.get("next")), { replace: true });
    },
  });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!m.isPending) m.mutate();
  };
  return (
    <div className="login">
      <form className="card login-card stack" onSubmit={submit}>
        <div className="login-brand"><Logo /> Rainforest</div>
        <h1>Sign in</h1>
        <label className="field">
          Username
          <input name="username" autoComplete="username" value={username} required
            onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label className="field">
          Password
          <input name="password" type="password" autoComplete="current-password" value={password} required autoFocus
            onChange={(e) => setPassword(e.target.value)} />
        </label>
        {m.error && <ErrorAlert error={m.error} />}
        <button type="submit" className="primary" disabled={m.isPending}>{m.isPending ? "Signing in…" : "Sign in"}</button>
      </form>
    </div>
  );
}
