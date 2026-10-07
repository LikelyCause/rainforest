import { NavLink, Navigate, Route, Routes, useMatch } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useOptions } from "./hooks";
import { api } from "./api";
import { DatasetsPage } from "./pages/DatasetsPage";
import { ReviewPage } from "./pages/ReviewPage";
import { QueryPage } from "./pages/QueryPage";
import { ExportPage } from "./pages/ExportPage";

function Logo() {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 3 6 15h5l-6 8h22l-6-8h5z" fill="var(--brand)" />
      <rect x="14.5" y="22" width="3" height="7" rx="1" fill="var(--brand-trunk)" />
    </svg>
  );
}

function FamilySwitch({ datasetId, family }: { datasetId: number; family: string }) {
  const qc = useQueryClient();
  const { data: options } = useOptions();
  const m = useMutation({
    mutationFn: (f: string) => api.patchDataset(datasetId, { model_family: f }),
    onSuccess: (ds) => {
      qc.setQueryData(["dataset", datasetId], ds);
      for (const k of ["summary", "fieldQuery", "exportPreview"]) qc.invalidateQueries({ queryKey: [k, datasetId] });
    },
  });
  if (!options) return null;
  return (
    <label className="row" style={{ gap: 6, flexWrap: "nowrap", fontSize: 12.5 }} title="Which kind of model the preprocessing settings are for">
      <span className="muted">Preparing for</span>
      <select value={family} disabled={m.isPending} onChange={(e) => m.mutate(e.target.value)} aria-label="Model family"
        style={{ minHeight: 28, padding: "2px 6px" }}>
        {options.families.map((f) => (
          <option key={f.value} value={f.value}>{f.label}{f.value === options.default_family ? " (default)" : ""}</option>
        ))}
      </select>
    </label>
  );
}

function TopBar() {
  const m = useMatch("/d/:datasetId/*");
  const datasetId = m ? Number(m.params.datasetId) : null;
  const { data: ds } = useQuery({
    queryKey: ["dataset", datasetId],
    queryFn: () => api.dataset(datasetId!),
    enabled: datasetId !== null,
  });
  const { data: summary } = useQuery({
    queryKey: ["summary", datasetId],
    queryFn: () => api.summary(datasetId!),
    enabled: datasetId !== null && ds?.status === "ready",
  });
  const tab = (to: string, label: string) => (
    <NavLink to={to} className={({ isActive }) => `tab${isActive ? " active" : ""}`}>{label}</NavLink>
  );
  return (
    <header className="topbar">
      <NavLink to="/" className="brand"><Logo /> Rainforest</NavLink>
      {datasetId !== null && ds && (
        <>
          <span className="muted">/</span>
          <span style={{ fontWeight: 600, maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ds.name}</span>
          <nav className="tabs" aria-label="Dataset sections">
            {tab(`/d/${datasetId}/review`, "Review")}
            {tab(`/d/${datasetId}/query`, "Query")}
            {tab(`/d/${datasetId}/export`, "Export")}
          </nav>
        </>
      )}
      <div className="spacer" />
      {datasetId !== null && ds?.status === "ready" && <FamilySwitch datasetId={datasetId} family={ds.model_family} />}
      {summary && summary.total_fields > 0 && (
        <div className="row" style={{ gap: 10, flexWrap: "nowrap" }} title="Fields reviewed">
          <span className="secondary num" style={{ fontSize: 12.5 }}>
            {summary.reviewed} / {summary.total_fields} reviewed
          </span>
          <div className="meter" style={{ width: 120 }}>
            <span style={{ width: `${(100 * summary.reviewed) / summary.total_fields}%` }} />
          </div>
        </div>
      )}
    </header>
  );
}

export default function App() {
  return (
    <div className="app">
      <TopBar />
      <main className="main">
        <Routes>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/d/:datasetId" element={<Navigate to="review" replace />} />
          <Route path="/d/:datasetId/review/:fieldId?" element={<ReviewPage />} />
          <Route path="/d/:datasetId/query" element={<QueryPage />} />
          <Route path="/d/:datasetId/export" element={<ExportPage />} />
          <Route path="*" element={<div className="empty">Page not found. <NavLink to="/">Go home</NavLink></div>} />
        </Routes>
      </main>
    </div>
  );
}
