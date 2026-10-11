import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { describeDerived, parseGroups } from "../derived";
import { useFields } from "../hooks";
import { showError, toast } from "../toast";
import type { Dataset, DerivedInput } from "../types";
import { ErrorAlert } from "./ui";

type Kind = DerivedInput["kind"];
const KIND_LABEL: Record<Kind, string> = { formula: "Formula", groups: "Group values", count: "Count matches" };
const KIND_HELP: Record<Kind, string> = {
  formula: "Arithmetic over number fields: + - * / ** and log, log1p, exp, sqrt, abs, min, max, clip. Put a name with spaces or dashes in `backticks`.",
  groups: "One group per line as Label: values. 390-459 is a range; anything else is an exact value. Rows matching no group get the fallback label.",
  count: "How many of the chosen fields hold one of the values, per row.",
};

/** Columns computed from the dataset's own columns. They become ordinary fields: reviewed, exported and trained on. */
export function DerivedFields({ ds }: { ds: Dataset }) {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["derived", ds.id], queryFn: () => api.derived(ds.id) });
  const { data: fields } = useFields(ds.id);
  const [kind, setKind] = useState<Kind>("formula");
  const [name, setName] = useState("");
  const [expr, setExpr] = useState("");
  const [source, setSource] = useState("");
  const [groupText, setGroupText] = useState("");
  const [other, setOther] = useState("Other");
  const [countFields, setCountFields] = useState<string[]>([]);
  const [countFilter, setCountFilter] = useState("");
  const [countValues, setCountValues] = useState("");

  const parsed = useMemo(() => parseGroups(groupText), [groupText]);
  const input: DerivedInput | null =
    kind === "formula" ? (expr.trim() ? { kind, spec: { expr } } : null)
    : kind === "groups" ? (source && !parsed.error ? { kind, spec: { source, other, groups: parsed.groups } } : null)
    : countFields.length && countValues.trim() ? { kind, spec: { fields: countFields, values: countValues.split(",").map((v) => v.trim()).filter(Boolean) } }
    : null;

  const preview = useMutation({ mutationFn: (body: DerivedInput) => api.previewDerived(ds.id, body) });
  const refresh = () => {
    for (const k of ["derived", "fields", "fieldQuery", "exportPreview", "corrMatrix", "reviewQueue", "datasets"]) qc.invalidateQueries({ queryKey: [k, ds.id] });
    qc.invalidateQueries({ queryKey: ["dataset", ds.id] });
  };
  const add = useMutation({
    mutationFn: () => api.addDerived(ds.id, { ...input!, name }),
    onSuccess: (r) => { toast({ message: `Added ${r.name}`, tone: "good" }); setName(""); preview.reset(); refresh(); },
  });
  const remove = useMutation({
    mutationFn: (id: number) => api.removeDerived(ds.id, id),
    onSuccess: (r) => { toast({ message: `Removed ${r.removed}`, tone: "good" }); refresh(); },
    onError: (e) => showError("Couldn't remove the field", e),
  });

  const shown = (fields ?? []).filter((f) => !countFilter || f.name.toLowerCase().includes(countFilter.toLowerCase()));
  return (
    <section className="card stack">
      <p className="muted small-text">
        A derived field is computed from this dataset's fields and added to its data. Add them while the baseline is being set up,
        or once it is finalized and then start a new version so they get assessed.
      </p>
      {list.data && list.data.length > 0 && (
        <table className="table">
          <thead><tr><th>Field</th><th>Kind</th><th>Definition</th><th /></tr></thead>
          <tbody>
            {list.data.map((f) => (
              <tr key={f.id}>
                <td>{f.name}</td><td>{KIND_LABEL[f.kind]}</td><td><code>{describeDerived(f)}</code></td>
                <td><button type="button" className="ghost small" disabled={remove.isPending}
                  onClick={() => { if (window.confirm(`Remove ${f.name} and its review?`)) remove.mutate(f.id); }}>Remove</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="chips" role="radiogroup" aria-label="Kind">
        {(Object.keys(KIND_LABEL) as Kind[]).map((k) => (
          <button key={k} type="button" role="radio" aria-checked={kind === k} className={`chip${kind === k ? " on" : ""}`}
            onClick={() => { setKind(k); preview.reset(); }}>{KIND_LABEL[k]}</button>
        ))}
      </div>
      <span className="muted small-text">{KIND_HELP[kind]}</span>
      {kind === "formula" && (
        <input aria-label="Formula" placeholder="water / (cement + blast_furnace_slag + fly_ash)" value={expr} onChange={(e) => setExpr(e.target.value)} />
      )}
      {kind === "groups" && (
        <>
          <label className="row">Field
            <select value={source} onChange={(e) => setSource(e.target.value)}>
              <option value="">Choose…</option>
              {fields?.map((f) => <option key={f.id} value={f.name}>{f.name}</option>)}
            </select>
          </label>
          <textarea aria-label="Groups" rows={5} placeholder={"Circulatory: 390-459, 785\nDiabetes: 250-250.99"} value={groupText} onChange={(e) => setGroupText(e.target.value)} />
          {groupText.trim() && parsed.error && <span className="small-text" style={{ color: "var(--bad)" }}>{parsed.error}</span>}
          <label className="row">Everything else <input value={other} onChange={(e) => setOther(e.target.value)} /></label>
        </>
      )}
      {kind === "count" && (
        <>
          <input aria-label="Filter fields" placeholder={`Filter fields (${countFields.length} chosen)`} value={countFilter} onChange={(e) => setCountFilter(e.target.value)} />
          <div className="chips" style={{ maxHeight: 160, overflowY: "auto" }}>
            {shown.map((f) => {
              const on = countFields.includes(f.name);
              return <button key={f.id} type="button" aria-pressed={on} className={`chip${on ? " on" : ""}`}
                onClick={() => setCountFields(on ? countFields.filter((n) => n !== f.name) : [...countFields, f.name])}>{f.name}</button>;
            })}
          </div>
          <input aria-label="Values to count" placeholder="Up, Down" value={countValues} onChange={(e) => setCountValues(e.target.value)} />
        </>
      )}
      <div className="row">
        <button type="button" disabled={!input || preview.isPending} onClick={() => input && preview.mutate(input)}>Preview</button>
        <input aria-label="New field name" placeholder="New field name" value={name} onChange={(e) => setName(e.target.value)} />
        <button type="button" className="primary" disabled={!input || !name.trim() || add.isPending} onClick={() => add.mutate()}>
          {add.isPending ? "Adding…" : "Add field"}
        </button>
      </div>
      <ErrorAlert error={preview.error ?? add.error} />
      {preview.data && (
        <div className="small-text">
          <span className="muted">{preview.data.type} · {preview.data.distinct.toLocaleString()} distinct · {preview.data.missing.toLocaleString()} of {preview.data.rows.toLocaleString()} missing · first rows: </span>
          <code>{preview.data.sample.map((v) => (v === null ? "—" : typeof v === "number" ? +v.toPrecision(4) : v)).join(", ")}</code>
        </div>
      )}
    </section>
  );
}
