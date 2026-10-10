export const SAMPLES = [1000, 5000, 10000, 20000];

export interface ExploreParams { data: string; isRaw: boolean; kept: boolean; target: string | null; sample: number }

/**
 * Explore's toolbar state from the URL, made valid for this dataset: raw data unless an export is named,
 * Kept only for raw data alone, a target that is one of the dataset's (else its first), a known sample size.
 */
export function parseExploreParams(params: URLSearchParams, targets: string[]): ExploreParams {
  const d = params.get("data");
  const data = d && /^export:\d+$/.test(d) ? d : "raw";
  const isRaw = data === "raw";
  const t = params.get("target");
  const s = Number(params.get("sample"));
  return {
    data,
    isRaw,
    kept: isRaw && params.get("kept") === "1",
    target: targets.length ? (t && targets.includes(t) ? t : targets[0]) : null,
    sample: SAMPLES.includes(s) ? s : 5000,
  };
}
