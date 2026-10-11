import { useEffect, useState } from "react";

export interface ChartTheme {
  surface: string;
  text: string;
  textSecondary: string;
  textMuted: string;
  grid: string;
  series: string;
  negative: string;
  muted: string;
  accentWeak: string;
  target: string;
  categorical: string[];
  seq0: string;
  seqLo: string;
  seqHi: string;
  divMid: string;
  font: string;
}

// Tokens are light-dark() expressions, which getPropertyValue returns unresolved; a probe's computed colour resolves them.
function read(): ChartTheme {
  const probe = document.createElement("span");
  probe.style.display = "none";
  document.body.appendChild(probe);
  const color = (name: string) => {
    probe.style.color = `var(${name})`;
    return getComputedStyle(probe).color;
  };
  const theme = {
    surface: color("--surface-1"),
    text: color("--text-primary"),
    textSecondary: color("--text-secondary"),
    textMuted: color("--text-muted"),
    grid: color("--grid"),
    series: color("--series-1"),
    negative: color("--series-neg"),
    muted: color("--series-muted"),
    accentWeak: color("--accent-weak"),
    target: color("--target"),
    categorical: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => color(`--cat-${i}`)),
    seq0: color("--seq-0"),
    seqLo: color("--seq-lo"),
    seqHi: color("--seq-hi"),
    divMid: color("--div-mid"),
    font: getComputedStyle(document.documentElement).getPropertyValue("--font").trim(),
  };
  probe.remove();
  return theme;
}

/** Chart colours resolved from CSS tokens; re-resolves when the OS colour scheme flips. */
export function useChartTheme(): ChartTheme {
  const [theme, setTheme] = useState<ChartTheme>(read);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setTheme(read());
    mq.addEventListener("change", update);
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq.removeEventListener("change", update);
      obs.disconnect();
    };
  }, []);
  return theme;
}

/** The colour scheme someone chose: follow the OS, or pin light or dark. Kept per browser. */
export type ThemePref = "system" | "light" | "dark";
const THEME_KEY = "rainforest.theme";

export function getThemePref(): ThemePref {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

/** Pins the scheme with data-theme on <html> (styles.css and useChartTheme follow it), or unpins it for "system". */
export function applyThemePref(pref: ThemePref = getThemePref()): void {
  if (pref === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = pref;
}

export function setThemePref(pref: ThemePref): void {
  try {
    if (pref === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, pref);
  } catch {
    // Storage blocked (private window): the choice still applies until the page reloads.
  }
  applyThemePref(pref);
}
