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
  font: string;
}

function read(): ChartTheme {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string) => cs.getPropertyValue(name).trim();
  return {
    surface: v("--surface-1"),
    text: v("--text-primary"),
    textSecondary: v("--text-secondary"),
    textMuted: v("--text-muted"),
    grid: v("--grid"),
    series: v("--series-1"),
    negative: v("--series-neg"),
    muted: v("--series-muted"),
    accentWeak: v("--accent-weak"),
    font: v("--font"),
  };
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
