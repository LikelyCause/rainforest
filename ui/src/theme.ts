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
