/** Up to two initials: the first word's and the last word's. */
export function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return "?";
  const first = words[0][0];
  return (words.length > 1 ? first + words[words.length - 1][0] : first).toUpperCase();
}

/** A palette slot (1-8) that stays the same for an account. */
export function tint(id: number): number {
  let h = id * 2654435761;
  h = (h ^ (h >>> 16)) >>> 0;
  return (h % 8) + 1;
}
