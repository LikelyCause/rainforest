import { initials, tint } from "../avatar";

/** An account's circle: its picture, or its initials on a colour of its own. */
export function Avatar({
  user,
  size,
}: {
  user: { id: number; name: string; avatar_url?: string | null };
  size?: number | string;
}) {
  const style =
    size !== undefined
      ? { width: size, height: size, fontSize: typeof size === "number" ? size * 0.38 : undefined }
      : undefined;
  return user.avatar_url ? (
    <img className="avatar" src={user.avatar_url} alt="" style={style} />
  ) : (
    <span
      className="avatar initials"
      style={{ ...style, background: `var(--cat-${tint(user.id)})` }}
      aria-hidden="true"
    >
      {initials(user.name)}
    </span>
  );
}
