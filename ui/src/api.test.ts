import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  REFRESH_LOCKED,
  api,
  errorFromBody,
  isRefreshLock,
  isServerPath,
  safeNext,
  setRefreshLockHandler,
} from "./api";

describe("errorFromBody", () => {
  it("keeps the message and the details of every error shape", () => {
    expect(errorFromBody(404, { detail: "Dataset not found" })).toMatchObject({
      message: "Dataset not found",
      details: [],
    });
    expect(
      errorFromBody(422, { detail: { message: "Not a valid target", errors: ["price: 5,000 distinct values"] } }),
    ).toMatchObject({ message: "Not a valid target", details: ["price: 5,000 distinct values"] });
    expect(
      errorFromBody(400, { detail: { message: "Only one statement", detail: "syntax error at ;" } }).details,
    ).toEqual(["syntax error at ;"]);
    const e = errorFromBody(422, { detail: [{ loc: ["body", "rows"], msg: "too big" }] });
    expect(e).toBeInstanceOf(ApiError);
    expect(e).toMatchObject({ message: "Invalid request", status: 422, details: ["body.rows: too big"] });
  });

  it("states a refused pull's estimate against the limit", () => {
    expect(
      errorFromBody(422, { detail: { message: "Too large", estimated: 700_000_000, max: 600_000_000 } }),
    ).toMatchObject({ message: "Too large", details: ["700,000,000 cells exceeds the limit of 600,000,000"] });
  });

  it("falls back for bodies that aren't API errors", () => {
    expect(errorFromBody(502, null).message).toBe("Request failed (502)");
    expect(errorFromBody(413, "<html>", "Upload failed (413)").message).toBe("Upload failed (413)");
  });
});

describe("the refresh lock", () => {
  const answer = (status: number, detail: string) =>
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail }), { status })),
    );
  afterEach(() => {
    vi.unstubAllGlobals();
    setRefreshLockHandler(() => {});
  });

  it("is the 409 with the lock's message and nothing else", () => {
    expect(isRefreshLock(errorFromBody(409, { detail: REFRESH_LOCKED }))).toBe(true);
    expect(isRefreshLock(errorFromBody(409, { detail: "A pull of this dataset is already running" }))).toBe(false);
    expect(isRefreshLock(errorFromBody(422, { detail: REFRESH_LOCKED }))).toBe(false);
    expect(isRefreshLock(new Error(REFRESH_LOCKED))).toBe(false);
  });

  it("tells the handler when a call is refused by it, and still fails the call", async () => {
    const told = vi.fn();
    setRefreshLockHandler(told);
    answer(409, REFRESH_LOCKED);
    await expect(api.dataset(1)).rejects.toMatchObject({ status: 409, message: REFRESH_LOCKED });
    expect(told).toHaveBeenCalledTimes(1);
  });

  it("leaves other refusals alone", async () => {
    const told = vi.fn();
    setRefreshLockHandler(told);
    answer(409, "Only a finished run can be chosen");
    await expect(api.dataset(1)).rejects.toMatchObject({ status: 409 });
    expect(told).not.toHaveBeenCalled();
  });
});

describe("safeNext", () => {
  it("only returns same-app paths", () => {
    expect(safeNext("/d/1/review")).toBe("/d/1/review");
    expect(safeNext("//evil.example")).toBe("/");
    expect(safeNext("https://evil.example")).toBe("/");
    expect(safeNext("/login?next=/x")).toBe("/");
    expect(safeNext("/forbidden?reason=no_group")).toBe("/");
    expect(safeNext("/\\evil.example")).toBe("/");
    expect(safeNext(null)).toBe("/");
  });
});

describe("isServerPath", () => {
  it("is true only under /mlflow/", () => {
    expect(isServerPath("/mlflow/")).toBe(true);
    expect(isServerPath("/mlflow/#/experiments/1")).toBe(true);
    expect(isServerPath("/mlflowx")).toBe(false);
    expect(isServerPath("/d/1/export")).toBe(false);
  });
});
