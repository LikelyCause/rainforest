import { describe, expect, it } from "vitest";
import { ApiError, errorFromBody, isServerPath, safeNext } from "./api";

describe("errorFromBody", () => {
  it("keeps the message and the details of every error shape", () => {
    expect(errorFromBody(404, { detail: "Dataset not found" })).toMatchObject({ message: "Dataset not found", details: [] });
    expect(errorFromBody(422, { detail: { message: "Not a valid target", errors: ["price: 5,000 distinct values"] } }))
      .toMatchObject({ message: "Not a valid target", details: ["price: 5,000 distinct values"] });
    expect(errorFromBody(400, { detail: { message: "Only one statement", detail: "syntax error at ;" } }).details)
      .toEqual(["syntax error at ;"]);
    const e = errorFromBody(422, { detail: [{ loc: ["body", "rows"], msg: "too big" }] });
    expect(e).toBeInstanceOf(ApiError);
    expect(e).toMatchObject({ message: "Invalid request", status: 422, details: ["body.rows: too big"] });
  });

  it("falls back for bodies that aren't API errors", () => {
    expect(errorFromBody(502, null).message).toBe("Request failed (502)");
    expect(errorFromBody(413, "<html>", "Upload failed (413)").message).toBe("Upload failed (413)");
  });
});

describe("safeNext", () => {
  it("only returns same-app paths", () => {
    expect(safeNext("/d/1/review")).toBe("/d/1/review");
    expect(safeNext("//evil.example")).toBe("/");
    expect(safeNext("https://evil.example")).toBe("/");
    expect(safeNext("/login?next=/x")).toBe("/");
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
