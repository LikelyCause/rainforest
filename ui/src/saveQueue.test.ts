// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { DecisionInput } from "./types";

const saveDecision = vi.fn();
const showError = vi.fn();
vi.mock("./api", () => ({ api: { saveDecision: (...a: unknown[]) => saveDecision(...a) } }));
vi.mock("./toast", () => ({ showError: (...a: unknown[]) => showError(...a) }));

const { flushSave, pendingDraft, queueSave } = await import("./saveQueue");

const draft = (notes: string): DecisionInput => ({
  family: "tree", action: "keep", semantic_type: "numeric_continuous", imputation: "none", impute_value: null,
  add_missing_indicator: false, transform: "none", scaling: "none", encoding: "none", params: {}, notes, reviewed: false,
});

/** A promise per call that the test settles by hand. */
function deferredSaves() {
  const calls: { input: DecisionInput; resolve: () => void; reject: (e: Error) => void }[] = [];
  saveDecision.mockImplementation((_id: number, input: DecisionInput) => new Promise((resolve, reject) => {
    calls.push({ input, resolve: () => resolve({ ...input, field_id: 1 }), reject });
  }));
  return calls;
}

const tick = () => new Promise((r) => setTimeout(r, 0));

describe("saveQueue", () => {
  beforeEach(() => {
    saveDecision.mockReset();
    showError.mockReset();
  });

  it("sends one save at a time per field, in order", async () => {
    const calls = deferredSaves();
    const first = queueSave(101, 1, "a", draft("one"));
    await tick();
    const second = queueSave(101, 1, "a", draft("two"));
    await tick();
    expect(calls.map((c) => c.input.notes)).toEqual(["one"]);  // "two" waits for "one"
    calls[0].resolve();
    await tick();
    expect(calls.map((c) => c.input.notes)).toEqual(["one", "two"]);
    calls[1].resolve();
    await Promise.all([first, second]);
    expect(pendingDraft(101)).toBeNull();
  });

  it("skips drafts that a newer one replaced while a save was in flight", async () => {
    const calls = deferredSaves();
    void queueSave(102, 1, "b", draft("one"));
    await tick();
    void queueSave(102, 1, "b", draft("two"));
    void queueSave(102, 1, "b", draft("three"));
    expect(pendingDraft(102)?.notes).toBe("three");
    calls[0].resolve();
    await tick();
    await tick();
    expect(calls.map((c) => c.input.notes)).toEqual(["one", "three"]);
    calls[1].resolve();
    await tick();
  });

  it("keeps lanes of different fields independent", async () => {
    const calls = deferredSaves();
    void queueSave(103, 1, "c", draft("c"));
    void queueSave(104, 1, "d", draft("d"));
    await tick();
    expect(calls.map((c) => c.input.notes).sort()).toEqual(["c", "d"]);
    calls.forEach((c) => c.resolve());
    await tick();
  });

  it("keeps a failed draft and retries it from the error toast", async () => {
    const calls = deferredSaves();
    const done = queueSave(105, 1, "e", draft("lost?"));
    await tick();
    calls[0].reject(new Error("offline"));
    await done;
    expect(pendingDraft(105)?.notes).toBe("lost?");
    expect(showError).toHaveBeenCalledOnce();
    const [title, , action] = showError.mock.calls[0];
    expect(title).toContain("e");
    expect(action.label).toBe("Retry");
    action.run();
    await tick();
    expect(calls.map((c) => c.input.notes)).toEqual(["lost?", "lost?"]);
    calls[1].resolve();
    await tick();
    await tick();
    expect(pendingDraft(105)).toBeNull();
  });

  it("debounces typing until the delay passes or a flush", async () => {
    vi.useFakeTimers();
    try {
      const calls = deferredSaves();
      void queueSave(106, 1, "f", draft("t"), 400);
      void queueSave(106, 1, "f", draft("ty"), 400);
      expect(calls).toHaveLength(0);
      await vi.advanceTimersByTimeAsync(400);
      expect(calls.map((c) => c.input.notes)).toEqual(["ty"]);
      calls[0].resolve();
      void queueSave(106, 1, "f", draft("typed"), 400);
      void flushSave(106);
      await vi.advanceTimersByTimeAsync(0);
      expect(calls.map((c) => c.input.notes)).toEqual(["ty", "typed"]);
      calls[1].resolve();
    } finally {
      vi.useRealTimers();
    }
  });
});
