"""What the datasets list, team progress and the Baselines page tell about the team process at a glance: assessor
counts, how each target is predicted (as the viewer may read it), open disagreements, last activity and each
version's review summary."""
from .team_helpers import app_, fields, people, review_all, save, start, upload  # noqa: F401


def _listed(client, ds) -> dict:
    return next(x for x in client.get("/api/datasets").json() if x["id"] == ds["id"])


def _assessors(client, ds) -> dict[int, dict]:
    return {a["user_id"]: a for a in client.get(f"/api/datasets/{ds['id']}/team").json()["assessors"]}


def _review(client, ds) -> dict | None:
    return client.get(f"/api/datasets/{ds['id']}/baselines").json()[0]["versions"][0]["review"]


def test_counts_kinds_open_disagreements_and_review_summaries(people):
    from app.db import connection

    lead, a, b, c = people["lead"], people["a"], people["b"], people["c"]
    ds = upload(lead)
    try:
        listed = _listed(c, ds)
        assert listed["assessors"] is None and listed["review_open"] is None
        start(lead, ds, "tree", [a, b])
        listed = _listed(c, ds)
        assert listed["assessors"] == {"total": 2, "submitted": 0}
        assert listed["target_kinds"] == {"y": "classes"} and listed["target_kinds_suggested"] is True
        assert _review(c, ds) is None

        # B predicts y as one class against the rest in their own assessment; nobody else reads that.
        f = fields(b, ds)
        assert save(b, f["y"]["id"], target_task="bins", target_bins={"positive": ["a"]}).status_code == 200
        assert _listed(b, ds)["target_kinds"] == {"y": "bands"} and not _listed(b, ds)["target_kinds_suggested"]
        for who in (a, c, lead):
            assert _listed(who, ds)["target_kinds"] == {"y": "classes"}
        assert _listed(a, ds)["target_kinds_suggested"] is False, "an assessor reads their own assessment"

        progress = _assessors(c, ds)
        assert progress[b.user["id"]]["last_active_at"] is not None
        assert progress[a.user["id"]]["last_active_at"] is None, "nothing changed since A was assigned"

        # They disagree on x (and on y's task); the rest is adopted.
        assert save(a, f["x"]["id"], action="drop").status_code == 200
        assert _assessors(c, ds)[a.user["id"]]["last_active_at"] is not None
        for who in (a, b):
            review_all(who, ds)
            assert who.post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 200
        listed = _listed(c, ds)
        assert listed["phase"] == "review" and listed["assessors"] == {"total": 2, "submitted": 2}
        assert listed["review_open"] == 2 and listed["target_kinds_suggested"] is False
        assert _review(c, ds) == {"adopted": 3, "disagreements": 2, "settled": 0, "open_at_finalize": None}

        assert save(lead, f["x"]["id"], action="drop", reviewed=True).status_code == 200
        assert _listed(c, ds)["review_open"] == 1
        r = lead.post(f"/api/datasets/{ds['id']}/finalize", json={"accept_open": True})
        assert r.status_code == 200, r.text
        assert _listed(c, ds)["review_open"] is None
        assert _review(c, ds) == {"adopted": 3, "disagreements": 2, "settled": 1, "open_at_finalize": 1}

        # A version finalized before summaries were kept: what the stored assessments still tell.
        rid = r.json()["recipe"]["id"]
        with connection() as conn:
            conn.execute("ALTER TABLE meta.recipes DISABLE TRIGGER recipes_frozen")
            conn.execute("UPDATE meta.recipes SET review_summary = NULL WHERE id = %s", (rid,))
            conn.execute("ALTER TABLE meta.recipes ENABLE TRIGGER recipes_frozen")
        assert _review(c, ds) == {"adopted": 3, "disagreements": 2, "settled": None, "open_at_finalize": None}
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_a_number_target_reads_as_number(people):
    lead, c = people["lead"], people["c"]
    ds = upload(lead, targets="z")
    try:
        start(lead, ds, "tree", [lead], targets=("z",))
        assert _listed(lead, ds)["target_kinds"] == {"z": "number"}
        assert _listed(lead, ds)["target_kinds_suggested"] is False
        assert _listed(c, ds)["target_kinds_suggested"] is True
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")
