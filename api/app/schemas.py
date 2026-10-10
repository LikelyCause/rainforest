from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SampleRequest(BaseModel):
    rows: int = Field(50_000, ge=1_000, le=2_000_000)
    fields: int = Field(150, ge=30, le=400)
    seed: int = 42
    target_kind: Literal["mixed", "binary", "multiclass"] = "mixed"
    numeric_target: bool = Field(False, description="Add column_c, a continuous target, after column_a and column_b")
    name: str | None = None


class DatasetPatch(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_family: str | None = None
    name: str | None = Field(None, min_length=1, max_length=200)
    target_columns: list[str] | None = Field(None, max_length=50, description="Exactly these targets; [] for none")


class DecisionIn(BaseModel):
    family: str | None = Field(None, description="Model family these settings are for (default: dataset's)")
    action: Literal["keep", "drop", "target"]
    semantic_type: str
    imputation: str = "none"
    impute_value: str | None = None
    add_missing_indicator: bool = False
    transform: str = "none"
    scaling: str = "none"
    encoding: str = "none"
    params: dict[str, Any] = Field(default_factory=dict)
    target_task: Literal["classes", "regression", "bins"] | None = Field(
        None, description="How a target is predicted; omitted keeps the stored task (or the type's default)")
    target_bins: dict[str, Any] | None = Field(None, description='Bands: {"k": 2-20} or {"edges": [ascending numbers]}')
    notes: str | None = Field(None, max_length=4000)
    reviewed: bool = True


FilterOp = Literal["=", "!=", ">", ">=", "<", "<=", "in", "contains", "is_null", "not_null"]


class FieldFilter(BaseModel):
    column: str
    op: FilterOp
    value: Any = None


class BulkDecisionIn(BaseModel):
    """Target set is either explicit `field_ids` or `filters` resolved server-side (same semantics as /fields/query)."""
    field_ids: list[int] | None = Field(None, min_length=1, max_length=5000)
    filters: list[FieldFilter] | None = Field(None, max_length=20)
    excluded_ids: list[int] = Field(default_factory=list, max_length=5000)
    accept_suggestion: bool = False
    action: Literal["keep", "drop"] | None = None
    reviewed: bool | None = None
    dry_run: bool = False

    @model_validator(mode="after")
    def _check(self) -> BulkDecisionIn:
        if (self.field_ids is None) == (self.filters is None):
            raise ValueError("Give exactly one of field_ids or filters")
        if not self.accept_suggestion and self.action is None and self.reviewed is None:
            raise ValueError("Nothing to change: set accept_suggestion, action or reviewed")
        return self


class FieldQuery(BaseModel):
    filters: list[FieldFilter] = Field(default_factory=list, max_length=20)
    order_by: str = "ordinal"
    descending: bool = False
    limit: int = Field(500, ge=1, le=5000)


class SqlRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20_000)


class ExportRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    source: Literal["recipe", "assessment"] | None = Field(
        None, description="recipe: a finalized recipe version; assessment: your own assessment (needs experiment_id). "
                          "Default: the experiment's source, else recipe")
    recipe_id: int | None = Field(None, description="The finalized version to build from (default: the latest)")
    experiment_id: int | None = Field(None, description="The experiment the export belongs to (default for a team "
                                                        "export: the version's Team experiment)")
    model_family: str | None = Field(None, description="Prepare features for this family (default: dataset's)")
    valid_size: float = Field(0.15, ge=0, lt=0.5)
    test_size: float = Field(0.15, ge=0, lt=0.5)
    seed: int = 42
    fit_sample_rows: int = Field(500_000, ge=1_000, le=5_000_000)
    chunk_rows: int = Field(200_000, ge=10_000, le=2_000_000)


class RunRequest(BaseModel):
    model: str = "lightgbm"
    order: list[str] | None = None
    cv: int = Field(5, ge=0, le=10, description="0 chains the ground truth; otherwise 2-10 folds")
    seed: int = 42
    params: dict[str, Any] = Field(default_factory=dict)
    device: Literal["cpu", "cuda"] = "cpu"
    experiment_id: int | None = Field(None, description="Must be the export's experiment, which the run joins")

    @model_validator(mode="after")
    def _check(self) -> RunRequest:
        if self.cv == 1:
            raise ValueError("cv must be 0 (chain the ground truth) or 2-10 folds")
        return self


class AnalysisParams(BaseModel):
    data: str = Field("raw", pattern=r"^(raw|export:\d+)$")
    kept: bool = False
    target: str | None = Field(None, max_length=500)
    sample: int = Field(5_000, ge=1_000, le=20_000)
    method: Literal["tsne", "umap"] = "tsne"
    perplexity: int = Field(30, ge=5, le=100)
    n_neighbors: int = Field(15, ge=5, le=100)

    def normalised(self, kind: str) -> dict[str, Any]:
        """Only the keys that change the result for `kind`, so equivalent requests share a cache key."""
        out: dict[str, Any] = {"data": self.data, "kept": self.kept and self.data == "raw",
                               "target": self.target or None, "sample": self.sample}
        if kind == "map":
            out["method"] = self.method
            if self.method == "tsne":
                out["perplexity"] = self.perplexity
            else:
                out["n_neighbors"] = self.n_neighbors
        return out


class AnalysisRequest(BaseModel):
    kind: Literal["map", "tests", "importance"]
    params: AnalysisParams = Field(default_factory=AnalysisParams)
    recompute: bool = False
    view: str | None = Field(None, max_length=40, description="Whose decisions raw-data analyses use: "
                             "experiment:<id> or baseline:<recipe_id>; default the baseline in progress")


class CorrelationQuery(BaseModel):
    kept: bool = False
    view: str | None = Field(None, max_length=40)
    target: str | None = Field(None, max_length=500)
    limit: int = Field(60, ge=2, le=400)
    add: str | None = Field(None, pattern=r"^\s*\d+(\s*,\s*\d+)*\s*$|^\s*$",
                            description="Comma-separated field ids to include beyond the limit")

    @property
    def add_ids(self) -> list[int]:
        return list(dict.fromkeys(int(v) for v in (self.add or "").split(",") if v.strip()))


class AssessorIn(BaseModel):
    user_id: int


class CommentIn(BaseModel):
    body: str = Field(min_length=1, max_length=4000, pattern=r"\S")


class StartBaselineIn(BaseModel):
    family: str
    start: Literal["scratch", "baseline", "recipe"] = Field(
        "scratch", description="scratch: the suggestions; baseline: a finalized baseline of this dataset; recipe: a "
                               "finalized baseline of another dataset (a new dataset's first baseline only)")
    source_recipe_id: int | None = None
    targets: list[str] = Field(default_factory=list, max_length=50)
    assessors: list[int] = Field(default_factory=list, max_length=200)


class NewVersionIn(BaseModel):
    family: str | None = Field(None, description="The baseline's model family; default: the current baseline's")
    reopen: list[int] = Field(default_factory=list, max_length=5000,
                              description="Fields to assess again; none means the lead edits the decisions directly")


class ApplyRecipeIn(BaseModel):
    recipe_id: int


class FinalizeIn(BaseModel):
    accept_open: bool = Field(False, description="Finalize with disagreements left open, as their team decision stands")


class ExperimentIn(BaseModel):
    name: str = Field(min_length=1, max_length=200, pattern=r"\S")
    source: str = Field(max_length=100, description="recipe:<recipe_id> or assessment:<recipe_id>:<your user id>")
    notes: str | None = Field(None, max_length=10_000)


class RevertIn(BaseModel):
    field_ids: list[int] | None = Field(None, max_length=5000, description="Fields to revert; null for every field")


class ExperimentPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200, pattern=r"\S")
    notes: str | None = Field(None, max_length=10_000)
    archived: bool | None = None


class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str | None = Field(None, max_length=320)
    role: Literal["lead", "member", "viewer"] = "member"


class UserPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    email: str | None = Field(None, max_length=320)
    role: Literal["lead", "member", "viewer"] | None = None
    active: bool | None = None
