from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SampleRequest(BaseModel):
    rows: int = Field(50_000, ge=1_000, le=2_000_000)
    fields: int = Field(150, ge=30, le=400)
    seed: int = 42
    target_kind: Literal["mixed", "binary", "multiclass"] = "mixed"
    name: str | None = None


class DatasetPatch(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_family: str | None = None
    name: str | None = Field(None, min_length=1, max_length=200)


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
    model_family: str | None = Field(None, description="Prepare features for this family (default: dataset's)")
    valid_size: float = Field(0.15, ge=0, lt=0.5)
    test_size: float = Field(0.15, ge=0, lt=0.5)
    seed: int = 42
    fit_sample_rows: int = Field(500_000, ge=1_000, le=5_000_000)
    chunk_rows: int = Field(200_000, ge=10_000, le=2_000_000)


class RunRequest(BaseModel):
    model: str = "lightgbm"
    order: list[str] | None = None
    cv: int = Field(5, ge=0, le=10)
    seed: int = 42
    params: dict[str, Any] = Field(default_factory=dict)
