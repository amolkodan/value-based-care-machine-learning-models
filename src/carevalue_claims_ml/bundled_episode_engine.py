from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


def _norm_code(value: Any, *, upper: bool = True, max_len: int | None = None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    if upper:
        text = text.upper()
    text = re.sub(r"\s+", "", text)
    if max_len is not None and len(text) > max_len:
        text = text[:max_len]
    return text


def _icd_prefix(code: str, n: int = 3) -> str:
    c = _norm_code(code)
    if not c:
        return ""
    body = c.replace(".", "")
    return body[:n] if len(body) >= n else body


def _ndc_prefix(code: str, n: int = 9) -> str:
    c = re.sub(r"[^0-9A-Za-z]", "", _norm_code(code))
    return c[:n] if c else ""


def claim_row_to_feature_text(
    row: pd.Series,
    *,
    care_domain_col: str | None = "care_domain",
    diagnosis_code_col: str | None = "diagnosis_code",
    procedure_code_col: str | None = "procedure_code",
    ndc_col: str | None = "ndc",
    amount_col: str = "allowed_amount",
) -> str:
    """
    Build a single sparse-friendly text blob per claim line for hashing models.
    Prefixes keep ICD / CPT / HCPCS / NDC semantics explicit for the vectorizer.
    """
    parts: list[str] = []
    if care_domain_col and care_domain_col in row.index:
        dom = _norm_code(row.get(care_domain_col), upper=False)
        if dom:
            parts.append(f"dom:{dom}")
    if diagnosis_code_col and diagnosis_code_col in row.index:
        icd = _norm_code(row.get(diagnosis_code_col))
        if icd:
            parts.append(f"icd:{icd}")
            parts.append(f"icdp:{_icd_prefix(icd)}")
    if procedure_code_col and procedure_code_col in row.index:
        proc = _norm_code(row.get(procedure_code_col))
        if proc:
            parts.append(f"proc:{proc}")
            parts.append(f"procp:{proc[:5]}")
    if ndc_col and ndc_col in row.index:
        ndc = _norm_code(row.get(ndc_col), upper=False)
        if ndc:
            parts.append(f"ndc:{ndc}")
            pfx = _ndc_prefix(ndc)
            if pfx:
                parts.append(f"ndcp:{pfx}")
    if amount_col in row.index:
        try:
            amt = float(row.get(amount_col) or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        bucket = int(np.floor(np.log1p(max(amt, 0.0)) / np.log(1.05)))
        parts.append(f"amtb:{bucket}")
    return " ".join(parts) if parts else "empty"


def _parse_episode_labels_cell(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    if "|" in text:
        return [p.strip() for p in text.split("|") if p.strip()]
    return [text]


@dataclass
class EpisodeCodeDefinitions:
    """
    Human-readable episode definitions mined from labeled bundles: frequent
    ICD prefixes, procedure prefixes, and NDC prefixes per episode family label.
    """

    by_episode_family: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {"by_episode_family": self.by_episode_family, "metadata": self.metadata}
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> EpisodeCodeDefinitions:
        data = json.loads(text)
        return cls(
            by_episode_family=data.get("by_episode_family", {}),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EpisodeCodeDefinitions:
        return cls.from_json(path.read_text(encoding="utf-8"))


def learn_episode_definitions_from_labels(
    claims_df: pd.DataFrame,
    *,
    episode_family_col: str,
    diagnosis_code_col: str | None = "diagnosis_code",
    procedure_code_col: str | None = "procedure_code",
    ndc_col: str | None = "ndc",
    top_n: int = 25,
    min_support: int = 2,
) -> EpisodeCodeDefinitions:
    """
    Summarize which ICD, CPT/HCPCS, and NDC patterns co-occur with each episode
    family in historical bundled-labeled data (supervised episode mining).
    """
    if episode_family_col not in claims_df.columns:
        raise ValueError(f"missing column {episode_family_col!r}")

    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family, chunk in claims_df.groupby(episode_family_col):
        fam = str(family)
        icd_counts: dict[str, int] = {}
        proc_counts: dict[str, int] = {}
        ndc_counts: dict[str, int] = {}
        n_rows = len(chunk)
        if diagnosis_code_col and diagnosis_code_col in chunk.columns:
            for val in chunk[diagnosis_code_col].dropna():
                p = _icd_prefix(str(val))
                if p:
                    icd_counts[p] = icd_counts.get(p, 0) + 1
        if procedure_code_col and procedure_code_col in chunk.columns:
            for val in chunk[procedure_code_col].dropna():
                pr = _norm_code(val)[:5]
                if pr:
                    proc_counts[pr] = proc_counts.get(pr, 0) + 1
        if ndc_col and ndc_col in chunk.columns:
            for val in chunk[ndc_col].dropna():
                nd = _ndc_prefix(str(val), n=9)
                if nd:
                    ndc_counts[nd] = ndc_counts.get(nd, 0) + 1

        def top_table(counts: dict[str, int]) -> list[dict[str, Any]]:
            rows = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
            return [
                {"code_prefix": k, "count": int(v), "share": float(v) / max(n_rows, 1)}
                for k, v in rows
                if v >= min_support
            ]

        out[fam] = {
            "icd_prefixes": top_table(icd_counts),
            "procedure_prefixes": top_table(proc_counts),
            "ndc_prefixes": top_table(ndc_counts),
            "labeled_row_count": int(n_rows),
        }

    return EpisodeCodeDefinitions(
        by_episode_family=out,
        metadata={"episode_family_col": episode_family_col},
    )


@dataclass
class BundledEpisodeAttributionModel:
    """
    Multi-label episode-family attribution from medical + pharmacy claim lines
    using hashed sparse n-grams over code-oriented feature text.

    Training rows may each carry one or more episode family labels (e.g. a line
    can legitimately touch multiple bundled episodes in concurrent morbidity).
    """

    pipeline: Pipeline
    mlb: MultiLabelBinarizer
    feature_config: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "mlb": self.mlb, "feature_config": self.feature_config}, path)

    @classmethod
    def load(cls, path: Path) -> BundledEpisodeAttributionModel:
        blob = joblib.load(path)
        return cls(
            pipeline=blob["pipeline"],
            mlb=blob["mlb"],
            feature_config=blob.get("feature_config", {}),
        )

    def predict_proba_matrix(self, claims_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        texts = [claim_row_to_feature_text(claims_df.iloc[i], **self.feature_config) for i in range(len(claims_df))]
        probas = self.pipeline.predict_proba(texts)
        return probas, list(self.mlb.classes_)

    def predict_multi_attribution(
        self,
        claims_df: pd.DataFrame,
        *,
        min_probability: float = 0.12,
        max_labels_per_row: int = 8,
        claim_id_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Return long-format claim-to-episode-family probabilities above ``min_probability``,
        sorted descending per claim (multi-attribution).
        """
        probas, classes = self.predict_proba_matrix(claims_df)
        rows_out: list[dict[str, Any]] = []
        for i in range(len(claims_df)):
            row = claims_df.iloc[i]
            scores = probas[i]
            order = np.argsort(-scores)
            kept = 0
            for j in order:
                p = float(scores[j])
                if p < min_probability:
                    break
                rec: dict[str, Any] = {
                    "episode_family": classes[int(j)],
                    "attribution_probability": p,
                    "claim_row_index": i,
                }
                if claim_id_col and claim_id_col in claims_df.columns:
                    rec[claim_id_col] = row[claim_id_col]
                for c in claims_df.columns:
                    if c not in rec:
                        rec[c] = row[c]
                rows_out.append(rec)
                kept += 1
                if kept >= max_labels_per_row:
                    break
        return pd.DataFrame(rows_out)


def fit_bundled_episode_attribution_model(
    claims_df: pd.DataFrame,
    *,
    episode_labels_col: str | None = None,
    episode_labels_list_col: str | None = None,
    n_features: int = 1 << 18,
    max_iter: int = 2000,
    claim_row_to_feature_text_kwargs: dict[str, Any] | None = None,
) -> BundledEpisodeAttributionModel:
    """
    Fit multi-label episode attribution. Supply either:

    - ``episode_labels_list_col``: cell values are ``"ortho;cardio"`` or lists, or
    - ``episode_labels_col``: single string label per row (wrapped as one-element list).
    """
    if (episode_labels_col is None) == (episode_labels_list_col is None):
        raise ValueError("provide exactly one of episode_labels_col or episode_labels_list_col")

    kw = dict(
        care_domain_col="care_domain",
        diagnosis_code_col="diagnosis_code",
        procedure_code_col="procedure_code",
        ndc_col="ndc",
        amount_col="allowed_amount",
    )
    if claim_row_to_feature_text_kwargs:
        kw.update(claim_row_to_feature_text_kwargs)

    y_list: list[list[str]] = []
    if episode_labels_list_col is not None:
        if episode_labels_list_col not in claims_df.columns:
            raise ValueError(f"missing {episode_labels_list_col!r}")
        for raw in claims_df[episode_labels_list_col]:
            y_list.append(_parse_episode_labels_cell(raw))
    else:
        assert episode_labels_col is not None
        if episode_labels_col not in claims_df.columns:
            raise ValueError(f"missing {episode_labels_col!r}")
        for raw in claims_df[episode_labels_col]:
            lab = str(raw).strip() if raw is not None and not (isinstance(raw, float) and np.isnan(raw)) else ""
            y_list.append([lab] if lab else [])

    if not any(y_list):
        raise ValueError("all episode label lists are empty; nothing to fit")

    texts = [claim_row_to_feature_text(claims_df.iloc[i], **kw) for i in range(len(claims_df))]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)

    hasher = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        ngram_range=(1, 3),
        analyzer="word",
        token_pattern=r"[^\s]+",
    )
    base = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
    )
    ovr = OneVsRestClassifier(base, n_jobs=-1)
    pipeline = Pipeline([("vec", hasher), ("clf", ovr)])
    pipeline.fit(texts, Y)

    return BundledEpisodeAttributionModel(pipeline=pipeline, mlb=mlb, feature_config=kw)


def materialize_gap_episodes_per_family(
    claims_df: pd.DataFrame,
    multi_attr_df: pd.DataFrame,
    *,
    member_col: str = "member_id",
    date_col: str = "service_date",
    amount_col: str = "allowed_amount",
    episode_family_col: str = "episode_family",
    prob_col: str = "attribution_probability",
    window_days: int = 90,
    min_probability: float = 0.15,
    npi_col: str | None = "rendering_npi",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within each member and predicted episode family, apply gap-based episode
    segmentation so each run of claims separated by > ``window_days`` becomes
    a concrete ``episode_instance_id``.

    Returns:

    - ``episodes_df``: one row per episode instance with span, spend, top NPI.
    - ``claim_episode_df``: claim rows with ``episode_instance_id`` attached.
    """
    claims_pos = claims_df.reset_index(drop=True).copy()
    claims_pos[date_col] = pd.to_datetime(claims_pos[date_col])
    attr = multi_attr_df.copy()
    attr[date_col] = pd.to_datetime(attr[date_col])
    attr = attr[attr[prob_col] >= min_probability]

    if "claim_row_index" not in attr.columns:
        raise ValueError("multi_attr_df must include claim_row_index (from predict_multi_attribution)")
    attr["_row_idx"] = attr["claim_row_index"].astype(int)

    episode_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []

    for (member, family), grp in attr.groupby([member_col, episode_family_col], sort=False):
        grp = grp.sort_values(prob_col, ascending=False).drop_duplicates("_row_idx", keep="first")
        grp = grp.sort_values([date_col, "_row_idx"], kind="mergesort")
        sub_indices = grp["_row_idx"].to_numpy(dtype=int)
        prob_by_pos = grp.set_index("_row_idx")[prob_col].to_dict()
        sub = claims_pos.iloc[sub_indices].copy()
        sub = sub.sort_values([date_col], kind="mergesort")
        if sub.empty:
            continue
        sub["attribution_probability"] = sub.index.map(lambda pos: float(prob_by_pos[int(pos)]))
        spacing = sub[date_col].diff().dt.days.fillna(window_days + 1)
        sub["_episode_open"] = (spacing > int(window_days)).astype(int)
        sub["_ep_idx"] = sub["_episode_open"].cumsum()
        sub["episode_instance_id"] = (
            str(member) + "_" + str(family) + "_seg" + sub["_ep_idx"].astype(str)
        )
        for eid, ech in sub.groupby("episode_instance_id", sort=False):
            start = ech[date_col].min()
            end = ech[date_col].max()
            allowed = float(ech[amount_col].fillna(0).sum())
            npi_attributed: str | None = None
            if npi_col and npi_col in ech.columns:
                weights = ech[amount_col].fillna(0).astype(float)
                vc = ech.assign(_w=weights).groupby(npi_col, dropna=True)["_w"].sum()
                if len(vc) > 0:
                    npi_attributed = str(vc.idxmax())
            episode_rows.append(
                {
                    "episode_instance_id": eid,
                    member_col: member,
                    episode_family_col: family,
                    "episode_start": start,
                    "episode_end": end,
                    "episode_allowed_total": allowed,
                    "claim_line_count": int(len(ech)),
                    "attributed_npi": npi_attributed,
                }
            )
        for _, r in sub.iterrows():
            claim_rows.append(dict(r))

    episodes_out = pd.DataFrame(episode_rows)
    claims_out = pd.DataFrame(claim_rows)
    return episodes_out, claims_out


def run_bundled_episode_engine(
    claims_df: pd.DataFrame,
    model: BundledEpisodeAttributionModel,
    *,
    min_probability: float = 0.12,
    window_days: int = 90,
    materialize_min_probability: float = 0.15,
    npi_col: str | None = "rendering_npi",
) -> dict[str, pd.DataFrame]:
    """
    End-to-end: multi-label attribution + gap-based episode instances + provider rollups.
    """
    multi = model.predict_multi_attribution(claims_df, min_probability=min_probability)
    episodes_df, claim_episode_df = materialize_gap_episodes_per_family(
        claims_df,
        multi,
        window_days=window_days,
        min_probability=materialize_min_probability,
        npi_col=npi_col,
    )
    return {"claim_episode_attribution": multi, "episodes": episodes_df, "claims_in_episodes": claim_episode_df}


def episode_family_from_episode_id(episode_id: Any, *, archetype_index: int = 1) -> str:
    """
    Derive a coarse episode family token from gap-built ``episode_id`` strings
    shaped like ``member_archetype_idx`` (see ``build_bundled_episodes``).
    """
    text = str(episode_id)
    parts = text.split("_")
    if len(parts) > archetype_index:
        return parts[archetype_index]
    return text


def training_frame_from_gap_bundles(
    claims_df: pd.DataFrame,
    *,
    archetype: str = "general",
    window_days: int = 90,
    episode_id_col: str = "episode_id",
    episode_family_col: str = "episode_family",
) -> pd.DataFrame:
    """
    Convenience: run deterministic gap bundling, then add ``episode_family``
    parsed from ``episode_id`` for supervised ML training on legacy claims.
    """
    from carevalue_claims_ml.episodes import build_bundled_episodes

    bundled = build_bundled_episodes(
        claims_df, archetype=archetype, window_days=window_days
    )
    bundled[episode_family_col] = bundled[episode_id_col].map(episode_family_from_episode_id)
    return bundled
