from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis import ingest  # noqa: E402
from cpkanalysis.models import SiteDescription, SourceFile  # noqa: E402


def _build_ptr_record(
    *,
    test_num: int = 1,
    test_name: str = "VDD",
    result: float = 1.2,
    lo_limit: float | None = 1.0,
    hi_limit: float | None = 2.0,
    opt_flg: int = 0,
    res_scal: int = 0,
    llm_scal: int = 0,
    hlm_scal: int = 0,
    parm_flg: int = 0,
    test_flg: int = 0,
    test_tim: float = 0.5,
) -> dict:
    return {
        "TEST_NUM": test_num,
        "TEST_TXT": test_name,
        "UNITS": "V",
        "RES_SCAL": res_scal,
        "RESULT": result,
        "LLM_SCAL": llm_scal,
        "LO_LIMIT": lo_limit,
        "HLM_SCAL": hlm_scal,
        "HI_LIMIT": hi_limit,
        "OPT_FLG": opt_flg,
        "TEST_FLG": test_flg,
        "PARM_FLG": parm_flg,
        "TEST_TIM": test_tim,
    }


def _build_sdr_payload(
    *,
    head_num: int = 0,
    site_group: int = 1,
    site_numbers: Iterable[int] = (1,),
    **extra: object,
) -> dict:
    site_tuple = tuple(site_numbers)
    payload: dict[str, object] = {
        "HEAD_NUM": head_num,
        "SITE_GRP": site_group,
        "SITE_CNT": len(site_tuple),
        "SITE_NUM": list(site_tuple),
    }
    payload.update(extra)
    return payload


@dataclass(frozen=True)
class _FakeRecord:
    name: str
    payload: dict

    def to_dict(self) -> dict:
        return dict(self.payload)


class _FakeSTDFReader:
    """Minimal stub that mimics the istfd reader used by ingest."""

    records_map: dict[Path, list[_FakeRecord]] = {}

    def __init__(self, path: Path, ignore_unknown: bool = True) -> None:
        del ignore_unknown  # noise
        resolved = Path(path).resolve()
        try:
            self._records = list(self.records_map[resolved])
        except KeyError:  # pragma: no cover - defensive guard for misconfigured test
            raise FileNotFoundError(f"No fake STDF records registered for {resolved}") from None

    def __enter__(self) -> "_FakeSTDFReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing to clean up
        return None

    def __iter__(self) -> Iterable[_FakeRecord]:
        return iter(self._records)


def _make_source(tmp_path: Path, name: str, records: list[_FakeRecord]) -> SourceFile:
    path = tmp_path / name
    path.write_bytes(b"stdf")
    _FakeSTDFReader.records_map[path.resolve()] = records
    return SourceFile(path)


def test_ingest_sources_combines_files_and_writes_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "STDFReader", _FakeSTDFReader)

    file_one_records = [
        _FakeRecord("PIR", {"PART_ID": "A1", "SITE_NUM": 1}),
        _FakeRecord("PTR", _build_ptr_record(test_num=1, test_name="VDD", result=1.1)),
        # Invalid measurement should be filtered out
        _FakeRecord("PTR", _build_ptr_record(test_num=2, test_name="IDDQ", result=0.2, parm_flg=ingest.RESULT_INVALID)),
        _FakeRecord("PRR", {"PART_ID": "A1", "SITE_NUM": 1, "PART_FLG": 0}),
    ]
    file_two_records = [
        _FakeRecord("PIR", {"PART_ID": "B2", "SITE_NUM": 3}),
        _FakeRecord("PTR", _build_ptr_record(test_num=3, test_name="IOUT", result=5.5)),
        _FakeRecord("PRR", {"PART_ID": "B2", "SITE_NUM": 3, "PART_FLG": 0x08}),
    ]

    source_one = _make_source(tmp_path, "first.stdf", file_one_records)
    source_two = _make_source(tmp_path, "second.stdf", file_two_records)

    session_dir = tmp_path / "session"
    result = ingest.ingest_sources([source_one, source_two], session_dir)

    # Two valid measurements should be present (invalid PTR filtered out)
    assert len(result.frame) == 2
    assert result.frame["file"].tolist() == ["first.stdf", "second.stdf"]
    assert result.frame["test_name"].tolist() == ["VDD", "IOUT"]
    assert result.frame["site"].tolist() == [1, 3]

    # Per-file stats track measurement counts and invalid measurement filtering
    stats_by_file = {entry["file"]: entry for entry in result.per_file_stats}
    assert stats_by_file["first.stdf"]["measurement_count"] == 1
    assert stats_by_file["first.stdf"]["invalid_measurements_filtered"] == 1
    assert stats_by_file["second.stdf"]["measurement_count"] == 1

    # Ensure parquet store exists and round-tripping matches the in-memory frame
    assert result.raw_store_path.exists()
    parquet_frame = pd.read_parquet(result.raw_store_path, engine="pyarrow")
    pd.testing.assert_frame_equal(parquet_frame, result.frame)


def test_ingest_sources_tracks_indices_and_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "STDFReader", _FakeSTDFReader)

    records = [
        _FakeRecord("PIR", {"PART_ID": "DUP", "SITE_NUM": 1}),
        _FakeRecord("PTR", _build_ptr_record(test_num=10, test_name="GAIN", result=0.9)),
        _FakeRecord("PRR", {"PART_ID": "DUP", "SITE_NUM": 1, "PART_FLG": 0}),
        _FakeRecord("PIR", {"PART_ID": "DUP", "SITE_NUM": 1}),
        _FakeRecord("PTR", _build_ptr_record(test_num=10, test_name="GAIN", result=1.1)),
        _FakeRecord("PRR", {"PART_ID": "DUP", "SITE_NUM": 1, "PART_FLG": 0x08}),
        _FakeRecord("PIR", {"PART_ID": None, "SITE_NUM": 2}),
        _FakeRecord("PTR", _build_ptr_record(test_num=11, test_name="OFFSET", result=0.05)),
        _FakeRecord("PRR", {"PART_ID": None, "SITE_NUM": 2, "PART_FLG": 0}),
    ]

    source = _make_source(tmp_path, "repeat.stdf", records)

    session_dir = tmp_path / "session"
    result = ingest.ingest_sources([source], session_dir)

    assert len(result.frame) == 3
    assert result.frame["file"].tolist() == ["repeat.stdf", "repeat.stdf", "repeat.stdf"]
    assert result.frame["test_name"].tolist() == ["GAIN", "GAIN", "OFFSET"]
    assert result.frame["part_status"].tolist() == ["PASS", "FAIL", "PASS"]
    assert result.frame["measurement_index"].tolist() == [1, 2, 1]
    assert result.frame["site"].tolist() == [1, 1, 2]

    # Device IDs should reuse the serial when present and fall back to SITE-based name when absent
    assert result.frame["device_id"].tolist()[0:2] == ["DUP", "DUP"]
    assert result.frame["device_id"].tolist()[2].startswith("SITE2_")


def test_ingest_sources_captures_site_descriptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "STDFReader", _FakeSTDFReader)

    sdr_payload = _build_sdr_payload(
        head_num=2,
        site_group=7,
        site_numbers=(1, 2),
        HAND_TYP="Handler-A",
        LOAD_ID="Loader-42",
    )
    records = [
        _FakeRecord("SDR", sdr_payload),
        _FakeRecord("SDR", sdr_payload),  # duplicate should be deduplicated
        _FakeRecord("PIR", {"PART_ID": "S1", "SITE_NUM": 1}),
        _FakeRecord("PTR", _build_ptr_record(result=1.23)),
        _FakeRecord("PRR", {"PART_ID": "S1", "SITE_NUM": 1, "PART_FLG": 0}),
    ]

    source = _make_source(tmp_path, "sdr.stdf", records)
    session_dir = tmp_path / "session"
    result = ingest.ingest_sources([source], session_dir)

    assert len(result.site_descriptions) == 1
    description = result.site_descriptions[0]
    assert description == SiteDescription(
        head_num=2,
        site_group=7,
        site_numbers=(1, 2),
        handler_type="Handler-A",
        load_id="Loader-42",
    )


def test_detect_site_support_positive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "STDFReader", _FakeSTDFReader)

    records = [
        _FakeRecord("PIR", {"PART_ID": "OK", "SITE_NUM": 2}),
        _FakeRecord("PRR", {"PART_ID": "OK", "SITE_NUM": 2, "PART_FLG": 0}),
    ]
    source = _make_source(tmp_path, "site.stdf", records)

    has_site, message = ingest.detect_site_support([source])
    assert has_site is True
    assert message is None


def test_detect_site_support_reports_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "STDFReader", _FakeSTDFReader)

    records = [
        _FakeRecord("PIR", {"PART_ID": "NO_SITE"}),
        _FakeRecord("PRR", {"PART_ID": "NO_SITE", "PART_FLG": 0}),
    ]
    source = _make_source(tmp_path, "nosite.stdf", records)

    has_site, message = ingest.detect_site_support([source])
    assert has_site is False
    assert isinstance(message, str)
    assert "SITE_NUM" in message


def test_detect_site_support_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "STDFReader", _FakeSTDFReader)

    site_records = [
        _FakeRecord("PIR", {"PART_ID": "A", "SITE_NUM": 5}),
        _FakeRecord("PRR", {"PART_ID": "A", "SITE_NUM": 5, "PART_FLG": 0}),
    ]
    nosite_records = [
        _FakeRecord("PIR", {"PART_ID": "B"}),
        _FakeRecord("PRR", {"PART_ID": "B", "PART_FLG": 0}),
    ]
    source_site = _make_source(tmp_path, "has_site.stdf", site_records)
    source_nosite = _make_source(tmp_path, "no_site.stdf", nosite_records)

    has_site, message = ingest.detect_site_support([source_nosite, source_site])
    assert has_site is True
    assert message is None


def test_detect_site_support_respects_sample_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _CountingReader(_FakeSTDFReader):
        def __iter__(self):
            # produce many PIRs without SITE_NUM, eventually include one
            for _ in range(1000):
                yield _FakeRecord("PIR", {"PART_ID": "X"})
            yield _FakeRecord("PIR", {"PART_ID": "Y", "SITE_NUM": 3})
            yield _FakeRecord("PRR", {"PART_ID": "Y", "SITE_NUM": 3, "PART_FLG": 0})

    monkeypatch.setattr(ingest, "STDFReader", _CountingReader)
    source = _make_source(tmp_path, "count.stdf", [])
    has_site, message = ingest.detect_site_support([source], sample_records=10)
    # Should not find site within first 10 records
    assert has_site is False
    assert isinstance(message, str)


def test_has_site_data_mixed_values() -> None:
    frame = pd.DataFrame({"site": [1, None, 3, float("nan")]})
    assert ingest.has_site_data(frame) is True


def test_has_site_data_numeric_strings() -> None:
    frame = pd.DataFrame({"site": ["1", " 2 ", None]})
    assert ingest.has_site_data(frame) is True


def test_has_site_data_all_null() -> None:
    frame = pd.DataFrame({"site": [None, float("nan"), None]})
    assert ingest.has_site_data(frame) is False


def test_has_site_data_missing_column() -> None:
    frame = pd.DataFrame({"file": ["a"]})
    assert ingest.has_site_data(frame) is False


def test_detect_site_support_from_sdr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "STDFReader", _FakeSTDFReader)

    records = [
        _FakeRecord("SDR", _build_sdr_payload(head_num=1, site_group=3, site_numbers=(1, 2))),
    ]
    source = _make_source(tmp_path, "sdr_only.stdf", records)

    has_site, message = ingest.detect_site_support([source])
    assert has_site is True
    assert isinstance(message, str) and "SDR" in message


def test_detect_site_support_skips_summary_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "STDFReader", _FakeSTDFReader)

    records = [
        _FakeRecord("PTR", {"HEAD_NUM": 255, "SITE_NUM": 0}),
        _FakeRecord("PRR", {"HEAD_NUM": 255, "SITE_NUM": 0, "PART_FLG": 0}),
    ]
    source = _make_source(tmp_path, "summary_only.stdf", records)

    has_site, message = ingest.detect_site_support([source])
    assert has_site is False
    assert isinstance(message, str)
