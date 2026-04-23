#!/usr/bin/env python3
"""
Step4 输入优先级测试。
"""

from pathlib import Path

from scripts import step4_rag_report as step4_module


def test_run_merged_rag_input_preferred(monkeypatch, tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / "test_run"
    merged_dir = run_dir / "03_merged_rag"
    aco_dir = run_dir / "03_aco"
    merged_dir.mkdir(parents=True)
    aco_dir.mkdir(parents=True)

    merged_path = merged_dir / "aco_merged_top30_enriched.json"
    aco_path = aco_dir / "aco_topk_enriched.json"
    global_path = tmp_path / "data" / "processed" / "rag" / "aco_merged_top30_enriched.json"
    global_path.parent.mkdir(parents=True)

    merged_path.write_text("merged", encoding="utf-8")
    aco_path.write_text("aco", encoding="utf-8")
    global_path.write_text("global", encoding="utf-8")

    monkeypatch.setattr(step4_module, "get_run_dir", lambda run_id: run_dir)
    monkeypatch.setattr(step4_module, "RAG_ENRICHED_JSON", global_path)

    resolved = step4_module._resolve_input_json(run_id="test_run")
    assert resolved == merged_path


def test_global_rag_fallback_when_run_artifacts_missing(monkeypatch, tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / "test_run"
    run_dir.mkdir(parents=True)
    global_path = tmp_path / "data" / "processed" / "rag" / "aco_merged_top30_enriched.json"
    global_path.parent.mkdir(parents=True)
    global_path.write_text("global", encoding="utf-8")

    monkeypatch.setattr(step4_module, "get_run_dir", lambda run_id: run_dir)
    monkeypatch.setattr(step4_module, "RAG_ENRICHED_JSON", global_path)

    resolved = step4_module._resolve_input_json(run_id="test_run")
    assert resolved == global_path
