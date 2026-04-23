from pathlib import Path

from scripts import merge_aco_candidates as merge_module
from scripts import plot_opportunity_network as plot_module


def test_plot_network_prefers_run_local_input(monkeypatch, tmp_path):
    run_local = tmp_path / "outputs" / "runs" / "run_x" / "03_merged_rag" / "aco_merged_top30_enriched.json"
    global_path = tmp_path / "data" / "processed" / "rag" / "aco_merged_top30_enriched.json"

    monkeypatch.setattr(plot_module, "get_run_merged_rag_enriched_json", lambda run_id: run_local)
    monkeypatch.setattr(plot_module, "RAG_ENRICHED_JSON", global_path)

    assert plot_module._resolve_input_path(run_id="run_x") == run_local
    explicit = tmp_path / "custom.json"
    assert plot_module._resolve_input_path(input_path=explicit, run_id="run_x") == explicit
    assert plot_module._resolve_input_path() == global_path


def test_merge_feature_names_fall_back_to_registry(monkeypatch):
    fake_registry = {
        "New_n": object(),
        "New_e": object(),
        "Constraint": object(),
    }
    monkeypatch.setattr(merge_module, "FEATURE_REGISTRY", fake_registry)

    assert merge_module._resolve_feature_names({"Eigen": 1.0}) == ["Eigen"]
    assert merge_module._resolve_feature_names({}) == list(fake_registry.keys())
