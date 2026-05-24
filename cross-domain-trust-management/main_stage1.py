from mininet.cli import CLI
import copy
import csv
from pathlib import Path

from scenario.domain_builder import DomainBuilder
from scenario.score_generator import ScoreGenerator
from scenario.label_assigner import LabelAssigner

from topo.net_adapter import MininetDomainAdapter

from trust.semantic_alignment import SemanticAligner
from trust.observation_aggregator import ObservationAggregator
from trust.simple_gap_detector import SimpleGapDetector
from trust.node_residual_attributor import NodeResidualAttributor, NodeResidualConfig

from attacks.network_attack_injector import NetworkAttackInjector
from simulation.interaction_simulator import InteractionSimulator
from simulation.event_dataset import load_interaction_events, save_interaction_events

from evaluation.metrics import ClassificationMetrics

ENABLE_CLI = False


DOMAIN_A_CONFIG = {
    "domain_id": "a",
    "num_nodes": 50,
    "score_dist": "beta",
    "score_params": {"alpha": 5, "beta": 2},   # optimistic domain
    "network_profile": {
        "bandwidth_mbps": 5,
        "delay_ms": 3,
        "jitter_ms": 1,
        "loss_pct": 0,
    },
}

DOMAIN_B_CONFIG = {
    "domain_id": "b",
    "num_nodes": 50,
    "score_dist": "beta",
    "score_params": {"alpha": 2, "beta": 5},   # conservative domain
    "network_profile": {
        "bandwidth_mbps": 5,
        "delay_ms": 8,
        "jitter_ms": 3,
        "loss_pct": 1,
    },
}

RANDOM_SEEDS = {
    "score": 42,
    "label": 123,
    "interaction": 456,
}

LABEL_CONFIG = {
    "malicious_ratio": 0.25,
    "malicious_attack_type": {
        "dishonest": 0.5,
        "on_off": 0.2,
        "bad_mouthing": 0.15,
        "sybil": 0.15,
    },
}

INTERACTION_CONFIG = {
    "num_epochs": 50,
    "interactions_per_ordered_pair": 4,
    "socket_timeout_secs": 3.0,
    "client_retries": 1,
    "payload_min_bytes": 64,
    "payload_max_bytes": 4096,
    "min_target_observations_per_epoch": 3,
    "min_source_interactions_per_epoch": 1,
    "baseline_ewma_alpha": 0.20,
    "min_baseline_samples": 5,
    "latency_tolerance_ratio": 0.50,
}

EVENT_DATASET_CONFIG = {
    "enabled": True,
    "path": "datasets/interactions_50epochs_semantic_delta_seed456.json",
    "generate_if_missing": True,
}

TRAINING_CONFIG = {
    "enabled": True,
    "history_window_epochs": 3,
    "min_eval_epoch": 5,
    "output_csv": "results/training/epoch_weight_metrics.csv",
    "summary_csv": "results/training/global_weight_search.csv",
    "best_csv": "results/training/best_global_weight_epoch_metrics.csv",
    "weight_grid_step": 0.10,
    "threshold_min": 0.10,
    "threshold_max": 0.30,
    "threshold_step": 0.01,
}

SEMANTIC_CONFIG = {
    "fusion_alpha": 0.65,
    "min_anchor_observation_count": 2,
}

ATTACK_CONFIG = {
    "enabled": True,
    "profiles": {
        "dishonest": {
            "delay_ms": 80,
            "jitter_ms": 10,
            "loss_pct": 15,
            "rate_kbit": 512,
        },
        "on_off": {
            "delay_ms": 150,
            "jitter_ms": 20,
            "loss_pct": 25,
            "rate_kbit": 256,
            "period": 4,
            "active_epochs": 2,
        },
        "bad_mouthing": {
            "delay_ms": 120,
            "jitter_ms": 15,
            "loss_pct": 10,
            "rate_kbit": 768,
            "duplicate_pct": 2,
        },
        "sybil": {
            "qdisc": "tbf",
            "rate_kbit": 384,
            "burst_kbit": 16,
            "latency_ms": 150,
        },
    },
}

DETECTOR_CONFIG = {
    "threshold": 0.30,
    "min_observation_count": 3,
    "default_prediction_when_missing": "normal",
    "history_window_epochs": 3,
    "visualization_score_bin_edges": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "weight_public_obs_gap": 0.45,
    "weight_temporal_gap": 0.20,
    "weight_local_distribution_gap": 0.20,
    "weight_domain_drift": 0.15,
}

NODE_ATTRIBUTION_CONFIG = {
    "semantic_weight": 0.30,
    "history_weight": 0.50,
    "bucket_weight": 0.20,
    "min_observation_count": 3,
    "cusum_slack": 0.03,
    "node_threshold": 0.18,
    "attribution_threshold": 0.18,
    "residual_weight": 0.70,
    "contribution_weight": 0.30,
    "blackhole_observed_threshold": 0.10,
    "contribution_normalizer": 0.10,
    "output_dir": "results/node_residual_attribution",
}


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def flatten_unified_records(records_by_domain: dict) -> list:
    all_records = []
    for _, records in records_by_domain.items():
        all_records.extend(records)
    return all_records


def build_semantic_score_index(records_by_domain: dict) -> dict:
    index = {}
    for _, records in records_by_domain.items():
        for record in records:
            index[(record.src_domain, record.node_id)] = record.semantic_score
    return index


def print_domain_basic_info(domains):
    print("\n===== Domain Node Basic Info =====")
    for domain in domains:
        print(f"[Domain {domain.domain_id}]")
        print(
            f"{'node_id':<8} {'role':<14} {'ip':<15} "
            f"{'raw_score':<10} {'label':<10} {'attack_type':<12}"
        )

        ordered_ids = [domain.trust_manager_id, domain.gateway_id] + list(domain.node_ids)
        for node_id in ordered_ids:
            node = domain.nodes[node_id]
            raw_score = (
                f"{node.raw_score:.6f}" if node.raw_score is not None else "None"
            )
            print(
                f"{node.node_id:<8} "
                f"{node.role:<14} "
                f"{str(node.ip):<15} "
                f"{raw_score:<10} "
                f"{node.label:<10} "
                f"{node.attack_type:<12}"
            )


def print_unified_records_by_domain(records_by_domain: dict):
    for domain_id, records in records_by_domain.items():
        print(f"\n===== Unified Trust Records for Domain {domain_id.upper()} =====")
        print(
            f"{'node_id':<8} {'raw_score':<10} {'quantile':<10} "
            f"{'calib':<10} {'semantic':<10} "
            f"{'grade':<6} {'fresh':<8} {'conf':<8}"
        )

        for r in sorted(records, key=lambda x: x.node_id):
            calibrated_score = (
                f"{r.calibrated_score:.6f}" if r.calibrated_score is not None else "None"
            )
            print(
                f"{r.node_id:<8} "
                f"{r.raw_score:<10.6f} "
                f"{r.quantile:<10.6f} "
                f"{calibrated_score:<10} "
                f"{r.semantic_score:<10.6f} "
                f"{r.grade:<6} "
                f"{r.freshness:<8.6f} "
                f"{r.confidence:<8.6f}"
            )


def print_domain_summaries(aligner, records_by_domain: dict, domains=None):
    print("\n===== Domain Summaries =====")
    domain_index = {domain.domain_id: domain for domain in domains or []}
    for domain_id, records in records_by_domain.items():
        summary = aligner.build_domain_summary(
            records,
            domain=domain_index.get(domain_id),
        )
        print(f"[Domain {domain_id}] {summary}")


def run_basic_connectivity_test(net):
    print("\n===== Basic Connectivity Test =====")
    # use IP addresses instead of hostnames to avoid name resolution issues
    print(net.get("a1").cmd("ping -c 2 10.0.2.11"))
    print(net.get("a_tm").cmd("ping -c 2 10.0.2.10"))


def generate_weight_candidates(step: float = 0.10) -> list[dict]:
    if step <= 0 or step > 1:
        raise ValueError("weight_grid_step must be in (0, 1]")

    scale = round(1.0 / step)
    candidates = []
    seen = set()

    for public_units in range(scale + 1):
        for temporal_units in range(scale + 1 - public_units):
            for local_units in range(scale + 1 - public_units - temporal_units):
                drift_units = scale - public_units - temporal_units - local_units
                weights = (
                    round(public_units / scale, 6),
                    round(temporal_units / scale, 6),
                    round(local_units / scale, 6),
                    round(drift_units / scale, 6),
                )
                if weights in seen:
                    continue
                seen.add(weights)
                candidates.append(
                    {
                        "weight_public_obs_gap": weights[0],
                        "weight_temporal_gap": weights[1],
                        "weight_local_distribution_gap": weights[2],
                        "weight_domain_drift": weights[3],
                    }
                )

    return candidates


def generate_threshold_candidates(
    min_value: float,
    max_value: float,
    step: float,
) -> list[float]:
    if step <= 0:
        raise ValueError("threshold_step must be positive")
    if min_value > max_value:
        raise ValueError("threshold_min must be <= threshold_max")

    values = []
    current = min_value
    while current <= max_value + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def run_epoch_weight_training(
    domains,
    events,
    output_csv: str,
) -> list[dict]:
    print("\n===== Global Weight Search over Rolling Epochs =====")
    training_domains = copy.deepcopy(domains)
    aggregator = ObservationAggregator()
    aligner = SemanticAligner()
    metrics_engine = ClassificationMetrics(positive_label="malicious")
    weight_candidates = generate_weight_candidates(TRAINING_CONFIG["weight_grid_step"])
    threshold_candidates = generate_threshold_candidates(
        min_value=TRAINING_CONFIG["threshold_min"],
        max_value=TRAINING_CONFIG["threshold_max"],
        step=TRAINING_CONFIG["threshold_step"],
    )

    max_epoch = max((event.epoch for event in events), default=0)
    min_epoch = max(
        TRAINING_CONFIG["min_eval_epoch"],
        TRAINING_CONFIG["history_window_epochs"] + 2,
    )
    epoch_contexts = []

    for epoch in range(min_epoch, max_epoch + 1):
        calibration_obs_records = aggregator.aggregate(
            events,
            mode="global",
            epoch_from=1,
            epoch_to=epoch - 1,
        )
        calibrated_records_by_domain = aligner.calibrate_many_domains(
            domains=training_domains,
            observation_records=calibration_obs_records,
            epoch=epoch,
            fusion_alpha=SEMANTIC_CONFIG["fusion_alpha"],
            min_anchor_observation_count=SEMANTIC_CONFIG["min_anchor_observation_count"],
        )
        all_public_records = flatten_unified_records(calibrated_records_by_domain)

        current_obs_records = aggregator.aggregate(
            events,
            mode="global",
            epoch_from=epoch,
            epoch_to=epoch,
        )
        historical_obs_records = aggregator.aggregate(
            events,
            mode="global",
            epoch_from=max(1, epoch - TRAINING_CONFIG["history_window_epochs"]),
            epoch_to=epoch - 1,
        )

        epoch_contexts.append(
            {
                "epoch": epoch,
                "domains": training_domains,
                "unified_records": all_public_records,
                "current_observation_records": current_obs_records,
                "historical_observation_records": historical_obs_records,
            }
        )

    summary_rows = []
    best_summary = None
    best_epoch_rows = []

    for weights in weight_candidates:
        for threshold in threshold_candidates:
            epoch_rows = []
            total_tp = total_fp = total_tn = total_fn = 0
            macro_precision = macro_recall = macro_f1 = macro_accuracy = 0.0

            detector = SimpleGapDetector(
                threshold=threshold,
                min_observation_count=DETECTOR_CONFIG["min_observation_count"],
                default_prediction_when_missing=DETECTOR_CONFIG["default_prediction_when_missing"],
                **weights,
            )

            for context in epoch_contexts:
                detection_results = detector.detect(
                    domains=context["domains"],
                    unified_records=context["unified_records"],
                    current_observation_records=context["current_observation_records"],
                    historical_observation_records=context["historical_observation_records"],
                )
                metrics = metrics_engine.evaluate(detection_results)
                total_tp += metrics["tp"]
                total_fp += metrics["fp"]
                total_tn += metrics["tn"]
                total_fn += metrics["fn"]
                macro_precision += metrics["precision"]
                macro_recall += metrics["recall"]
                macro_f1 += metrics["f1"]
                macro_accuracy += metrics["accuracy"]
                epoch_rows.append(
                    {
                        "epoch": context["epoch"],
                        "threshold": threshold,
                        **weights,
                        **metrics,
                    }
                )

            epoch_count = max(1, len(epoch_contexts))
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            micro_f1 = (
                2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0
                else 0.0
            )
            micro_accuracy = (
                (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)
                if (total_tp + total_fp + total_tn + total_fn) > 0
                else 0.0
            )

            summary = {
                "threshold": threshold,
                **weights,
                "epoch_count": len(epoch_contexts),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_tn": total_tn,
                "total_fn": total_fn,
                "macro_precision": round(macro_precision / epoch_count, 6),
                "macro_recall": round(macro_recall / epoch_count, 6),
                "macro_f1": round(macro_f1 / epoch_count, 6),
                "macro_accuracy": round(macro_accuracy / epoch_count, 6),
                "micro_precision": round(micro_precision, 6),
                "micro_recall": round(micro_recall, 6),
                "micro_f1": round(micro_f1, 6),
                "micro_accuracy": round(micro_accuracy, 6),
            }
            summary_rows.append(summary)

            current_key = (
                summary["macro_f1"],
                summary["micro_f1"],
                summary["macro_recall"],
                summary["macro_precision"],
                summary["macro_accuracy"],
            )
            best_key = (
                best_summary["macro_f1"],
                best_summary["micro_f1"],
                best_summary["macro_recall"],
                best_summary["macro_precision"],
                best_summary["macro_accuracy"],
            ) if best_summary is not None else None

            if best_summary is None or current_key > best_key:
                best_summary = summary
                best_epoch_rows = epoch_rows

    rows = best_epoch_rows
    for row in rows:
        print(
            f"epoch={row['epoch']:<3} "
            f"F1={row['f1']:.6f} "
            f"P={row['precision']:.6f} "
            f"R={row['recall']:.6f} "
            f"Acc={row['accuracy']:.6f} "
            f"threshold={row['threshold']}, "
            f"global_weights=({row['weight_public_obs_gap']}, "
            f"{row['weight_temporal_gap']}, "
            f"{row['weight_local_distribution_gap']}, "
            f"{row['weight_domain_drift']})"
        )

    print(
        "\nBest global detector config: "
        f"threshold={best_summary['threshold']}, "
        f"weights=({best_summary['weight_public_obs_gap']}, "
        f"{best_summary['weight_temporal_gap']}, "
        f"{best_summary['weight_local_distribution_gap']}, "
        f"{best_summary['weight_domain_drift']}) "
        f"macro_F1={best_summary['macro_f1']:.6f}, "
        f"micro_F1={best_summary['micro_f1']:.6f}"
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "threshold",
        "weight_public_obs_gap",
        "weight_temporal_gap",
        "weight_local_distribution_gap",
        "weight_domain_drift",
        "num_samples",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = Path(TRAINING_CONFIG["summary_csv"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_fieldnames = [
        "threshold",
        "weight_public_obs_gap",
        "weight_temporal_gap",
        "weight_local_distribution_gap",
        "weight_domain_drift",
        "epoch_count",
        "total_tp",
        "total_fp",
        "total_tn",
        "total_fn",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_accuracy",
    ]
    summary_rows.sort(
        key=lambda row: (
            row["macro_f1"],
            row["micro_f1"],
            row["macro_recall"],
            row["macro_precision"],
            row["macro_accuracy"],
        ),
        reverse=True,
    )
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    best_path = Path(TRAINING_CONFIG["best_csv"])
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Best global epoch metrics exported to: {output_path}")
    print(f"All global detector summaries exported to: {summary_path}")
    print(f"Best global epoch metrics copy exported to: {best_path}")
    return rows


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def main():
    # 1) Build abstract domains
    builder = DomainBuilder()

    domain_a = builder.build_domain(
        domain_id=DOMAIN_A_CONFIG["domain_id"],
        num_nodes=DOMAIN_A_CONFIG["num_nodes"],
        score_dist=DOMAIN_A_CONFIG["score_dist"],
        score_params=DOMAIN_A_CONFIG["score_params"],
        network_profile=DOMAIN_A_CONFIG["network_profile"],
    )

    domain_b = builder.build_domain(
        domain_id=DOMAIN_B_CONFIG["domain_id"],
        num_nodes=DOMAIN_B_CONFIG["num_nodes"],
        score_dist=DOMAIN_B_CONFIG["score_dist"],
        score_params=DOMAIN_B_CONFIG["score_params"],
        network_profile=DOMAIN_B_CONFIG["network_profile"],
    )

    domains = [domain_a, domain_b]

    # 2) Generate domain-local raw scores
    generator = ScoreGenerator(seed=RANDOM_SEEDS["score"])
    generator.populate_many_domains(domains, current_epoch=1)

    # 3) Assign labels (ground truth)
    assigner = LabelAssigner(seed=RANDOM_SEEDS["label"])
    assigner.assign_many_domains(
        domains=domains,
        malicious_ratio=LABEL_CONFIG["malicious_ratio"],
        malicious_attack_type=LABEL_CONFIG["malicious_attack_type"],
        include_gateway=False,
        include_trust_manager=False,
        overwrite_existing=True,
    )
    assigner.print_label_summary(domains)

    net = None
    simulator = None
    attack_injector = None
    events = None

    try:
        dataset_path = Path(EVENT_DATASET_CONFIG["path"])
        if EVENT_DATASET_CONFIG["enabled"] and dataset_path.exists():
            print(f"\n===== Loading Cached Interaction Dataset =====")
            print(f"Dataset: {dataset_path}")
            events = load_interaction_events(dataset_path)
        else:
            if EVENT_DATASET_CONFIG["enabled"] and not EVENT_DATASET_CONFIG["generate_if_missing"]:
                raise FileNotFoundError(f"Interaction dataset not found: {dataset_path}")

            # Build/start Mininet only when interaction events must be generated.
            adapter = MininetDomainAdapter(domains=domains)
            adapter.print_plan()

            net = adapter.start_network()

            # Connectivity test
            run_basic_connectivity_test(net)

            if ATTACK_CONFIG["enabled"]:
                attack_injector = NetworkAttackInjector(
                    attack_profiles=ATTACK_CONFIG["profiles"]
                )

        # 6) Print node basic info after score + label assignment
        print_domain_basic_info(domains)

        # 7) Stage-1 semantic alignment
        aligner = SemanticAligner()
        records_by_domain = aligner.unify_many_domains(domains, epoch=1)

        print("\n===== Baseline Semantic Alignment =====")
        print_unified_records_by_domain(records_by_domain)
        print_domain_summaries(aligner, records_by_domain, domains=domains)

        # Optional example: project one foreign quantile into the other domain scale
        if records_by_domain["a"]:
            sample_record = records_by_domain["a"][0]
            projected_score_in_b = aligner.project_quantile_to_local_score(
                sample_record.quantile, domain_b
            )
            print("\n===== Cross-domain Interpretation Example =====")
            print(
                f"Node {sample_record.node_id} in Domain A has "
                f"raw_score={sample_record.raw_score:.6f}, "
                f"quantile={sample_record.quantile:.6f}, grade={sample_record.grade}. "
                f"If interpreted in Domain B's score scale, "
                f"its projected local score is {projected_score_in_b:.6f}."
            )

        # 8) Simulate cross-domain interactions when no cached dataset is available.
        if events is None:
            simulator = InteractionSimulator(
                seed=RANDOM_SEEDS["interaction"],
                socket_timeout_secs=INTERACTION_CONFIG["socket_timeout_secs"],
                client_retries=INTERACTION_CONFIG["client_retries"],
                payload_min_bytes=INTERACTION_CONFIG["payload_min_bytes"],
                payload_max_bytes=INTERACTION_CONFIG["payload_max_bytes"],
                min_target_observations_per_epoch=INTERACTION_CONFIG["min_target_observations_per_epoch"],
                min_source_interactions_per_epoch=INTERACTION_CONFIG["min_source_interactions_per_epoch"],
                baseline_ewma_alpha=INTERACTION_CONFIG["baseline_ewma_alpha"],
                min_baseline_samples=INTERACTION_CONFIG["min_baseline_samples"],
                latency_tolerance_ratio=INTERACTION_CONFIG["latency_tolerance_ratio"],
            )
            semantic_score_index = build_semantic_score_index(records_by_domain)
            events = simulator.simulate_many_epochs(
                net=net,
                domains=domains,
                num_epochs=INTERACTION_CONFIG["num_epochs"],
                interactions_per_ordered_pair=INTERACTION_CONFIG["interactions_per_ordered_pair"],
                start_epoch=1,
                semantic_score_index=semantic_score_index,
                epoch_callback=(
                    (lambda epoch: attack_injector.apply_epoch(net, domains, epoch))
                    if attack_injector is not None
                    else None
                ),
            )
            if EVENT_DATASET_CONFIG["enabled"]:
                saved_path = save_interaction_events(
                    events,
                    dataset_path,
                    metadata={
                        "num_epochs": INTERACTION_CONFIG["num_epochs"],
                        "interaction_seed": RANDOM_SEEDS["interaction"],
                        "score_seed": RANDOM_SEEDS["score"],
                        "label_seed": RANDOM_SEEDS["label"],
                        "observation_model": "semantic_delta_v1",
                    },
                )
                print(f"\nInteraction dataset saved to: {saved_path}")

        event_printer = simulator or InteractionSimulator()
        event_printer.print_event_sample(events, limit=15)

        # 9) Aggregate external observations
        aggregator = ObservationAggregator()
        global_obs_records = aggregator.aggregate(events, mode="global")
        aggregator.print_records(global_obs_records)

        current_epoch = max((event.epoch for event in events), default=INTERACTION_CONFIG["num_epochs"])
        current_obs_records = aggregator.aggregate(
            events,
            mode="global",
            epoch_from=current_epoch,
            epoch_to=current_epoch,
        )
        history_epoch_from = max(1, current_epoch - DETECTOR_CONFIG["history_window_epochs"])
        history_epoch_to = current_epoch - 1
        historical_obs_records = (
            aggregator.aggregate(
                events,
                mode="global",
                epoch_from=history_epoch_from,
                epoch_to=history_epoch_to,
            )
            if history_epoch_to >= history_epoch_from
            else []
        )

        # 10) Anchor-calibrated semantic alignment using external observations
        calibrated_records_by_domain = aligner.calibrate_many_domains(
            domains=domains,
            observation_records=global_obs_records,
            epoch=INTERACTION_CONFIG["num_epochs"],
            fusion_alpha=SEMANTIC_CONFIG["fusion_alpha"],
            min_anchor_observation_count=SEMANTIC_CONFIG["min_anchor_observation_count"],
        )
        print("\n===== Anchor-Calibrated Semantic Alignment =====")
        aligner.print_anchor_states(domains, limit=12)
        print_unified_records_by_domain(calibrated_records_by_domain)
        print_domain_summaries(aligner, calibrated_records_by_domain, domains=domains)

        all_public_records = flatten_unified_records(calibrated_records_by_domain)

        # 11) Simple gap-based detection
        detector = SimpleGapDetector(
            threshold=DETECTOR_CONFIG["threshold"],
            min_observation_count=DETECTOR_CONFIG["min_observation_count"],
            default_prediction_when_missing=DETECTOR_CONFIG["default_prediction_when_missing"],
            weight_public_obs_gap=DETECTOR_CONFIG["weight_public_obs_gap"],
            weight_temporal_gap=DETECTOR_CONFIG["weight_temporal_gap"],
            weight_local_distribution_gap=DETECTOR_CONFIG["weight_local_distribution_gap"],
            weight_domain_drift=DETECTOR_CONFIG["weight_domain_drift"],
        )

        detection_results = detector.detect(
            domains=domains,
            unified_records=all_public_records,
            current_observation_records=current_obs_records,
            historical_observation_records=historical_obs_records,
        )

        detector.print_results(
            results=detection_results,
            unified_records=all_public_records,
            current_observation_records=current_obs_records,
            historical_observation_records=historical_obs_records,
            domains=domains,
        )

        distribution_report = detector.build_distribution_report(
            domains=domains,
            unified_records=all_public_records,
            current_observation_records=current_obs_records,
            historical_observation_records=historical_obs_records,
        )
        detector.print_distribution_report(distribution_report)
        exported_distribution_paths = detector.export_distribution_report(
            report=distribution_report,
            output_dir="results/pqh_distributions",
        )

        visualization_detector = SimpleGapDetector(
            threshold=DETECTOR_CONFIG["threshold"],
            min_observation_count=DETECTOR_CONFIG["min_observation_count"],
            default_prediction_when_missing=DETECTOR_CONFIG["default_prediction_when_missing"],
            score_bin_edges=DETECTOR_CONFIG["visualization_score_bin_edges"],
            weight_public_obs_gap=DETECTOR_CONFIG["weight_public_obs_gap"],
            weight_temporal_gap=DETECTOR_CONFIG["weight_temporal_gap"],
            weight_local_distribution_gap=DETECTOR_CONFIG["weight_local_distribution_gap"],
            weight_domain_drift=DETECTOR_CONFIG["weight_domain_drift"],
        )
        visualization_distribution_report = visualization_detector.build_distribution_report(
            domains=domains,
            unified_records=all_public_records,
            current_observation_records=current_obs_records,
            historical_observation_records=historical_obs_records,
        )
        exported_visualization_paths = visualization_detector.export_distribution_report(
            report=visualization_distribution_report,
            output_dir="results/pqh_distributions",
            filename_prefix="pqh_distribution_bucket_0_1",
        )
        print("\n===== Distribution Data Exported =====")
        print(f"JSON: {exported_distribution_paths['json']}")
        print(f"CSV : {exported_distribution_paths['csv']}")
        print(f"0.1 bucket JSON: {exported_visualization_paths['json']}")
        print(f"0.1 bucket CSV : {exported_visualization_paths['csv']}")

        # 12) Node-level residual attribution:
        # explain which nodes contribute to the domain-level P/Q/H drift.
        node_attributor = NodeResidualAttributor(
            NodeResidualConfig(
                semantic_weight=NODE_ATTRIBUTION_CONFIG["semantic_weight"],
                history_weight=NODE_ATTRIBUTION_CONFIG["history_weight"],
                bucket_weight=NODE_ATTRIBUTION_CONFIG["bucket_weight"],
                min_observation_count=NODE_ATTRIBUTION_CONFIG["min_observation_count"],
                cusum_slack=NODE_ATTRIBUTION_CONFIG["cusum_slack"],
                node_threshold=NODE_ATTRIBUTION_CONFIG["node_threshold"],
                attribution_threshold=NODE_ATTRIBUTION_CONFIG["attribution_threshold"],
                residual_weight=NODE_ATTRIBUTION_CONFIG["residual_weight"],
                contribution_weight=NODE_ATTRIBUTION_CONFIG["contribution_weight"],
                blackhole_observed_threshold=NODE_ATTRIBUTION_CONFIG["blackhole_observed_threshold"],
                contribution_normalizer=NODE_ATTRIBUTION_CONFIG["contribution_normalizer"],
                score_bin_edges=DETECTOR_CONFIG["visualization_score_bin_edges"],
            )
        )
        node_attribution_results = node_attributor.analyze(
            domains=domains,
            unified_records=all_public_records,
            current_observation_records=current_obs_records,
            historical_observation_records=historical_obs_records,
        )
        node_attributor.print_results(node_attribution_results, limit=20)
        exported_node_attribution_paths = node_attributor.export_results(
            results=node_attribution_results,
            output_dir=NODE_ATTRIBUTION_CONFIG["output_dir"],
        )
        print("\n===== Node Attribution Data Exported =====")
        print(f"JSON: {exported_node_attribution_paths['json']}")
        print(f"CSV : {exported_node_attribution_paths['csv']}")

        # 13) Metrics
        metrics_engine = ClassificationMetrics(positive_label="malicious")
        metrics = metrics_engine.evaluate(detection_results)
        metrics_engine.print_report(metrics)

        # 14) Rolling per-epoch training over cached/generated events
        if TRAINING_CONFIG["enabled"]:
            run_epoch_weight_training(
                domains=domains,
                events=events,
                output_csv=TRAINING_CONFIG["output_csv"],
            )

        # 15) Optional interactive CLI
        if ENABLE_CLI and net is not None:
            print("\n===== Entering Mininet CLI =====")
            CLI(net)

    finally:
        if attack_injector is not None and net is not None:
            attack_injector.clear(net, domains)
        if simulator is not None and net is not None:
            simulator.cleanup_services(net, domains)
        if net is not None:
            net.stop()


if __name__ == "__main__":
    main()
