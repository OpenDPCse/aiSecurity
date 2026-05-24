import json
import math
import random
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

from core.entities import Domain, InteractionEvent


@dataclass
class CrossDomainBaseline:
    latency_ms: float | None = None
    throughput_kbps: float | None = None
    success_rate: float = 1.0
    retry_rate: float = 0.0
    sample_count: int = 0

    def snapshot(self) -> dict:
        return {
            "latency_ms": self.latency_ms,
            "throughput_kbps": self.throughput_kbps,
            "success_rate": round(self.success_rate, 6),
            "retry_rate": round(self.retry_rate, 6),
            "sample_count": self.sample_count,
        }


class InteractionSimulator:
    """
    Persistent TCP-server + random-payload client simulator for cross-domain communication.
    """

    def __init__(
        self,
        seed: int | None = None,
        probe_port: int = 18080,
        socket_timeout_secs: float = 3.0,
        client_retries: int = 1,
        payload_min_bytes: int = 64,
        payload_max_bytes: int = 4096,
        min_target_observations_per_epoch: int = 1,
        min_source_interactions_per_epoch: int = 1,
        baseline_ewma_alpha: float = 0.20,
        min_baseline_samples: int = 5,
        latency_tolerance_ratio: float = 0.50,
        observation_reward: float = 0.05,
        observation_penalty: float = 0.10,
        score_bin_edges: list[float] | None = None,
    ):
        self.rng = random.Random(seed)
        self.probe_port = probe_port
        self.socket_timeout_secs = socket_timeout_secs
        self.client_retries = client_retries
        self.payload_min_bytes = payload_min_bytes
        self.payload_max_bytes = payload_max_bytes
        self.min_target_observations_per_epoch = min_target_observations_per_epoch
        self.min_source_interactions_per_epoch = min_source_interactions_per_epoch
        self.baseline_ewma_alpha = baseline_ewma_alpha
        self.min_baseline_samples = min_baseline_samples
        self.latency_tolerance_ratio = latency_tolerance_ratio
        self.observation_reward = observation_reward
        self.observation_penalty = observation_penalty
        self.score_bin_edges = score_bin_edges or [0.2, 0.4, 0.6, 0.8]
        self._event_counter = 0
        self._servers_started = False
        self._server_script_path = str((Path(__file__).resolve().parent / "tcp_echo_server.py"))
        self._client_script_path = str((Path(__file__).resolve().parent / "tcp_probe_client.py"))
        self._baselines: dict[tuple[str, str], CrossDomainBaseline] = {}
        self._observation_state: dict[tuple[str, str, str], float] = {}

    def _make_event(
        self,
        epoch: int,
        src_domain: str,
        src_node: str,
        dst_domain: str,
        dst_node: str,
        success: int,
        **kwargs,
    ) -> InteractionEvent:
        self._event_counter += 1
        observed_score = kwargs.pop("observed_score", float(success))
        return InteractionEvent(
            event_id=f"evt_{epoch}_{self._event_counter}",
            epoch=epoch,
            src_domain=src_domain,
            src_node=src_node,
            dst_domain=dst_domain,
            dst_node=dst_node,
            success=success,
            observed_score=observed_score,
            **kwargs,
        )

    def _baseline_key(self, src_domain: str, dst_domain: str) -> tuple[str, str]:
        return (src_domain, dst_domain)

    def _get_baseline(self, src_domain: str, dst_domain: str) -> CrossDomainBaseline:
        key = self._baseline_key(src_domain, dst_domain)
        if key not in self._baselines:
            self._baselines[key] = CrossDomainBaseline()
        return self._baselines[key]

    def _ewma(self, old_value: float | None, new_value: float, alpha: float | None = None) -> float:
        alpha = self.baseline_ewma_alpha if alpha is None else alpha
        if old_value is None:
            return new_value
        return (1.0 - alpha) * old_value + alpha * new_value

    def _update_baseline(
        self,
        src_domain: str,
        dst_domain: str,
        success: int,
        probe_result: dict,
    ) -> None:
        baseline = self._get_baseline(src_domain, dst_domain)
        payload_ok = bool(probe_result.get("payload_ok"))
        retries = int(probe_result.get("retries", 0))
        retry_unit = 1.0 if retries > 0 else 0.0

        baseline.success_rate = self._ewma(baseline.success_rate, float(success))
        baseline.retry_rate = self._ewma(baseline.retry_rate, retry_unit)

        if success != 1 or not payload_ok:
            return

        latency_ms = probe_result.get("latency_ms")
        throughput_kbps = probe_result.get("throughput_kbps")

        if latency_ms is not None:
            if baseline.latency_ms is not None:
                max_accepted_latency = baseline.latency_ms * 3.0
                latency_ms = min(latency_ms, max_accepted_latency)
            baseline.latency_ms = self._ewma(baseline.latency_ms, float(latency_ms))

        if throughput_kbps is not None:
            if baseline.throughput_kbps is not None:
                min_accepted_throughput = baseline.throughput_kbps * 0.25
                throughput_kbps = max(throughput_kbps, min_accepted_throughput)
            baseline.throughput_kbps = self._ewma(
                baseline.throughput_kbps,
                float(throughput_kbps),
            )

        baseline.sample_count += 1

    def _relative_latency_score(
        self,
        latency_ms: float | None,
        baseline: CrossDomainBaseline,
    ) -> float:
        if latency_ms is None:
            return 0.0
        if baseline.sample_count < self.min_baseline_samples or baseline.latency_ms is None:
            return 0.75

        if latency_ms <= baseline.latency_ms:
            return 1.0

        tolerance = max(1e-6, baseline.latency_ms * self.latency_tolerance_ratio)
        excess = latency_ms - baseline.latency_ms
        return math.exp(-excess / tolerance)

    def _relative_throughput_score(
        self,
        throughput_kbps: float | None,
        baseline: CrossDomainBaseline,
    ) -> float:
        if throughput_kbps is None:
            return 0.0
        if baseline.sample_count < self.min_baseline_samples or baseline.throughput_kbps is None:
            return 0.75

        if baseline.throughput_kbps <= 0:
            return 0.0

        return min(1.0, throughput_kbps / baseline.throughput_kbps)

    def _reliability_score(self, probe_result: dict) -> float:
        if probe_result.get("timeout_occurred"):
            return 0.0

        retries = int(probe_result.get("retries", 0))
        max_attempts = max(1, self.client_retries + 1)
        retry_penalty = min(1.0, retries / max_attempts)
        return max(0.0, 1.0 - retry_penalty)

    def _score_to_bucket(self, score: float) -> int:
        for idx, edge in enumerate(self.score_bin_edges):
            if score < edge:
                return idx
        return len(self.score_bin_edges)

    def _bucket_upper_bound(self, score: float) -> float:
        for edge in self.score_bin_edges:
            if score < edge:
                return edge
        return 1.0

    def _observation_key(
        self,
        observer_domain: str,
        target_domain: str,
        target_node: str,
    ) -> tuple[str, str, str]:
        return (observer_domain, target_domain, target_node)

    def _public_semantic_score(
        self,
        semantic_score_index: dict[tuple[str, str], float] | None,
        target_domain: str,
        target_node: str,
    ) -> float:
        if semantic_score_index is None:
            return 0.5
        return semantic_score_index.get((target_domain, target_node), 0.5)

    def _capability_threshold(self, public_semantic_score: float) -> float:
        bucket = self._score_to_bucket(public_semantic_score)
        max_bucket = max(1, len(self.score_bin_edges))
        return 0.55 + 0.30 * (bucket / max_bucket)

    def _assess_interaction_compliance(
        self,
        success: int,
        probe_result: dict,
        baseline: CrossDomainBaseline,
        public_semantic_score: float,
    ) -> tuple[bool, dict]:
        components = self._build_observation_components(
            success=success,
            probe_result=probe_result,
            baseline=baseline,
        )
        components["capability_threshold"] = round(
            self._capability_threshold(public_semantic_score),
            6,
        )
        components["public_semantic_score"] = round(public_semantic_score, 6)
        components["public_semantic_bucket"] = self._score_to_bucket(public_semantic_score)

        if success != 1 or not probe_result.get("payload_ok"):
            components["compliance_evidence"] = 0.0
            return False, components

        if probe_result.get("timeout_occurred"):
            components["compliance_evidence"] = 0.0
            return False, components

        compliance_evidence = (
            0.40 * components["success_score"]
            + 0.25 * components["latency_score"]
            + 0.15 * components["throughput_score"]
            + 0.15 * components["reliability_score"]
            + 0.05 * components["integrity_score"]
        )
        components["compliance_evidence"] = round(compliance_evidence, 6)

        return compliance_evidence >= components["capability_threshold"], components

    def _update_observed_semantic_score(
        self,
        observer_domain: str,
        target_domain: str,
        target_node: str,
        public_semantic_score: float,
        compliant: bool,
    ) -> tuple[float, float]:
        key = self._observation_key(observer_domain, target_domain, target_node)
        previous_score = self._observation_state.get(key, public_semantic_score)

        if compliant:
            updated_score = min(
                self._bucket_upper_bound(public_semantic_score),
                previous_score + self.observation_reward,
            )
        else:
            updated_score = max(0.0, previous_score - self.observation_penalty)

        updated_score = round(updated_score, 6)
        self._observation_state[key] = updated_score
        return round(previous_score, 6), updated_score

    def _build_observation_components(
        self,
        success: int,
        probe_result: dict,
        baseline: CrossDomainBaseline,
    ) -> dict:
        return {
            "success_score": round(float(success), 6),
            "latency_score": round(
                self._relative_latency_score(probe_result.get("latency_ms"), baseline),
                6,
            ),
            "throughput_score": round(
                self._relative_throughput_score(
                    probe_result.get("throughput_kbps"),
                    baseline,
                ),
                6,
            ),
            "reliability_score": round(self._reliability_score(probe_result), 6),
            "integrity_score": 1.0 if probe_result.get("payload_ok") else 0.0,
        }

    def _random_payload_size(self) -> int:
        return self.rng.randint(self.payload_min_bytes, self.payload_max_bytes)

    def _start_tcp_server(self, host, node) -> None:
        host.cmd(
            f"pkill -f 'tcp_echo_server.py --bind {node.ip} --port {self.probe_port}' >/dev/null 2>&1"
        )

        cmd = (
            f"nohup python3 -u {self._server_script_path} "
            f"--bind {node.ip} --port {self.probe_port} "
            f">/tmp/{node.node_id}_tcp_server.log 2>&1 &"
        )
        host.cmd(cmd)

    def _ensure_servers_started(self, net, domains: list[Domain]) -> None:
        if self._servers_started:
            return

        for domain in domains:
            for node_id in domain.node_ids:
                node = domain.nodes[node_id]
                host = net.get(node_id)
                self._start_tcp_server(host, node)

        time.sleep(0.5)
        self._servers_started = True

    def cleanup_services(self, net, domains: list[Domain]) -> None:
        for domain in domains:
            for node_id in domain.node_ids:
                node = domain.nodes[node_id]
                host = net.get(node_id)
                host.cmd(
                    f"pkill -f 'tcp_echo_server.py --bind {node.ip} --port {self.probe_port}' >/dev/null 2>&1"
                )
        self._servers_started = False

    def _probe_tcp(self, src_host, dst_ip: str, payload_size: int) -> dict:
        cmd = (
            f"python3 {self._client_script_path} "
            f"--host {dst_ip} "
            f"--port {self.probe_port} "
            f"--payload-size {payload_size} "
            f"--timeout {self.socket_timeout_secs} "
            f"--retries {self.client_retries}"
        )
        output = src_host.cmd(cmd).strip()

        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            result = {
                "success": 0,
                "latency_ms": None,
                "response_size": None,
                "connection_error": "invalid_probe_output",
                "retries": self.client_retries,
                "throughput_kbps": None,
                "payload_ok": False,
                "timeout_occurred": 0,
                "raw_output": output,
            }

        result.setdefault("raw_output", output)
        return result

    def _build_epoch_pairs(
        self,
        src_domain: Domain,
        dst_domain: Domain,
        interactions_per_ordered_pair: int,
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        src_ids = list(src_domain.node_ids)
        dst_ids = list(dst_domain.node_ids)

        if not src_ids or not dst_ids:
            return pairs

        # Coverage 1: every target node is observed enough times in each epoch.
        for round_idx in range(self.min_target_observations_per_epoch):
            shuffled_targets = list(dst_ids)
            self.rng.shuffle(shuffled_targets)
            for target_idx, dst_node_id in enumerate(shuffled_targets):
                src_node_id = src_ids[(target_idx + round_idx) % len(src_ids)]
                pairs.append((src_node_id, dst_node_id))

        # Coverage 2: every source node actively participates enough times in each epoch.
        source_counts = {node_id: 0 for node_id in src_ids}
        for src_node_id, _ in pairs:
            source_counts[src_node_id] += 1

        shuffled_sources = list(src_ids)
        self.rng.shuffle(shuffled_sources)
        for src_idx, src_node_id in enumerate(shuffled_sources):
            while source_counts[src_node_id] < self.min_source_interactions_per_epoch:
                dst_node_id = dst_ids[(source_counts[src_node_id] + src_idx) % len(dst_ids)]
                pairs.append((src_node_id, dst_node_id))
                source_counts[src_node_id] += 1

        # Optional extra random interactions for more diversity within the same epoch.
        for _ in range(max(0, interactions_per_ordered_pair)):
            pairs.append((self.rng.choice(src_ids), self.rng.choice(dst_ids)))

        self.rng.shuffle(pairs)
        return pairs

    def simulate_many_epochs(
        self,
        net,
        domains: list[Domain],
        num_epochs: int,
        interactions_per_ordered_pair: int = 10,
        start_epoch: int = 1,
        epoch_callback=None,
        semantic_score_index: dict[tuple[str, str], float] | None = None,
    ) -> list[InteractionEvent]:
        all_events: list[InteractionEvent] = []

        if len(domains) < 2:
            return all_events

        self._ensure_servers_started(net, domains)

        for epoch in range(start_epoch, start_epoch + num_epochs):
            if epoch_callback is not None:
                epoch_callback(epoch)

            for src_domain, dst_domain in permutations(domains, 2):
                if not src_domain.node_ids or not dst_domain.node_ids:
                    continue

                scheduled_pairs = self._build_epoch_pairs(
                    src_domain=src_domain,
                    dst_domain=dst_domain,
                    interactions_per_ordered_pair=interactions_per_ordered_pair,
                )

                for src_node_id, dst_node_id in scheduled_pairs:
                    payload_size = self._random_payload_size()

                    src_host = net.get(src_node_id)
                    dst_node = dst_domain.nodes[dst_node_id]
                    dst_ip = dst_node.ip

                    probe_result = self._probe_tcp(src_host, dst_ip, payload_size)
                    success = int(probe_result.get("success", 0))
                    baseline = self._get_baseline(
                        src_domain=src_domain.domain_id,
                        dst_domain=dst_domain.domain_id,
                    )
                    baseline_before = baseline.snapshot()
                    public_semantic_score = self._public_semantic_score(
                        semantic_score_index=semantic_score_index,
                        target_domain=dst_domain.domain_id,
                        target_node=dst_node_id,
                    )
                    compliant, observation_components = self._assess_interaction_compliance(
                        success=success,
                        probe_result=probe_result,
                        baseline=baseline,
                        public_semantic_score=public_semantic_score,
                    )
                    previous_observed_score, observed_score = self._update_observed_semantic_score(
                        observer_domain=src_domain.domain_id,
                        target_domain=dst_domain.domain_id,
                        target_node=dst_node_id,
                        public_semantic_score=public_semantic_score,
                        compliant=compliant,
                    )
                    self._update_baseline(
                        src_domain=src_domain.domain_id,
                        dst_domain=dst_domain.domain_id,
                        success=success,
                        probe_result=probe_result,
                    )
                    baseline_after = baseline.snapshot()

                    event = self._make_event(
                        epoch=epoch,
                        src_domain=src_domain.domain_id,
                        src_node=src_node_id,
                        dst_domain=dst_domain.domain_id,
                        dst_node=dst_node_id,
                        success=success,
                        observed_score=observed_score,
                        protocol="tcp",
                        latency_ms=probe_result.get("latency_ms"),
                        timeout_occurred=int(probe_result.get("timeout_occurred", 0)),
                        response_size=probe_result.get("response_size"),
                        connection_error=probe_result.get("connection_error"),
                        retries=probe_result.get("retries", 0),
                        throughput_kbps=probe_result.get("throughput_kbps"),
                        metadata={
                            "dst_ip": dst_ip,
                            "payload_size": payload_size,
                            "payload_ok": probe_result.get("payload_ok", False),
                            "probe_output": probe_result.get("raw_output", ""),
                            "observation_components": observation_components,
                            "compliant_with_semantic_capability": compliant,
                            "public_semantic_score": round(public_semantic_score, 6),
                            "previous_observed_semantic_score": previous_observed_score,
                            "baseline_before": baseline_before,
                            "baseline_after": baseline_after,
                        },
                    )
                    all_events.append(event)

        return all_events

    def print_event_sample(self, events: list[InteractionEvent], limit: int = 10) -> None:
        print("\n===== Interaction Event Sample =====")
        for e in events[:limit]:
            payload_size = e.metadata.get("payload_size") if e.metadata else None
            components = e.metadata.get("observation_components", {}) if e.metadata else {}
            baseline_before = e.metadata.get("baseline_before", {}) if e.metadata else {}
            compliant = e.metadata.get("compliant_with_semantic_capability") if e.metadata else None
            public_semantic = e.metadata.get("public_semantic_score") if e.metadata else None
            previous_observed = e.metadata.get("previous_observed_semantic_score") if e.metadata else None
            print(
                f"{e.event_id}: "
                f"{e.src_domain}/{e.src_node} -> {e.dst_domain}/{e.dst_node}, "
                f"epoch={e.epoch}, success={e.success}, protocol={e.protocol}, "
                f"public_semantic={public_semantic}, prev_observed={previous_observed}, "
                f"observed_score={e.observed_score}, compliant={compliant}, "
                f"payload_size={payload_size}, "
                f"latency_ms={e.latency_ms}, size={e.response_size}, "
                f"throughput_kbps={e.throughput_kbps}, error={e.connection_error}, "
                f"retries={e.retries}, "
                f"baseline_samples={baseline_before.get('sample_count')}, "
                f"components={components}"
            )
