from __future__ import annotations

from typing import Dict, List

from core.entities import Domain


def _build_netem_parts(profile: Dict) -> List[str]:
    parts: List[str] = []

    delay_ms = profile.get("delay_ms")
    jitter_ms = profile.get("jitter_ms")
    loss_pct = profile.get("loss_pct")
    corrupt_pct = profile.get("corrupt_pct")
    duplicate_pct = profile.get("duplicate_pct")
    reorder_pct = profile.get("reorder_pct")
    reorder_correlation_pct = profile.get("reorder_correlation_pct", 0)
    rate_kbit = profile.get("rate_kbit")

    if delay_ms is not None:
        part = f"delay {delay_ms}ms"
        if jitter_ms is not None:
            part += f" {jitter_ms}ms"
        parts.append(part)

    if loss_pct is not None:
        parts.append(f"loss {loss_pct}%")

    if corrupt_pct is not None:
        parts.append(f"corrupt {corrupt_pct}%")

    if duplicate_pct is not None:
        parts.append(f"duplicate {duplicate_pct}%")

    if reorder_pct is not None:
        parts.append(f"reorder {reorder_pct}% {reorder_correlation_pct}%")

    if rate_kbit is not None:
        parts.append(f"rate {rate_kbit}kbit")

    return parts


class NetworkAttackInjector:
    """
    Apply node-specific network attacks to Mininet hosts based on attack_type.
    """

    def __init__(self, attack_profiles: Dict[str, Dict] | None = None):
        self.attack_profiles = attack_profiles or {}

    def _interface_name(self, node_id: str) -> str:
        return f"{node_id}-eth0"

    def _effective_profile(self, node, epoch: int) -> Dict | None:
        if node.label != "malicious":
            return None

        attack_type = node.attack_type or "none"
        profile = self.attack_profiles.get(attack_type)
        if not profile:
            return None

        if attack_type == "on_off":
            period = int(profile.get("period", 2))
            active_epochs = int(profile.get("active_epochs", 1))
            phase = (epoch - 1) % max(period, 1)
            if phase >= active_epochs:
                return None

        return profile

    def _apply_profile_to_node(self, host, node_id: str, profile: Dict | None) -> None:
        intf = self._interface_name(node_id)
        host.cmd(f"tc qdisc del dev {intf} root >/dev/null 2>&1")

        if not profile:
            return

        qdisc = profile.get("qdisc", "netem")

        if qdisc == "tbf":
            rate_kbit = profile.get("rate_kbit", 512)
            burst_kbit = profile.get("burst_kbit", 32)
            latency_ms = profile.get("latency_ms", 100)
            host.cmd(
                f"tc qdisc replace dev {intf} root tbf "
                f"rate {rate_kbit}kbit burst {burst_kbit}kb latency {latency_ms}ms"
            )
            return

        parts = _build_netem_parts(profile)
        if parts:
            host.cmd(f"tc qdisc replace dev {intf} root netem {' '.join(parts)}")

    def apply_epoch(self, net, domains: list[Domain], epoch: int) -> None:
        for domain in domains:
            for node_id in domain.node_ids:
                node = domain.nodes[node_id]
                host = net.get(node_id)
                profile = self._effective_profile(node, epoch)
                self._apply_profile_to_node(host, node_id, profile)

    def clear(self, net, domains: list[Domain]) -> None:
        for domain in domains:
            for node_id in domain.node_ids:
                host = net.get(node_id)
                intf = self._interface_name(node_id)
                host.cmd(f"tc qdisc del dev {intf} root >/dev/null 2>&1")
