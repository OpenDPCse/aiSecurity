import random
from typing import List

from core.entities import Domain

#这个类负责为域内的节点分配真实的标签（正常或恶意）和攻击类型。
#它根据指定的恶意节点比例随机选择一些节点进行标记，并提供选项是否包括网关和信任管理器在内的候选节点，以及是否覆盖已有标签。
#它还提供了一个方法来批量处理多个域，并打印标签分布的摘要信息，便于调试和分析。
class LabelAssigner:
    """
    Assign ground-truth labels to nodes.

    First version:
    - default labels only ordinary nodes in domain.node_ids
    - label ∈ {normal, malicious}
    - default malicious attack_type = dishonest
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _choose_attack_type(self, malicious_attack_type):
        if isinstance(malicious_attack_type, str):
            return malicious_attack_type

        if isinstance(malicious_attack_type, list):
            if not malicious_attack_type:
                raise ValueError("malicious_attack_type list cannot be empty")
            return self.rng.choice(malicious_attack_type)

        if isinstance(malicious_attack_type, dict):
            if not malicious_attack_type:
                raise ValueError("malicious_attack_type dict cannot be empty")

            attack_types = list(malicious_attack_type.keys())
            weights = list(malicious_attack_type.values())
            if any(w < 0 for w in weights):
                raise ValueError("malicious_attack_type weights must be non-negative")
            if sum(weights) <= 0:
                raise ValueError("malicious_attack_type weights must sum to a positive value")

            return self.rng.choices(attack_types, weights=weights, k=1)[0]

        raise TypeError(
            "malicious_attack_type must be a string, list[str], or dict[str, float]"
        )

    def assign_labels(
        self,
        domain: Domain,
        malicious_ratio: float = 0.2,
        malicious_attack_type="dishonest",
        include_gateway: bool = False,
        include_trust_manager: bool = False,
        overwrite_existing: bool = True,
    ) -> None:
        """
        Assign labels inside one domain.

        Parameters
        ----------
        malicious_ratio:
            Fraction of candidate nodes labeled as malicious.
        malicious_attack_type:
            Can be:
            - one string, e.g. "dishonest"
            - a list, e.g. ["dishonest", "on_off", "sybil"]
            - a weighted dict, e.g. {"dishonest": 0.6, "on_off": 0.4}
        include_gateway / include_trust_manager:
            Whether gateway / trust manager can also be labeled malicious.
            Recommended: keep both False for now.
        overwrite_existing:
            If True, reset all candidate nodes to normal first.
        """
        if not (0.0 <= malicious_ratio <= 1.0):
            raise ValueError("malicious_ratio must be in [0, 1]")

        candidate_ids: List[str] = list(domain.node_ids)

        if include_gateway:
            candidate_ids.append(domain.gateway_id)

        if include_trust_manager:
            candidate_ids.append(domain.trust_manager_id)

        if not candidate_ids:
            return

        if overwrite_existing:
            for node_id in candidate_ids:
                node = domain.nodes[node_id]
                node.label = "normal"
                node.attack_type = "none"

        malicious_count = round(len(candidate_ids) * malicious_ratio)

        # if ratio > 0 but rounding gives 0, force at least one malicious node
        if malicious_ratio > 0 and malicious_count == 0:
            malicious_count = 1

        malicious_count = min(malicious_count, len(candidate_ids))
        malicious_ids = set(self.rng.sample(candidate_ids, malicious_count))

        for node_id in malicious_ids:
            node = domain.nodes[node_id]
            node.label = "malicious"
            node.attack_type = self._choose_attack_type(malicious_attack_type)

    def assign_many_domains(
        self,
        domains: list[Domain],
        malicious_ratio: float = 0.2,
        malicious_attack_type="dishonest",
        include_gateway: bool = False,
        include_trust_manager: bool = False,
        overwrite_existing: bool = True,
    ) -> None:
        """
        Assign labels to multiple domains with the same settings.
        """
        for domain in domains:
            self.assign_labels(
                domain=domain,
                malicious_ratio=malicious_ratio,
                malicious_attack_type=malicious_attack_type,
                include_gateway=include_gateway,
                include_trust_manager=include_trust_manager,
                overwrite_existing=overwrite_existing,
            )

    def print_label_summary(self, domains: list[Domain]) -> None:
        """
        Print a simple label summary for debugging.
        """
        print("\n===== Label Summary =====")
        for domain in domains:
            print(f"[Domain {domain.domain_id}]")
            for node_id in domain.node_ids:
                node = domain.nodes[node_id]
                print(
                    f"  {node.node_id:<6} "
                    f"label={node.label:<10} "
                    f"attack_type={node.attack_type}"
                )
