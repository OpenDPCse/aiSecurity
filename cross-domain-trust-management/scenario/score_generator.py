import random
from typing import List

from core.entities import Domain



class ScoreGenerator:
    """
    Generate domain-local raw trust scores.
    These scores are only meaningful inside each domain.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)


    #根据指定的分布类型和参数生成一个原始评分。支持均匀分布、Beta分布、截断正态分布和双峰Beta分布。
    def _generate_one(self, dist: str, params: dict) -> float:
        dist = dist.lower()

        if dist == "uniform":
            low = params.get("low", 0.0)
            high = params.get("high", 1.0)
            x = self.rng.uniform(low, high)

        elif dist == "beta":
            alpha = params.get("alpha", 2.0)
            beta = params.get("beta", 2.0)
            x = self.rng.betavariate(alpha, beta)

        elif dist == "truncnorm":
            mu = params.get("mu", 0.5)
            sigma = params.get("sigma", 0.15)
            while True:
                x = self.rng.gauss(mu, sigma)
                if 0.0 <= x <= 1.0:
                    break

        elif dist == "bimodal_beta":
            mix = params.get("mix", 0.5)

            alpha1 = params.get("alpha1", 2.0)
            beta1 = params.get("beta1", 6.0)

            alpha2 = params.get("alpha2", 6.0)
            beta2 = params.get("beta2", 2.0)

            if self.rng.random() < mix:
                x = self.rng.betavariate(alpha1, beta1)
            else:
                x = self.rng.betavariate(alpha2, beta2)

        else:
            raise ValueError(f"Unsupported score distribution: {dist}")

        return max(0.0, min(1.0, round(x, 6)))

    #为域中的节点分配原始评分和简单的元信息。默认情况下，只有普通域节点会收到原始评分。可以选择是否包括网关和信任管理器。
    def populate_domain_scores(
        self,
        domain: Domain,
        include_gateway: bool = False,
        include_trust_manager: bool = False,
        default_obs_range: tuple[int, int] = (5, 40),
        current_epoch: int = 1,
    ) -> None:
        """
        Assign raw_score and simple metadata to nodes in a domain.
        By default, only ordinary domain nodes receive raw scores.
        """
        target_ids: List[str] = list(domain.node_ids)

        if include_gateway:
            target_ids.append(domain.gateway_id)

        if include_trust_manager:
            target_ids.append(domain.trust_manager_id)

        for node_id in target_ids:
            node = domain.nodes[node_id]

            score = self._generate_one(domain.score_dist, domain.score_params)
            node.raw_score = score

            # 简单元信息，后面语义统一阶段会用到
            obs_count = self.rng.randint(*default_obs_range)
            last_update_epoch = max(1, current_epoch - self.rng.randint(0, 3))

            node.metadata["observation_count"] = obs_count
            node.metadata["last_update_epoch"] = last_update_epoch

    #为多个域批量生成原始评分，调用populate_domain_scores方法为每个域中的节点分配原始评分和元信息。
    def populate_many_domains(
        self,
        domains: list[Domain],
        include_gateway: bool = False,
        include_trust_manager: bool = False,
        current_epoch: int = 1,
    ) -> None:
        for domain in domains:
            self.populate_domain_scores(
                domain=domain,
                include_gateway=include_gateway,
                include_trust_manager=include_trust_manager,
                current_epoch=current_epoch,
            )