from core.entities import Domain, NodeProfile


#抽象的建立域，根据输入的参数创建一个Domain对象，并为其添加相应数量的节点（传感器）以及一个信任管理器和一个网关。

class DomainBuilder:
    def build_domain(
        self,
        domain_id: str,
        num_nodes: int,
        score_dist: str,
        score_params: dict,
        network_profile: dict | None = None,
    ):
        tm_id = f"{domain_id}_tm"
        gw_id = f"{domain_id}_gw"

        domain = Domain(
            domain_id=domain_id,
            trust_manager_id=tm_id,
            gateway_id=gw_id,
            score_dist=score_dist,
            score_params=score_params,
            network_profile=dict(network_profile or {}),
        )

        domain.nodes[tm_id] = NodeProfile(
            node_id=tm_id,
            role="trust_manager",
            domain_id=domain_id,
        )

        domain.nodes[gw_id] = NodeProfile(
            node_id=gw_id,
            role="gateway",
            domain_id=domain_id,
        )

        for i in range(1, num_nodes + 1):
            nid = f"{domain_id}{i}"
            domain.node_ids.append(nid)
            domain.nodes[nid] = NodeProfile(
                node_id=nid,
                role="sensor",
                domain_id=domain_id,
            )

        return domain
