from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from ipaddress import ip_interface
from typing import Dict, List, Optional, Tuple

from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import Node, OVSBridge
from mininet.topo import Topo

from core.entities import Domain


#功能是只把ip提取出来，不保留CIDR部分。
def ip_only(ip_cidr: str) -> str:
    """10.0.1.1/24 -> 10.0.1.1"""
    return str(ip_interface(ip_cidr).ip)

#这个是把IP地址转换成网络地址，保留CIDR部分。
def network_of(ip_cidr: str) -> str:
    """10.0.1.1/24 -> 10.0.1.0/24"""
    return str(ip_interface(ip_cidr).network)


def domain_profile_to_link_params(profile: Dict[str, float]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    if not profile:
        return params

    bandwidth_mbps = profile.get("bandwidth_mbps")
    delay_ms = profile.get("delay_ms")
    jitter_ms = profile.get("jitter_ms")
    loss_pct = profile.get("loss_pct")

    if bandwidth_mbps is not None:
        params["bw"] = bandwidth_mbps
    if delay_ms is not None:
        params["delay"] = f"{delay_ms}ms"
    if jitter_ms is not None:
        params["jitter"] = f"{jitter_ms}ms"
    if loss_pct is not None:
        params["loss"] = loss_pct

    return params

#这个类是一个Mininet节点的子类，代表一个启用了IP转发功能的Linux路由器。它重写了config方法，在配置节点时启用IP转发，并在终止节点时禁用IP转发。
class LinuxRouter(Node):
    """A Mininet node with IP forwarding enabled."""

    def config(self, **params):
        super().config(**params)
        self.cmd("sysctl -w net.ipv4.ip_forward=1")

    def terminate(self):
        self.cmd("sysctl -w net.ipv4.ip_forward=0")
        super().terminate()

#这个类定义了一个Mininet拓扑，基于适配器生成的规范构建网络。它的build方法根据每个域的规范添加主机、路由器、交换机和局域网链接，并根据域之间的链接添加网关之间的点对点链接。
@dataclass
class DomainNetSpec:
    """Network-level information for a domain."""
    domain_id: str
    switch_name: str
    lan_subnet: str
    gateway_name: str
    gateway_lan_ip: str
    trust_manager_name: str
    trust_manager_ip: str
    network_profile: Dict[str, float] = field(default_factory=dict)
    host_ips: Dict[str, str] = field(default_factory=dict)



@dataclass
class InterDomainLinkSpec:
    """Point-to-point link between two gateways."""
    domain_a: str
    domain_b: str
    gw_a: str
    gw_b: str
    a_intf: str
    b_intf: str
    a_ip: str
    b_ip: str
    subnet: str


class ScenarioTopo(Topo):
    """
    Mininet topology built from adapter-generated specs.
    """
    def build(self, domain_specs: Dict[str, DomainNetSpec], inter_links: List[InterDomainLinkSpec]):
        # 1) Add per-domain hosts, router, switch and LAN links
        for _, spec in domain_specs.items():
            sw = self.addSwitch(spec.switch_name)
            link_params = domain_profile_to_link_params(spec.network_profile)

            # trust manager
            self.addHost(spec.trust_manager_name, ip=spec.trust_manager_ip)
            self.addLink(spec.trust_manager_name, sw, **link_params)

            # ordinary hosts
            for host_name, host_ip in spec.host_ips.items():
                self.addHost(host_name, ip=host_ip)
                self.addLink(host_name, sw, **link_params)

            # gateway/router
            self.addNode(spec.gateway_name, cls=LinuxRouter, ip=spec.gateway_lan_ip)
            self.addLink(
                spec.gateway_name,
                sw,
                intfName1=f"{spec.gateway_name}-eth0",
                params1={"ip": spec.gateway_lan_ip},
                **link_params,
            )

        # 2) Add inter-domain gateway links
        for link in inter_links:
            self.addLink(
                link.gw_a,
                link.gw_b,
                intfName1=link.a_intf,
                intfName2=link.b_intf,
                params1={"ip": link.a_ip},
                params2={"ip": link.b_ip}
            )


class MininetDomainAdapter:
    """
    Translate abstract Domain objects into a runnable Mininet network.

    Responsibilities:
    1. Assign per-domain LAN subnet/IPs
    2. Assign gateway-to-gateway /30 IPs
    3. Generate a topology spec
    4. Build and start Mininet
    5. Configure static routes
    """

    def __init__(
        self,
        domains: List[Domain],
        inter_domain_pairs: Optional[List[Tuple[str, str]]] = None,
        switch_cls=OVSBridge,
        link_cls=TCLink,
    ):
        self.domains = domains
        self.domain_map: Dict[str, Domain] = {d.domain_id: d for d in domains}
        self.switch_cls = switch_cls
        self.link_cls = link_cls

        # If no custom inter-domain graph is given,
        # connect gateways in a simple chain: a-b-c-d
        self.inter_domain_pairs = inter_domain_pairs or self._default_chain_pairs()

        self.domain_specs: Dict[str, DomainNetSpec] = {}
        self.inter_links: List[InterDomainLinkSpec] = []

        self._validate_domains()
        self._assign_domain_lan_addresses()
        self._assign_inter_domain_links()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_domains(self):
        seen = set()
        for d in self.domains:
            if d.domain_id in seen:
                raise ValueError(f"Duplicate domain_id: {d.domain_id}")
            seen.add(d.domain_id)

            if d.trust_manager_id not in d.nodes:
                raise ValueError(f"{d.domain_id}: trust_manager_id not found in domain.nodes")

            if d.gateway_id not in d.nodes:
                raise ValueError(f"{d.domain_id}: gateway_id not found in domain.nodes")

            for nid in d.node_ids:
                if nid not in d.nodes:
                    raise ValueError(f"{d.domain_id}: node_id {nid} not found in domain.nodes")

    def _default_chain_pairs(self) -> List[Tuple[str, str]]:
        if len(self.domains) <= 1:
            return []
        pairs = []
        for i in range(len(self.domains) - 1):
            pairs.append((self.domains[i].domain_id, self.domains[i + 1].domain_id))
        return pairs

    # ------------------------------------------------------------------
    # Address assignment
    # ------------------------------------------------------------------
    def _assign_domain_lan_addresses(self):
        """
        Assign one /24 LAN subnet per domain.

        Domain index 1 -> 10.0.1.0/24
        Domain index 2 -> 10.0.2.0/24
        ...
        """
        for idx, domain in enumerate(self.domains, start=1):
            switch_name = f"s{idx}"
            lan_subnet = f"10.0.{idx}.0/24"
            gateway_lan_ip = f"10.0.{idx}.1/24"
            tm_ip = f"10.0.{idx}.10/24"

            host_ips: Dict[str, str] = {}
            next_host_octet = 11

            # trust manager
            domain.nodes[domain.trust_manager_id].ip = ip_only(tm_ip)

            # ordinary nodes
            for nid in domain.node_ids:
                host_ip = f"10.0.{idx}.{next_host_octet}/24"
                next_host_octet += 1
                host_ips[nid] = host_ip
                domain.nodes[nid].ip = ip_only(host_ip)

            # gateway
            domain.nodes[domain.gateway_id].ip = ip_only(gateway_lan_ip)

            spec = DomainNetSpec(
                domain_id=domain.domain_id,
                switch_name=switch_name,
                lan_subnet=lan_subnet,
                gateway_name=domain.gateway_id,
                gateway_lan_ip=gateway_lan_ip,
                trust_manager_name=domain.trust_manager_id,
                trust_manager_ip=tm_ip,
                network_profile=dict(domain.network_profile),
                host_ips=host_ips
            )
            self.domain_specs[domain.domain_id] = spec

    def _assign_inter_domain_links(self):
        """
        Assign /30 subnets for gateway-to-gateway links.

        Link 1 -> 10.255.1.0/30
        Link 2 -> 10.255.2.0/30
        ...
        """
        gw_degree = defaultdict(int)

        for link_idx, (da, db) in enumerate(self.inter_domain_pairs, start=1):
            if da not in self.domain_specs or db not in self.domain_specs:
                raise ValueError(f"Unknown domain in inter_domain_pairs: {(da, db)}")

            spec_a = self.domain_specs[da]
            spec_b = self.domain_specs[db]

            subnet = f"10.255.{link_idx}.0/30"
            a_ip = f"10.255.{link_idx}.1/30"
            b_ip = f"10.255.{link_idx}.2/30"

            # gateway LAN already uses eth0
            gw_degree[spec_a.gateway_name] += 1
            gw_degree[spec_b.gateway_name] += 1

            a_intf = f"{spec_a.gateway_name}-eth{gw_degree[spec_a.gateway_name]}"
            b_intf = f"{spec_b.gateway_name}-eth{gw_degree[spec_b.gateway_name]}"

            self.inter_links.append(
                InterDomainLinkSpec(
                    domain_a=da,
                    domain_b=db,
                    gw_a=spec_a.gateway_name,
                    gw_b=spec_b.gateway_name,
                    a_intf=a_intf,
                    b_intf=b_intf,
                    a_ip=a_ip,
                    b_ip=b_ip,
                    subnet=subnet
                )
            )

    # ------------------------------------------------------------------
    # Topology / network creation
    # ------------------------------------------------------------------
    def create_topology(self) -> ScenarioTopo:
        topo = ScenarioTopo(domain_specs=self.domain_specs, inter_links=self.inter_links)
        return topo

    def build_network(self) -> Mininet:
        topo = self.create_topology()
        net = Mininet(
            topo=topo,
            switch=self.switch_cls,
            link=self.link_cls,
            controller=None,
            autoSetMacs=True
        )
        return net

    def start_network(self) -> Mininet:
        net = self.build_network()
        net.start()
        self.configure_routes(net)
        return net

    # ------------------------------------------------------------------
    # Route configuration
    # ------------------------------------------------------------------
    def configure_routes(self, net: Mininet):
        """
        1) Set default route of all non-gateway nodes to local gateway
        2) Add static routes on gateways for all remote LANs
        """
        # Step 1: default route for each host / trust manager
        for domain_id, spec in self.domain_specs.items():
            gateway_ip = ip_only(spec.gateway_lan_ip)

            # trust manager
            tm = net.get(spec.trust_manager_name)
            tm.cmd(f"ip route replace default via {gateway_ip}")

            # ordinary nodes
            for host_name in spec.host_ips:
                host = net.get(host_name)
                host.cmd(f"ip route replace default via {gateway_ip}")

        # Step 2: static routes on gateways
        adjacency = self._build_domain_graph()

        for src_domain_id, src_spec in self.domain_specs.items():
            src_gw = net.get(src_spec.gateway_name)

            for dst_domain_id, dst_spec in self.domain_specs.items():
                if src_domain_id == dst_domain_id:
                    continue

                next_hop = self._find_first_hop_next_hop(
                    src_domain_id=src_domain_id,
                    dst_domain_id=dst_domain_id,
                    adjacency=adjacency
                )
                if next_hop is None:
                    continue

                src_gw.cmd(f"ip route replace {dst_spec.lan_subnet} via {next_hop}")

    def _build_domain_graph(self):
        """
        Build adjacency:
        adjacency[A] = [(B, next_hop_ip_on_A_to_B), ...]
        """
        adjacency = defaultdict(list)

        for link in self.inter_links:
            # From A to B, A should use B's inter-domain IP as next hop
            adjacency[link.domain_a].append((link.domain_b, ip_only(link.b_ip)))
            # From B to A, B should use A's inter-domain IP as next hop
            adjacency[link.domain_b].append((link.domain_a, ip_only(link.a_ip)))

        return adjacency

    def _find_first_hop_next_hop(self, src_domain_id: str, dst_domain_id: str, adjacency) -> Optional[str]:
        """
        Return the first-hop next-hop IP from src_domain to dst_domain.
        Works for chain or general connected graph.
        """
        if src_domain_id == dst_domain_id:
            return None

        visited = set([src_domain_id])
        queue = deque()

        # initialize BFS from src neighbors
        for neighbor_domain, next_hop_ip in adjacency[src_domain_id]:
            queue.append((neighbor_domain, next_hop_ip))
            visited.add(neighbor_domain)

        while queue:
            current_domain, first_hop_ip = queue.popleft()

            if current_domain == dst_domain_id:
                return first_hop_ip

            for next_domain, _ in adjacency[current_domain]:
                if next_domain not in visited:
                    visited.add(next_domain)
                    queue.append((next_domain, first_hop_ip))

        return None

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def print_plan(self):
        print("\n=== Domain Network Plan ===")
        for domain_id, spec in self.domain_specs.items():
            print(f"[Domain {domain_id}]")
            print(f"  switch: {spec.switch_name}")
            print(f"  lan_subnet: {spec.lan_subnet}")
            if spec.network_profile:
                print(f"  network_profile: {spec.network_profile}")
            print(f"  gateway: {spec.gateway_name} ({spec.gateway_lan_ip})")
            print(f"  trust_manager: {spec.trust_manager_name} ({spec.trust_manager_ip})")
            for host_name, host_ip in spec.host_ips.items():
                print(f"  host: {host_name} ({host_ip})")

        print("\n=== Inter-domain Links ===")
        for link in self.inter_links:
            print(
                f"{link.domain_a}:{link.gw_a}({link.a_ip}, {link.a_intf}) "
                f"<--> "
                f"{link.domain_b}:{link.gw_b}({link.b_ip}, {link.b_intf})"
            )
