from mininet.cli import CLI
from scenario.domain_builder import DomainBuilder
from topo.net_adapter import MininetDomainAdapter


def main():
    builder = DomainBuilder()

    domain_a = builder.build_domain(
        domain_id="a",
        num_nodes=5,
        score_dist="beta",
        score_params={"alpha": 5, "beta": 2}
    )

    domain_b = builder.build_domain(
        domain_id="b",
        num_nodes=5,
        score_dist="beta",
        score_params={"alpha": 2, "beta": 5}
    )

    adapter = MininetDomainAdapter(domains=[domain_a, domain_b])
    adapter.print_plan()

    net = adapter.start_network()

    print("\nNetwork started. Entering CLI...\n")
    CLI(net)

    net.stop()


if __name__ == "__main__":
    main()