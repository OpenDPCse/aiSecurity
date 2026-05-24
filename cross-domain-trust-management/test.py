from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSBridge
from mininet.log import setLogLevel


class TwoDomainTopo(Topo):
    def build(self):
        # Domain A
        a1 = self.addHost('a1')
        a2 = self.addHost('a2')
        sA = self.addSwitch('s1')

        # Domain B
        b1 = self.addHost('b1')
        b2 = self.addHost('b2')
        sB = self.addSwitch('s2')

        # Inter-domain link
        self.addLink(sA, sB)

        # Intra-domain links
        self.addLink(a1, sA)
        self.addLink(a2, sA)
        self.addLink(b1, sB)
        self.addLink(b2, sB)


if __name__ == '__main__':
    setLogLevel('info')
    topo = TwoDomainTopo()
    net = Mininet(topo=topo, switch=OVSBridge, controller=None)

    net.start()
    print("Ping loss:", net.pingAll())

    print("\nHosts in Domain A: a1, a2")
    print("Hosts in Domain B: b1, b2")

    net.stop()