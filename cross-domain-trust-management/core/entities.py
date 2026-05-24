from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NodeProfile:
    node_id: str
    role: str
    ip: Optional[str] = None
    domain_id: Optional[str] = None
    raw_score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    label: str = "normal"
    attack_type: str = "none"


@dataclass
class Domain:
    domain_id: str
    trust_manager_id: str
    gateway_id: str
    node_ids: List[str] = field(default_factory=list)
    score_dist: str = "beta"
    score_params: Dict = field(default_factory=dict)
    grade_thresholds: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    network_profile: Dict = field(default_factory=dict)
    nodes: Dict[str, NodeProfile] = field(default_factory=dict)


@dataclass
class UnifiedTrustRecord:
    node_id: str
    src_domain: str
    raw_score: float
    quantile: float
    calibrated_score: Optional[float]
    semantic_score: float
    grade: str
    freshness: float
    confidence: float
    role: str
    epoch: int


@dataclass
class InteractionEvent:
    event_id: str
    epoch: int
    src_domain: str
    src_node: str
    dst_domain: str
    dst_node: str
    success: int
    observed_score: float
    protocol: str = "logical"
    latency_ms: Optional[float] = None
    rtt_ms: Optional[float] = None
    packet_loss: Optional[float] = None
    timeout_occurred: int = 0
    response_code: Optional[int] = None
    response_size: Optional[int] = None
    connection_error: Optional[str] = None
    retries: int = 0
    throughput_kbps: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExternalObservationRecord:
    observer_domain: str
    target_domain: str
    target_node: str
    avg_observed_score: float
    observation_count: int
    epoch: int


@dataclass
class DetectionResult:
    node_id: str
    src_domain: str
    suspicion_score: float
    predicted_label: str
    true_label: str
