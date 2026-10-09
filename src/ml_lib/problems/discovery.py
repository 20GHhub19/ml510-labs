"""Describe an investigation without inventing a prediction target."""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class DiscoveryContract:
    name: str
    unit: str
    observation_window: str
    information_set: str
    output: str
    action: str
    limits: str

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
