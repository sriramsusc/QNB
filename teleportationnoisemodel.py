import numpy as np
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.qubits.qformalism import QFormalism
from netsquid.qubits.qubitapi import assign_qstate

class TeleportationNoiseModel(QuantumErrorModel):
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")
        self.add_property("alpha", alpha)

    @property
    def alpha(self) -> float:
        return self.properties["alpha"]
    
    def error_operation(self, qubits: list, delta_time: float = 0, **kwargs):
        alpha = self.alpha
        if alpha >= 1.0:
            return
        zero_dm = np.array([[1, 0],
                            [0, 0]], dtype=complex)
        for qb in qubits:
            if qb is None:
                continue
            old_dm = qb.qstate.dm.copy()
            new_dm = alpha * old_dm + (1.0 - alpha) * zero_dm
            assign_qstate(qb, new_dm, QFormalism.DM)
