from __future__ import annotations

import random
from typing import List, Dict, Tuple

import numpy as np
import netsquid as ns
from netsquid.qubits import operators as ops
from netsquid.components.instructions import IGate
from netsquid.components.component import Message
from netsquid.components.qsource import QSource
from netsquid.nodes.node import Node
from netsquid.protocols import Protocol
from netsquid.qubits.qubitapi import create_qubits, operate, exp_value
from netsquid.qubits.operators import Operator


class MultiNodeRB(Protocol):
    
    def __init__(
        self,
        nodes: List[Node],
        min_bounces: int,
        max_bounces: int,
        n_samples: int,
        name: str | None = None,
        shots: int = 4000,
        add_shot_noise: bool = True,
    ):
        """
        Parameters
        ----------
        nodes : List[Node]
            The ordered list of nodes in the chain.
        min_bounces : int
            The minimum number of bounces (round‐trip traversals) to perform.
        max_bounces : int
            The maximum number of bounces to perform.
        n_samples : int
            The number of random sequences (samples) to run per bounce length.
        name : str | None, optional
            Optional name for the protocol, by default None.
        shots : int, optional
            Number of measurement shots to simulate when adding shot noise.  The
            variance of the Gaussian noise scales as ``1/√shots``.  Default is
            4000.
        add_shot_noise : bool, optional
            Whether to add Gaussian shot noise to fidelity values.  When
            ``True`` (default), a noise term ``np.random.normal(0, σ)`` is
            added to each fidelity sample, with
            ``σ = sqrt((1 + fid) * (1 - fid)) / sqrt(shots)``.  If set to
            ``False``, fidelities are recorded without additional noise.
        """
        super().__init__(name=name)

        if max_bounces < min_bounces:
            raise ValueError("max_bounces must be ≥ min_bounces")

        self.nodes            = nodes
        self.n_nodes          = len(nodes)
        self.min_bounces      = min_bounces
        self.max_bounces      = max_bounces
        self.n_samples        = n_samples
        self.shots            = shots
        self.add_shot_noise   = add_shot_noise
        self._current_bounces = min_bounces
        self._current_sample  = 1
        self._gate_history: List[IGate]    = []
        self._fidelity_samples: List[float] = []
        self._mean_fidelity: Dict[int, float]      = {}
        self._all_samples: Dict[int, List[float]]  = {}
        self.cliffords: List[IGate] = self._generate_cliffords()
        self.src  = nodes[0]
        self.sink = nodes[-1]
        if not self.is_connected:
            raise RuntimeError("Nodes / channels / memories are not properly configured")
        self.add_signal("ROUND_DONE")

    @property
    def is_connected(self) -> bool:
        if any(node.qmemory.num_positions == 0 for node in self.nodes):
            return False
        for i in range(self.n_nodes - 1):
            label = f"quantum{i}{i+1}"
            if (self.nodes[i].get_conn_port(self.nodes[i+1].ID, label=label) is None or
                self.nodes[i+1].get_conn_port(self.nodes[i].ID, label=label) is None):
                return False
        if self._get_qsource(self.src) is None:
            return False
        return True

    def get_fidelity(self) -> Tuple[Dict[int, float], Dict[int, List[float]]]:
        return dict(self._mean_fidelity), {k: v.copy() for k, v in self._all_samples.items()}

    def run(self):
        for m in range(self.min_bounces, self.max_bounces + 1):
            self._current_bounces = m
            self._fidelity_samples.clear()
            for s in range(1, self.n_samples + 1):
                self._current_sample = s
                self._gate_history.clear()
                qubit = yield from self._prepare_initial_state()
                yield from self._random_bounce_walk(qubit, m)
                # Evaluate fidelity of the qubit after the random walk
                fid = float(self._evaluate_fidelity(qubit))
                # Add random shot noise if requested
                if self.add_shot_noise:
                    # The variance of a Pauli measurement with expectation fid is (1 - fid**2),
                    # but the original NetSquid code used sqrt((1+fid)*(1-fid)) which is equivalent.
                    sigma = np.sqrt((1.0 + fid) * (1.0 - fid)) / np.sqrt(self.shots)
                    fid += float(np.random.normal(0.0, sigma))
                self._fidelity_samples.append(fid)
                # Reset all memories for the next sample
                for node in self.nodes:
                    node.qmemory.reset()
            # Compute mean fidelity for this bounce count
            mean_fid = float(np.mean(self._fidelity_samples))
            self._mean_fidelity[m] = mean_fid
            self._all_samples[m]   = self._fidelity_samples.copy()
            # Notify listeners that a round has completed
            self.send_signal("ROUND_DONE", result=mean_fid)
        return self._mean_fidelity, self._all_samples

    def _get_qsource(self, node: Node) -> QSource | None:
        for comp in node.subcomponents.values():
            if isinstance(comp, QSource):
                return comp
        return None

    def _prepare_initial_state(self):
        qsource = self._get_qsource(self.src)
        qsource.trigger()
        yield self.await_port_output(qsource.ports["qout0"])
        msg   = qsource.ports["qout0"].rx_output()
        qubit = msg.items[0]
        self.src.qmemory.put(qubit, positions=[0])
        return qubit

    def _random_bounce_walk(self, qubit, m: int):
        forward  = list(range(self.n_nodes))
        backward = list(range(self.n_nodes - 2, -1, -1))
        single = forward + backward
        path = single + single[1:] * (m - 1)
        current_idx   = path[0]
        current_node  = self.nodes[current_idx]
        for step, next_idx in enumerate(path[1:], start=1):
            next_node = self.nodes[next_idx]
            instr = random.choice(self.cliffords)
            current_node.qmemory.execute_instruction(instr, [0])
            yield self.await_program(current_node.qmemory)
            self._gate_history.append(instr)
            (qubit,) = current_node.qmemory.pop(0)
            label    = f"quantum{min(current_idx, next_idx)}{max(current_idx, next_idx)}"
            out_port = current_node.get_conn_port(next_node.ID, label=label)
            in_port  = next_node.get_conn_port(current_node.ID, label=label)
            out_port.tx_output(Message(qubit))
            yield self.await_port_input(in_port)
            next_node.qmemory.put(qubit, positions=[0])
            current_idx, current_node = next_idx, next_node

    def _evaluate_fidelity(self, qubit):
        ref1, ref2 = create_qubits(2)
        operate(ref2, ops.X)
        for instr in self._gate_history:
            gate = instr._operator
            operate(ref1, gate)
            operate(ref2, gate)
        O_ref = Operator("ref", (ref1.qstate.dm - ref2.qstate.dm) / 2)
        return exp_value(qubit, O_ref)

    def _generate_cliffords(self):
        cliff_ops = [
            ops.I, ops.X, ops.Y, ops.Z, ops.H, ops.S,
            ops.X * ops.H,  ops.Y * ops.H,  ops.Z * ops.H,
            ops.X * ops.S,  ops.Y * ops.S,  ops.Z * ops.S,
            ops.X * ops.H * ops.S,  ops.Y * ops.H * ops.S,  ops.Z * ops.H * ops.S,
            ops.H * ops.S * ops.H,
            ops.X * ops.H * ops.S * ops.H,
            ops.Y * ops.H * ops.S * ops.H,
            ops.Z * ops.H * ops.S * ops.H,
            ops.S * ops.H * ops.S,
            ops.X * ops.S * ops.H * ops.S,
            ops.Y * ops.S * ops.H * ops.S,
            ops.Z * ops.S * ops.H * ops.S,
        ]
        return [IGate(f"Clifford_{i}", op) for i, op in enumerate(cliff_ops)]
