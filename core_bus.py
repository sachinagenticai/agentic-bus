import asyncio
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class BusMessage:
    sender: str
    indices: torch.Tensor
    values: torch.Tensor


class AgenticBus:
    """Asynchronous shared memory bus for sparse hidden-state communication."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._messages: Dict[str, BusMessage] = {}

    async def write_message(self, sender: str, indices: torch.Tensor, values: torch.Tensor) -> None:
        """Write a sparse hidden-state payload to the bus."""
        async with self._lock:
            self._messages[sender] = BusMessage(sender=sender, indices=indices.cpu(), values=values.cpu())

    async def read_messages(self, exclude_sender: str = None) -> List[BusMessage]:
        """Read all messages except optionally the sender's own."""
        async with self._lock:
            return [msg for key, msg in self._messages.items() if key != exclude_sender]

    async def clear_messages(self) -> None:
        """Remove all stored messages from the bus."""
        async with self._lock:
            self._messages.clear()


class Agent:
    """Agent that communicates using top-K high-variance hidden-state entries."""

    def __init__(self, name: str, hidden_size: int, top_k: int, bus: AgenticBus):
        self.name = name
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.bus = bus
        self.hidden_state = torch.randn(hidden_size)

    def update_hidden_state(self) -> None:
        """Simulate a hidden-state update step."""
        noise = torch.randn_like(self.hidden_state) * 0.1
        self.hidden_state = torch.tanh(self.hidden_state + noise)

    @staticmethod
    def select_topk_by_variance(hidden_state: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the top-K dimensions with highest variance or magnitude-driven variance surrogate."""
        if hidden_state.ndim == 1:
            # Use magnitude as a proxy for uncertainty in a single-vector hidden state.
            scores = hidden_state.abs()
        else:
            scores = hidden_state.var(dim=0, unbiased=False)
        topk = torch.topk(scores, k=min(k, hidden_state.numel()), largest=True)
        return topk.indices, hidden_state[topk.indices]

    async def send(self) -> None:
        """Write the agent's top-K sparse hidden state to the bus."""
        indices, values = self.select_topk_by_variance(self.hidden_state, self.top_k)
        await self.bus.write_message(sender=self.name, indices=indices, values=values)

    async def receive(self) -> Dict[str, torch.Tensor]:
        """Read peer messages from the bus and reconstruct a sparse combined tensor."""
        messages = await self.bus.read_messages(exclude_sender=self.name)
        reconstructed: Dict[str, torch.Tensor] = {}
        for msg in messages:
            tensor = torch.zeros(self.hidden_size)
            tensor[msg.indices.long()] = msg.values
            reconstructed[msg.sender] = tensor
        return reconstructed

    async def step(self) -> None:
        """One communication step: update, send sparse state, and read peers."""
        self.update_hidden_state()
        await self.send()
        peers = await self.receive()
        if peers:
            summary = {name: peer.sum().item() for name, peer in peers.items()}
            print(f"{self.name} received sparse states: {summary}")
        else:
            print(f"{self.name} found no peer messages yet.")


async def mock_simulation_run():
    bus = AgenticBus()
    agents = [Agent(name=f"agent_{i}", hidden_size=32, top_k=5, bus=bus) for i in range(3)]

    for step_idx in range(5):
        print(f"\n=== Simulation step {step_idx + 1} ===")
        await asyncio.gather(*(agent.step() for agent in agents))
        await asyncio.sleep(0.1)

    print("\nFinal bus contents:")
    messages = await bus.read_messages()
    for msg in messages:
        print(f"- {msg.sender}: indices={msg.indices.tolist()} values={msg.values.tolist()}")


if __name__ == "__main__":
    asyncio.run(mock_simulation_run())
