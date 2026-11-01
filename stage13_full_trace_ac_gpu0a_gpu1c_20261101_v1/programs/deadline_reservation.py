"""Online chain EDF and current-stage KV reservations. No trace reader."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Member:
    request: str
    chain: int
    deadline: float
    arrival: float
    prompt: int
    generated: int
    prediction: float
    max_tokens: int

    def blocks(self, block_size=16, horizon=50):
        # Only QRF output prediction, observed progress and public resource cap.
        forecast = max(self.prediction, self.generated + horizon)
        forecast = min(self.max_tokens, max(self.generated, forecast))
        return math.ceil((self.prompt + forecast) / block_size)


def reserve(members, now, capacity, max_seqs, block_size=16):
    """Pack current compound stages in chain deadline order; drain late work.

    Every chain receives one reward, regardless of compound width or token
    count. A stage is selected as a unit whenever it fits. If a single stage
    exceeds the GPU envelope, serialize its members so it cannot deadlock.
    """
    chains = {}
    for item in members:
        chains.setdefault(item.chain, []).append(item)
    ordered = sorted(chains.values(), key=lambda xs: (
        min(x.deadline for x in xs) < now,
        min(x.deadline for x in xs), min(x.arrival for x in xs)))
    on_time = any(x.deadline >= now for x in members)
    selected, rows, used = set(), [], 0
    for xs in ordered:
        xs = sorted(xs, key=lambda x: x.arrival)
        need = sum(x.blocks(block_size) for x in xs)
        late = min(x.deadline for x in xs) < now
        allowed = not (on_time and late)
        take = allowed and used + need <= capacity and len(selected) + len(xs) <= max_seqs
        chosen = xs if take else []
        if allowed and not selected and not take:
            # Oversized compound: guarantee useful forward progress without
            # ever requesting a larger KV pool or sequence/token budget.
            for x in xs:
                if (not chosen or used + x.blocks(block_size) <= capacity) and len(chosen) < max_seqs:
                    chosen.append(x)
                    used += x.blocks(block_size)
        elif take:
            used += need
        selected.update(x.request for x in chosen)
        rows.append({'chain': xs[0].chain, 'deadline': xs[0].deadline,
                     'late': late, 'reservation_blocks': need,
                     'members': [x.request for x in xs],
                     'selected': [x.request for x in chosen]})
    return selected, rows, used
