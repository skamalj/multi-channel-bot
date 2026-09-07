"""The route ledger - the resolver's memory of its own decisions.

History is a signal, not just a record: a strong prior raises the bar for
switching away from it, which is what stops the route flapping on one
ambiguous sentence.
"""
from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

DecidedBy = Literal[
    "entry", "producer_lookup", "holdings", "history",
    "intent_model", "asked_user", "explicit",
]

# A decision made on weak evidence is one the NEXT turn is allowed to
# overturn and label. A decision the user made explicitly is not - if they
# said "motor" and then said "health", that is two intents, not a mis-route.
WEAK = {"history", "holdings", "intent_model"}

# Asymmetric: cheap to establish a route, expensive to leave one, most
# expensive to leave one with captured work behind it.
T_ESTABLISH = 0.60
T_SWITCH_IDLE = 0.70
T_SWITCH_INFLIGHT = 0.85


class RouteEvent(BaseModel):
    turn: int
    ts: float = Field(default_factory=time.time)
    persona: str
    lob: str | None
    decided_by: DecidedBy
    confidence: float
    evidence: str = ""
    text: str = ""                    # what was said - the training example
    switched_from: str | None = None
    corrected_to: str | None = None   # back-annotated when the NEXT turn
                                      # proves this decision wrong


class RouteLedger(BaseModel):
    events: list[RouteEvent] = Field(default_factory=list)

    @property
    def last(self) -> RouteEvent | None:
        return self.events[-1] if self.events else None

    def prior_lob(self) -> str | None:
        return self.last.lob if self.last else None

    def threshold(self, work_in_flight: bool) -> float:
        if self.prior_lob() is None:
            return T_ESTABLISH
        return T_SWITCH_INFLIGHT if work_in_flight else T_SWITCH_IDLE

    def append(self, ev: RouteEvent) -> None:
        self.events.append(ev)

    def back_annotate(self, actual_lob: str, this_turn: int) -> RouteEvent | None:
        """RS-7. Turn n+1 proved turn n wrong; label turn n.

        Only weak decisions are labelled, and only when the correction is
        immediate. Annotating an explicit choice from six turns ago would
        manufacture training data out of a customer changing their mind -
        which is worse than no training data.
        """
        prev = self.last
        if prev is None or prev.lob == actual_lob:
            return None
        if prev.decided_by not in WEAK:
            return None
        if this_turn - prev.turn > 1:
            return None
        prev.corrected_to = actual_lob
        return prev
