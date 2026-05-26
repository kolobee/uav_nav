"""Mission mode state machine for UAV operations.

Defines the high-level mission modes (NOMINAL, GNSS_DEGRADED, GNSS_LOST,
REJOIN, EMERGENCY_LAND) and the state machine transitions between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np


class MissionMode(Enum):
    """High-level mission operating modes.

    Attributes:
        NOMINAL: Full GNSS + VIO, normal path following.
        GNSS_DEGRADED: GNSS available but degraded; increased reliance on VIO.
        GNSS_LOST: No GNSS; purely VIO + TILM landmark navigation.
        REJOIN: Executing re-join manoeuvre to return to planned path.
        EMERGENCY_LAND: Critical failure; initiate safe landing.
        HOLD: Hover in place (e.g. waiting for GNSS recovery).
    """

    NOMINAL = auto()
    GNSS_DEGRADED = auto()
    GNSS_LOST = auto()
    REJOIN = auto()
    EMERGENCY_LAND = auto()
    HOLD = auto()


@dataclass
class MissionStatus:
    """Snapshot of the current mission state.

    Attributes:
        mode: Active mission mode.
        gnss_accuracy: Current GNSS horizontal accuracy in metres.
        tilm_quality: Current TILM matching quality [0, 1].
        time_since_gnss: Seconds since last valid GNSS fix.
        position_uncertainty: Estimated position uncertainty 1-sigma (m).
        path_deviation: Cross-track deviation from planned path (m).
        battery_fraction: Battery state of charge [0, 1].
        altitude_agl: Altitude above ground level in metres.
    """

    mode: MissionMode = MissionMode.NOMINAL
    gnss_accuracy: float = 2.0
    tilm_quality: float = 1.0
    time_since_gnss: float = 0.0
    position_uncertainty: float = 0.5
    path_deviation: float = 0.0
    battery_fraction: float = 1.0
    altitude_agl: float = 50.0


@dataclass
class TransitionRule:
    """A single state machine transition rule.

    Attributes:
        from_mode: Source mode (None matches any).
        to_mode: Target mode.
        condition: Callable taking MissionStatus, returning bool.
        priority: Higher priority rules are checked first.
        description: Human-readable description of the trigger.
    """

    from_mode: Optional[MissionMode]
    to_mode: MissionMode
    condition: Callable[[MissionStatus], bool]
    priority: int = 0
    description: str = ""


class MissionManager:
    """State machine that manages mission mode transitions.

    Evaluates transition rules each step and fires transitions when
    conditions are met. Notifies registered callbacks on mode changes.

    Args:
        initial_mode: Starting mission mode.
        gnss_lost_timeout: Seconds without GNSS to trigger GNSS_LOST.
        gnss_degraded_threshold: GNSS accuracy (m) above which DEGRADED fires.
        emergency_uncertainty: Position uncertainty (m) triggering EMERGENCY.
    """

    def __init__(
        self,
        initial_mode: MissionMode = MissionMode.NOMINAL,
        gnss_lost_timeout: float = 5.0,
        gnss_degraded_threshold: float = 10.0,
        emergency_uncertainty: float = 100.0,
    ) -> None:
        self.gnss_lost_timeout = gnss_lost_timeout
        self.gnss_degraded_threshold = gnss_degraded_threshold
        self.emergency_uncertainty = emergency_uncertainty
        self._status = MissionStatus(mode=initial_mode)
        self._rules: list[TransitionRule] = []
        self._callbacks: list[Callable[[MissionMode, MissionMode], None]] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register the default set of mission state machine transitions.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError(
            "MissionManager._register_default_rules is not yet implemented."
        )

    def update(self, status: MissionStatus) -> MissionMode:
        """Evaluate all rules and apply any triggered transition.

        Args:
            status: Current system status snapshot.

        Returns:
            Active mission mode after potential transition.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("MissionManager.update is not yet implemented.")

    def add_rule(self, rule: TransitionRule) -> None:
        """Register a custom transition rule.

        Args:
            rule: TransitionRule to add (inserted by priority).
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def on_transition(
        self, callback: Callable[[MissionMode, MissionMode], None]
    ) -> None:
        """Register a callback invoked on every mode transition.

        Args:
            callback: Function(old_mode, new_mode) called on transitions.
        """
        self._callbacks.append(callback)

    def force_mode(self, mode: MissionMode) -> None:
        """Force a specific mode regardless of rule conditions.

        Triggers transition callbacks as usual.

        Args:
            mode: Target MissionMode to activate.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("MissionManager.force_mode is not yet implemented.")

    @property
    def current_mode(self) -> MissionMode:
        """Currently active mission mode."""
        return self._status.mode

    @property
    def status(self) -> MissionStatus:
        """Current mission status snapshot."""
        return self._status
