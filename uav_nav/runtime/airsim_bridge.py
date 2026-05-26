"""AirSim bridge for hardware-in-the-loop simulation.

Connects the NavigationPipeline to a running AirSim instance,
reading sensor data and writing velocity commands in real time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uav_nav.runtime.pipeline import NavigationPipeline


@dataclass
class AirSimConfig:
    """Connection parameters for the AirSim bridge.

    Attributes:
        host: AirSim RPC host address.
        port: AirSim RPC port.
        vehicle_name: Vehicle name configured in AirSim settings.json.
        camera_name: Camera name for image requests.
        use_stereo: Whether to request stereo camera pairs.
        control_mode: "velocity" or "position" command mode.
        command_rate_hz: Rate at which to send commands to AirSim.
    """

    host: str = "127.0.0.1"
    port: int = 41451
    vehicle_name: str = "UAV"
    camera_name: str = "front_center"
    use_stereo: bool = False
    control_mode: str = "velocity"
    command_rate_hz: float = 10.0


class AirSimBridge:
    """Real-time bridge between uav_nav pipeline and AirSim simulator.

    Args:
        pipeline: Initialised NavigationPipeline to drive.
        config: AirSimConfig with connection parameters.
    """

    def __init__(
        self,
        pipeline: NavigationPipeline,
        config: Optional[AirSimConfig] = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or AirSimConfig()
        self._client: Optional[object] = None
        self._running: bool = False

    def connect(self) -> None:
        """Connect to a running AirSim instance via its RPC API.

        Raises:
            NotImplementedError: Not yet implemented.
            ConnectionError: If AirSim is not reachable at the configured host/port.
        """
        raise NotImplementedError("AirSimBridge.connect is not yet implemented.")

    def disconnect(self) -> None:
        """Disconnect from AirSim and release resources.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("AirSimBridge.disconnect is not yet implemented.")

    def run(self, max_steps: Optional[int] = None) -> None:
        """Enter the main sensing/control loop.

        Reads sensor data from AirSim, processes it through the pipeline,
        and sends back velocity commands.

        Args:
            max_steps: Maximum number of control steps. None = run until
                interrupted or mission complete.

        Raises:
            NotImplementedError: Not yet implemented.
            RuntimeError: If ``connect()`` has not been called.
        """
        raise NotImplementedError("AirSimBridge.run is not yet implemented.")

    def _get_sensor_data(self) -> dict[str, object]:
        """Retrieve current sensor readings from AirSim.

        Returns:
            Dictionary with keys: "image_rgb", "image_right" (optional),
            "imu_acc", "imu_gyro", "gnss_position" (optional), "timestamp".

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("AirSimBridge._get_sensor_data is not yet implemented.")

    def _send_velocity_command(self, velocity_ned: np.ndarray) -> None:
        """Send a NED velocity command to AirSim.

        Args:
            velocity_ned: Desired velocity in NED frame, shape (3,), m/s.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("AirSimBridge._send_velocity_command is not yet implemented.")
