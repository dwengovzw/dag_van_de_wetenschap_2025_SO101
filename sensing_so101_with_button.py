from lerobot.robots.so101_follower import SO101Follower
import serial
import time


class SO101FollowerWithTouch(SO101Follower):
    def __init__(self, config, sensor_serial_port=None):
        super().__init__(config)
        self.sensor_serial = None
        if sensor_serial_port is not None:
            self.sensor_serial = serial.Serial(sensor_serial_port, 115200, timeout=0.1)
            time.sleep(2)  # allow Arduino or similar to reset

        # Must use float type to be compatible with hw_to_dataset_features
        self.observation_features.update({
            "touch_sensor_value": float,
            "button_state": float,
        })

    def get_observation(self):
        obs = super().get_observation()

        touch_value = 0
        button_state = 0

        if self.sensor_serial:
            try:
                line = self.sensor_serial.readline().decode("utf-8").strip()
                if line:
                    parts = dict(p.split(":") for p in line.split(","))
                    touch_value = float(parts.get("touchvalue", 0))
                    button_state = int(parts.get("button", 0))
            except Exception as e:
                print(f"Sensor read error: {e}")

        obs["touch_sensor_value"] = float(touch_value)
        obs["button_state"] = float(button_state)

        return obs
