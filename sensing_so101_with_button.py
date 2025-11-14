from lerobot.robots.so101_follower import SO101Follower
import serial  # if you're reading sensors over serial, e.g., Arduino Nano
import time

import rerun as rr

class SO101FollowerWithTouch(SO101Follower):
    def __init__(self, config, sensor_serial_port=None):
        super().__init__(config)
        self.sensor_serial = None
        if sensor_serial_port is not None:
            self.sensor_serial = serial.Serial(sensor_serial_port, 115200, timeout=0.1)
            time.sleep(2)  # allow Arduino or similar to reset

        # Extend observation feature definitions for dataset
        self.observation_features.update({
            "touch_sensor_value": {"shape": (1,), "dtype": "int32"},
            "button_state": {"shape": (1,), "dtype": "int32"},
        })

    def get_observation(self):
        # Get the standard robot observation first
        obs = super().get_observation()

        # Default values (if no serial connection)
        touch_value = 0
        button_state = 0

        # Try to read from serial if available
        if self.sensor_serial:
            try:
                line = self.sensor_serial.readline().decode("utf-8").strip()
                if line:
                    print(line)
                    # Expect something like: "touch:0.42,button:1"
                    parts = dict(p.split(":") for p in line.split(","))
                    touch_value = float(parts.get("touchvalue", 0))
                    button_state = int(parts.get("button", 0))
            except Exception as e:
                print(f"Sensor read error: {e}")

        # Add the custom sensor readings
        obs["touch_sensor_value"] = [touch_value]
        obs["button_state"] = [button_state]
        
        # ✅ Log custom sensor data to Rerun
        rr.log("sensors/touch/value", rr.Scalars([touch_value]))
        rr.log("sensors/button/state", rr.Scalars([button_state]))

        return obs
