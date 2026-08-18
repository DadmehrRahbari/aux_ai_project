# aux_ai_project\aux_ai\sensor_utils.py

def compute_sensor_score(sensor, device_aware=False):
    """
    Compute overall sensor health score in [0,1], higher = healthier.
    """
    imu_score = 1 - min(1.0, sensor.get("imu_var", 0.0) * 10)
    gps_score = 1 - min(1.0, sensor.get("gps_drift", 0.0) * 20)
    cam_score = 1 - min(1.0, sensor.get("cam_blur", 0.0) * 5)

    base_score = (imu_score + gps_score + cam_score) / 3

    if device_aware:
        device_factor = 1.0  # placeholder
        base_score *= device_factor
        base_score = min(1.0, max(0.0, base_score))

    return base_score