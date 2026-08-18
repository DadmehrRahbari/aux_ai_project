# aux_ai_project/aux_ai/feature_extractor.py

import numpy as np
import params

EMBED_DIM = 525
NUM_CLASSES = 5

def feature_dim():
    """
    Returns the exact dimensionality of the Aux-AI input feature vector.
    """
    return (
        EMBED_DIM          # Big-AI embedding
        + 4                # bbox
        + NUM_CLASSES      # class probabilities
        + 1                # confidence
        + 3                # sensor values (imu, gps, blur)
    )


def build_feature_vector(detection, sensor_vals):
    """
    Construct deterministic Aux-AI input feature vector.
    """

    embedding = detection["embedding"]
    bbox = detection["bbox"]
    class_probs = detection["class_probs"]
    confidence = np.array([detection["confidence"]])

    sensor_features = np.array([
        sensor_vals["imu_var"],
        sensor_vals["gps_drift"],
        sensor_vals["cam_blur"]
    ])

    # APSS Logic: Single Frame Consistency (SFC)
    # metadata proxies: bbox[2]*bbox[3] (Area), bbox[1] (Y-Position)
    area_val = bbox[2] * bbox[3]
    pos_y_val = bbox[1]

    # Calculate Geometric Violation Score
    sfc_score = 1.0 if (area_val > params.APSS_AREA_LIMIT or pos_y_val > params.APSS_POS_Y_LIMIT) else 0.0

    feature_vec = np.concatenate([
        embedding,
        bbox,
        class_probs,
        confidence,
        sensor_features,
        np.array([sfc_score]) # Explicit APSS Spatial Signal
    ])

    return feature_vec