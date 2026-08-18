import numpy as np
import params

class SFCMonitor:
    @staticmethod
    def check_spatial_consistency(mask):
        h, w = mask.shape
        total_pixels = h * w
        road_pixels = np.sum(mask > 0)
        
        # 1. Area Check using existing limit
        area_ratio = road_pixels / total_pixels
        area_fault = 1.0 if area_ratio > params.APSS_AREA_LIMIT else 0.0
        
        # 2. Position Check using existing limit
        y_coords, _ = np.where(mask > 0)
        if len(y_coords) > 0:
            centroid_y = np.mean(y_coords) / h
            # If road is too high (lower value means higher in image)
            pos_fault = 1.0 if centroid_y < params.APSS_POS_Y_LIMIT else 0.0
        else:
            pos_fault = 1.0
            
        # 3. Symmetry Check using existing max
        left_half = mask[:, :w//2]
        right_half = np.flip(mask[:, w//2:], axis=1)
        min_w = min(left_half.shape[1], right_half.shape[1])
        sym_diff = np.abs(np.sum(left_half[:, :min_w]) - np.sum(right_half[:, :min_w]))
        sym_score = sym_diff / (road_pixels + 1e-6)
        sym_fault = 1.0 if sym_score > params.APSS_SYMMETRY_MAX else 0.0
        
        return max(area_fault, pos_fault, sym_fault)