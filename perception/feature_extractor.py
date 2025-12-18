import numpy as np
from typing import Optional, Tuple, List, Dict


class FeatureExtractor:
    CLASS_ORDER = ['block', 'coin', 'platform', 'player', 'portal', 'spaceship', 'spike']
    LANDABLE_CLASSES = {'block', 'platform'}

    def __init__(self, screen_width: int, screen_height: int, max_obstacles: int = 6):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_obstacles = max_obstacles
        self.max_distance = np.sqrt(screen_width ** 2 + screen_height ** 2)
        self.prev_player_y: Optional[int] = None

    def extract_player_and_obstacles(self, results):
        class_names_dict = results[0].names
        obstacles = []
        player_pos = None
        player_box = None

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            center_x_int = int((x1 + x2) / 2)
            center_y_int = int((y1 + y2) / 2)
            bounding_box = (x1, y1, x2, y2)
            class_id = int(box.cls[0].cpu().numpy())
            class_name = class_names_dict[class_id]

            if class_id == 3:  # player
                player_pos = (center_x_int, center_y_int)
                player_box = bounding_box
            else:  # obstacles
                obstacles.append((center_x_int, center_y_int, class_name, bounding_box))

        return player_pos, player_box, obstacles

    def calculate_relative_positions(self, player_pos: Tuple[int, int], obstacles: List):
        obstacles_ahead = []

        if player_pos is None:
            return obstacles_ahead

        for obstacle in obstacles:
            obstacle_x, obstacle_y, class_name, bounding_box = obstacle
            dx = obstacle_x - player_pos[0]
            dy = obstacle_y - player_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if dx >= 0:  # Only obstacles ahead
                obstacles_ahead.append((dx, dy, distance, class_name, bounding_box))

        obstacles_ahead.sort(key=lambda x: (x[0], abs(x[1])))
        obstacles_ahead = obstacles_ahead[:self.max_obstacles]
        return obstacles_ahead

    def _normalize_value(self, value: float, max_value: float, allow_negative: bool = False) -> float:
        if max_value == 0:
            return 0.0
        normalized = value / max_value
        if allow_negative:
            normalized = np.clip(normalized, -1.0, 1.0)
        else:
            normalized = np.clip(normalized, 0.0, 1.0)
        return float(normalized)

    def _class_index(self, class_name: Optional[str]) -> int:
        if class_name is None:
            return -1
        try:
            return self.CLASS_ORDER.index(class_name)
        except ValueError:
            return -1

    def _one_hot_class(self, class_name: Optional[str]) -> List[float]:
        vec = [0.0] * len(self.CLASS_ORDER)
        idx = self._class_index(class_name)
        if idx >= 0:
            vec[idx] = 1.0
        return vec

    def _is_on_ground(self, player_box: Optional[Tuple[int, int, int, int]], obstacles: List) -> bool:
        if player_box is None:
            return False
        px1, py1, px2, py2 = player_box
        for _, _, class_name, obox in obstacles:
            if class_name not in self.LANDABLE_CLASSES:
                continue
            ox1, oy1, ox2, _ = obox
            horizontal_overlap = not (px2 < ox1 or px1 > ox2)
            vertical_gap = oy1 - py2
            if horizontal_overlap and -5 <= vertical_gap <= 15:
                return True
        return False

    def _can_land_on_top(self,
                         class_name: Optional[str],
                         player_box: Optional[Tuple[int, int, int, int]],
                         obstacle_box: Optional[Tuple[int, int, int, int]],
                         dy: float) -> float:
        if (class_name not in self.LANDABLE_CLASSES or
                player_box is None or obstacle_box is None or dy <= 0):
            return 0.0
        px1, _, px2, py2 = player_box
        ox1, oy1, ox2, _ = obstacle_box
        horizontal_overlap = not (px2 < ox1 or px1 > ox2)
        vertical_gap = oy1 - py2
        if horizontal_overlap and -5 <= vertical_gap <= 20:
            return 1.0
        return 0.0

    def _collision_risk(self,
                        class_name: Optional[str],
                        distance_norm: float,
                        can_land_on_top: float) -> float:
        base_risk_map = {
            'spike': 1.0,
            'block': 0.7,
            'platform': 0.7,
            'portal': 0.5,
            'spaceship': 0.4,
            'coin': 0.1
        }
        base = base_risk_map.get(class_name, 0.2)
        if can_land_on_top:
            base *= 0.3
        distance_scale = 1.0 - distance_norm
        risk = base * max(distance_scale, 0.0)
        return float(np.clip(risk, 0.0, 1.0))

    def _normalize_player(self,
                          player_pos: Optional[Tuple[int, int]],
                          player_box: Optional[Tuple[int, int, int, int]],
                          obstacles: List) -> Dict[str, float]:
        if player_pos is None:
            self.prev_player_y = None
            return {
                "player_y_norm": 0.0,
                "player_velocity_y_norm": 0.0,
                "on_ground": 0.0,
                "player_detected": 0.0
            }

        player_y_norm = self._normalize_value(player_pos[1], self.screen_height)
        if self.prev_player_y is not None:
            velocity_y = player_pos[1] - self.prev_player_y
            velocity_y_norm = self._normalize_value(velocity_y, self.screen_height, allow_negative=True)
        else:
            velocity_y_norm = 0.0
        self.prev_player_y = player_pos[1]

        on_ground = 1.0 if self._is_on_ground(player_box, obstacles) else 0.0

        return {
            "player_y_norm": player_y_norm,
            "player_velocity_y_norm": velocity_y_norm,
            "on_ground": on_ground,
            "player_detected": 1.0
        }

    def _normalize_obstacles(self,
                             player_box: Optional[Tuple[int, int, int, int]],
                             obstacles_ahead: List) -> List[Dict[str, float]]:
        normalized = []
        for dx, dy, distance, class_name, bounding_box in obstacles_ahead:
            dx_norm = self._normalize_value(dx, self.screen_width)
            dy_norm = self._normalize_value(dy, self.screen_height, allow_negative=True)
            distance_norm = self._normalize_value(distance, self.max_distance)
            class_idx = self._class_index(class_name)
            class_type_norm = self._normalize_value(class_idx if class_idx >= 0 else 0,
                                                    len(self.CLASS_ORDER) - 1 or 1)
            one_hot = self._one_hot_class(class_name)
            can_land = self._can_land_on_top(class_name, player_box, bounding_box, dy)
            collision_risk = self._collision_risk(class_name, distance_norm, can_land)

            normalized.append({
                "dx_norm": dx_norm,
                "dy_norm": dy_norm,
                "distance_norm": distance_norm,
                "class_type_norm": class_type_norm,
                "class_one_hot": one_hot,
                "can_land_on_top": can_land,
                "collision_risk": collision_risk,
                "class_name": class_name,
                "bounding_box": bounding_box
            })

        while len(normalized) < self.max_obstacles:
            normalized.append({
                "dx_norm": 0.0,
                "dy_norm": 0.0,
                "distance_norm": 0.0,
                "class_type_norm": 0.0,
                "class_one_hot": [0.0] * len(self.CLASS_ORDER),
                "can_land_on_top": 0.0,
                "collision_risk": 0.0,
                "class_name": None,
                "bounding_box": None
            })

        return normalized

    def _next_obstacle_distance(self, obstacles_ahead: List, target_classes: set) -> float:
        for dx, _, _, class_name, _ in obstacles_ahead:
            if class_name in target_classes:
                return self._normalize_value(dx, self.screen_width)
        return 1.0

    def _compute_environment_features(self,
                                      player_pos: Optional[Tuple[int, int]],
                                      obstacles_ahead: List) -> Dict[str, float]:
        if player_pos is None:
            return {
                "ground_distance_norm": 0.0,
                "next_spike_distance_norm": 1.0,
                "next_platform_distance_norm": 1.0
            }

        ground_distance = self.screen_height - player_pos[1]
        ground_distance_norm = self._normalize_value(ground_distance, self.screen_height)

        next_spike_distance_norm = self._next_obstacle_distance(obstacles_ahead, {'spike'})
        next_platform_distance_norm = self._next_obstacle_distance(obstacles_ahead, {'platform', 'block'})

        return {
            "ground_distance_norm": ground_distance_norm,
            "next_spike_distance_norm": next_spike_distance_norm,
            "next_platform_distance_norm": next_platform_distance_norm
        }

    def _build_state_vector(self,
                            player_features: Dict[str, float],
                            obstacle_features: List[Dict[str, float]],
                            env_features: Dict[str, float]) -> np.ndarray:
        state_values: List[float] = [
            player_features["player_y_norm"],
            player_features["player_velocity_y_norm"],
            player_features["on_ground"],
            env_features["ground_distance_norm"],
            env_features["next_spike_distance_norm"],
            env_features["next_platform_distance_norm"],
        ]

        for obstacle in obstacle_features:
            state_values.extend([
                obstacle["dx_norm"],
                obstacle["dy_norm"],
                obstacle["distance_norm"],
                obstacle["class_type_norm"],
                obstacle["can_land_on_top"],
                obstacle["collision_risk"],
            ])
            state_values.extend(obstacle["class_one_hot"])

        return np.array(state_values, dtype=np.float32)

    def extract(self, results):
        player_pos, player_box, obstacles = self.extract_player_and_obstacles(results)
        obstacles_ahead = self.calculate_relative_positions(player_pos, obstacles)
        player_features = self._normalize_player(player_pos, player_box, obstacles)
        obstacle_features = self._normalize_obstacles(player_box, obstacles_ahead)
        environment_features = self._compute_environment_features(player_pos, obstacles_ahead)
        state_vector = self._build_state_vector(player_features, obstacle_features, environment_features)

        normalized_features = {
            "player": player_features,
            "obstacles": obstacle_features,
            "environment": environment_features,
            "state_vector": state_vector
        }

        return player_pos, obstacles_ahead, normalized_features
