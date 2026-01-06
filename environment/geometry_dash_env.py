import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2 as cv
from gymnasium.utils.env_checker import check_env
from perception.screen_capture import ScreenCapture
from perception.detector import Detector
from perception.feature_extractor import FeatureExtractor
from control.action_executor import ActionExecutor

class GeometryDashEnv(gym.Env):    
    # metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(GeometryDashEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(84,), 
            dtype=np.float32
        ) # numpy array of shape (84,)
       
        print("Initializing Geometry Dash Environment...")
        monitor = {"top": 70, "left": 85, "width": 1295, "height": 810}
        self.cap = ScreenCapture(monitor_region=monitor, target_fps=60, resize=None, normalize=False, grayscale=False)
        self.detector = Detector(classes=[3, 6])
        self.extractor = FeatureExtractor(monitor['width'], monitor['height'])
        self.executor = ActionExecutor()
        self.current_state = None
        self.last_frame = None
        self.attempts = 0
        self.steps = 0 
        self.episode_reward = 0.0
        # self.last_infer_ms = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # when dying the game will show the menu screen press space to restart the level
        self.executor.restart_level()
        self.attempts += 1
        self.steps = 0 
        self.episode_reward = 0.0

        # Retry logic to ensure player is detected after restart
        max_retries = 10
        for attempt in range(max_retries):
            frame = self.cap.capture_and_preprocess()
            self.last_frame = frame
            detections = self.detector.detect(frame)
            player_pos, obstacles_ahead, normalized_features = self.extractor.extract(detections)
            
            # If player is detected, proceed normally
            if player_pos is not None:
                break
                
            # Small delay before retry to let game stabilize
            time.sleep(0.1)
            
            # If we've exhausted retries, use the last attempt anyway
            if attempt == max_retries - 1:
                print(f"Warning: Player not detected after {max_retries} attempts in reset()")
        
        observation = normalized_features["state_vector"] # initial state vector 
        self.current_state = observation # used in the step method later
        info = {"player_pos": player_pos, "obstacles_ahead": obstacles_ahead, "features": normalized_features}
        return observation, info


    def step(self, action):
        self.steps += 1
        self.executor.act(action) # perform the action
        time.sleep(0.05)  # Wait 50ms for death menu to fully appear if dying
        frame = self.cap.capture_and_preprocess() # next frame to detect the obstacles
        self.last_frame = frame

        # start_time = time.time()
        detections = self.detector.detect(frame) # detect the obstacles in the next frame
        # infer_ms = (time.time() - start_time) * 1000.0
        # self.last_infer_ms = infer_ms
        # print(f"YOLO inference {infer_ms:.1f}ms")
       

        player_pos, obstacles_ahead, normalized_features = self.extractor.extract(detections) # extract the features from the next frame

        observation = normalized_features["state_vector"] # next state vector
        self.current_state = observation # new state vector

        info = {"player_pos": player_pos, "obstacles_ahead": obstacles_ahead, "features": normalized_features}
        # check if the player is dead or we reached the end of the level and update the reward
        
        # if player_pos is None:
        #     # Check if this is just a temporary detection failure
        #     # Try one more frame capture before declaring death
        #     retry_frame = self.cap.capture_and_preprocess()
        #     retry_detections = self.detector.detect(retry_frame)
        #     retry_player_pos, _, _ = self.extractor.extract(retry_detections)
            
        #     if retry_player_pos is not None:
        #         # player is actually alive, use retry data
        #         terminated = False
        #         player_pos = retry_player_pos
        #         self.last_frame = retry_frame
        #         _, obstacles_ahead, normalized_features = self.extractor.extract(retry_detections)
        #         observation = normalized_features["state_vector"]
        #         self.current_state = observation
        #         info = {"player_pos": player_pos, "obstacles_ahead": obstacles_ahead, "features": normalized_features}
        #     else:
        #         terminated = True # player is actually dead
        # else:
        #     terminated = False
        terminated = self.check_death(frame)
        truncated = False # we don't need truncate for this environment
        if terminated:
            reward = 0.0
            self.executor.act(0) 
        else:   
            reward = 1.0

            
        
        self.episode_reward += reward
        if terminated:
            print(f"Episode reward: {self.episode_reward:.2f}")
        return observation, reward, terminated, truncated, info # gymnasium format


    def check_death(self, frame):
        menu_template = cv.imread("data/images/menu.png", 0)
        restart_template = cv.imread("data/images/restart.png", 0)
        
        if menu_template is None or restart_template is None:
            return False
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        
        best_menu_match = 0
        best_restart_match = 0
        
        for scale in scales:
            # Check menu template at this scale
            resized_menu = cv.resize(menu_template, None, fx=scale, fy=scale)
            if resized_menu.shape[0] <= gray_frame.shape[0] and resized_menu.shape[1] <= gray_frame.shape[1]:
                menu_match = cv.matchTemplate(gray_frame, resized_menu, cv.TM_CCOEFF_NORMED)
                _, menu_max_val, _, _ = cv.minMaxLoc(menu_match)
                best_menu_match = max(best_menu_match, menu_max_val)
            
            # Check restart template at this scale
            resized_restart = cv.resize(restart_template, None, fx=scale, fy=scale)
            if resized_restart.shape[0] <= gray_frame.shape[0] and resized_restart.shape[1] <= gray_frame.shape[1]:
                restart_match = cv.matchTemplate(gray_frame, resized_restart, cv.TM_CCOEFF_NORMED)
                _, restart_max_val, _, _ = cv.minMaxLoc(restart_match)
                best_restart_match = max(best_restart_match, restart_max_val)
        
        threshold = 0.6
        is_dead = best_menu_match > threshold or best_restart_match > threshold
        if is_dead or self.steps % 50 == 0:
            print(f"Menu match: {best_menu_match:.2f}, Restart match: {best_restart_match:.2f}")
        return is_dead
    
    def render(self):
        pass

    def close(self):
        # close the screen capture
        if hasattr(self, 'cap') and hasattr(self.cap, 'sct'):
            self.cap.sct.close()


if __name__ == "__main__":
    gym.register(
        id='GeometryDash-v0',
        entry_point=GeometryDashEnv,
        max_episode_steps = 20000,
    )
    print("Registered GeometryDash-v0")

    env = gym.make('GeometryDash-v0')
    print("Created environment")
    
    try:
        check_env(env.unwrapped)
        print("Environment is valid")
    except Exception as e:
        print(f"Environment is invalid: {e}")
        exit()
    
    print("\n--- TEST: PLAYING 50 FRAMES ---")
    print("Game starting in 2 seconds")
    time.sleep(2)

    raw_env = env.unwrapped
    writer = cv.VideoWriter(
        'env_random_policy.mp4',
        cv.VideoWriter_fourcc(*'mp4v'),
        30.0,
        (1295, 810)
    )

    try:
        observation, info = env.reset()
        for i in range(50):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}: Action={action}, Reward={reward}, Dead={terminated}, Attempts={raw_env.attempts}")

            overlay = raw_env.last_frame.copy()

            text1 = f"Step {i} | Action {action} | Reward {reward:.1f}"
            text2 = f"Cumulative {raw_env.episode_reward:.1f} | Terminated {terminated}"
            text3 = f"Attempt {raw_env.attempts} | Obstacles {len(info['obstacles_ahead'])}"
            cv.putText(overlay, text1, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(overlay, text2, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.putText(overlay, text3, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            for idx, obstacle in enumerate(info["obstacles_ahead"]):
                _, _, _, cls_name, bbox = obstacle
                if bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(
                    overlay,
                    f"{cls_name}",
                    (x1, max(y1 - 5, 15)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            if info["player_pos"] is not None:
                px, py = info["player_pos"]
                cv.circle(overlay, (int(px), int(py)), 6, (0, 0, 255), -1)
                cv.putText(
                    overlay,
                    "player",
                    (int(px) - 20, int(py) - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            writer.write(overlay)

            if terminated:
                observation, info = env.reset()
            
        print("Game completed successfully")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        writer.release()
        env.close()    
