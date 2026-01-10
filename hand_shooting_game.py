import cv2
import mediapipe as mp
import math
import random
import time
import numpy as np

# --- SETTINGS ---
game_duration = 60  # <--- CHANGE YOUR TIME HERE (in seconds)
target_count = 5
difficulty_multiplier = 0.2 # How much faster they get per point

# --- Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
cap = cv2.VideoCapture(0)

# Set Fullscreen
cv2.namedWindow("Pro AR Shooter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pro AR Shooter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Game Variables ---
score = 0
combo = 0
last_hit_time = 0
game_state = "START" 
start_time = 0
targets = [] 
shooting_locked = False

def spawn_target(w, h, current_score):
    # Speed increases based on score
    speed_boost = current_score * difficulty_multiplier
    base_speed = 3
    return {
        "x": random.randint(50, w-50),
        "y": random.randint(100, h-50),
        "dx": random.choice([-1, 1]) * (base_speed + speed_boost),
        "dy": random.choice([-1, 1]) * (base_speed + speed_boost),
        "r": random.randint(25, 40),
        "color": (random.randint(100,255), 0, random.randint(100,255))
    }

def draw_ui_overlay(img, score, time_left, combo):
    h_img, w_img, _ = img.shape
    # Top HUD Bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Score, Timer, and Combo
    cv2.putText(img, f"SCORE: {score}", (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img, f"TIME: {max(0, int(time_left))}s", (w_img//2 - 70, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
    
    if combo > 1:
        cv2.putText(img, f"{combo}X COMBO!", (w_img - 250, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)

# --- Main Loop ---
while cap.isOpened():
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if game_state == "START":
        cv2.putText(img, "AR FINGER GUN", (w//2-250, h//2), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 3)
        cv2.putText(img, f"REACH {game_duration}s LIMIT", (w//2-150, h//2+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "Pinch Thumb to Start", (w//2-160, h//2+120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
    elif game_state == "PLAYING":
        elapsed = time.time() - start_time
        time_left = game_duration - elapsed
        if time_left <= 0: game_state = "GAMEOVER"
        
        # Combo reset logic (reset if no hit in 2 seconds)
        if time.time() - last_hit_time > 2.0:
            combo = 0

        draw_ui_overlay(img, score, time_left, combo)

        # Update and Draw Targets
        for t in targets:
            t['x'] += t['dx']
            t['y'] += t['dy']
            if t['x'] <= 20 or t['x'] >= w-20: t['dx'] *= -1
            if t['y'] <= 90 or t['y'] >= h-20: t['dy'] *= -1
            
            # Draw Target with Glow
            cv2.circle(img, (int(t['x']), int(t['y'])), t['r'], t['color'], 3)
            cv2.circle(img, (int(t['x']), int(t['y'])), 8, (255, 255, 255), -1)

    elif game_state == "GAMEOVER":
        cv2.rectangle(img, (w//2-300, h//2-100), (w//2+300, h//2+150), (0,0,0), -1)
        cv2.putText(img, "MISSION COMPLETE", (w//2-220, h//2-20), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 2)
        cv2.putText(img, f"FINAL SCORE: {score}", (w//2-120, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "Pinch Thumb to Restart", (w//2-150, h//2+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Hand Tracking Logic ---
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            idx_tip = hand_lms.landmark[8]
            thm_tip = hand_lms.landmark[4]
            idx_knl = hand_lms.landmark[5]
            ix, iy = int(idx_tip.x * w), int(idx_tip.y * h)
            
            # Crosshair
            cv2.drawMarker(img, (ix, iy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 25, 2)

            dist = math.sqrt((thm_tip.x - idx_knl.x)**2 + (thm_tip.y - idx_knl.y)**2)
            if dist < 0.05:
                if not shooting_locked:
                    if game_state == "START":
                        game_state = "PLAYING"; start_time = time.time(); score = 0; combo = 0
                        targets = [spawn_target(w, h, score) for _ in range(target_count)]
                    elif game_state == "GAMEOVER":
                        game_state = "START"
                    elif game_state == "PLAYING":
                        # Shoot Visual
                        cv2.circle(img, (ix, iy), 50, (255, 255, 255), 2)
                        hit = False
                        for t in targets[:]:
                            if math.sqrt((ix-t['x'])**2 + (iy-t['y'])**2) < t['r']:
                                targets.remove(t)
                                targets.append(spawn_target(w, h, score))
                                combo += 1
                                score += (10 * combo)
                                last_hit_time = time.time()
                                hit = True
                        if not hit: combo = 0 # Miss resets combo
                    shooting_locked = True
            else:
                shooting_locked = False

    cv2.imshow("Pro AR Shooter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()