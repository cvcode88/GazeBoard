import time

backspace_rate = 0.0        # fraction of recent keystrokes that were backspaces (0.0 - 1.0)
avg_letters_per_10s = 0.0   # typing speed: letters typed in the last 10 seconds
dwell_time = 0.7            # current dwell time (700 milliseconds)
travel_time = 0.3           # time spent moving gaze/eyes between targets - real overhead
                            # a real person can never eliminate, so the achievable-speed
                            # ceiling must include it, not just dwell_time alone

def adaptive_algorithm(backspace_rate, avg_letters_per_10s, dwell_time, travel_time, min_dwell=0.35, max_dwell=1.4, max_step=0.06):
    max_achievable = 10 / (dwell_time + travel_time)
    utilization = min(1, avg_letters_per_10s / max_achievable)
    performance_score = max(0, min(1, utilization - backspace_rate))
    ideal_dwell = max_dwell - performance_score * (max_dwell - min_dwell)
    step = max(-max_step, min(max_step, ideal_dwell - dwell_time))
    return dwell_time + step


while True:
    dwell_time = adaptive_algorithm(backspace_rate, avg_letters_per_10s, dwell_time, travel_time)
    print(dwell_time)
    time.sleep(10)
