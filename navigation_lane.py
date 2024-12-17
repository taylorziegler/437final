import picar_4wd as fc
import math
from Astar import A_star, visualize_path
import time
import cv2 
import numpy as np
from picamera2 import Picamera2
from hough_lane_detector import HoughDetector

def left_90(dis):
    fc.turn_left(10)
    time.sleep(0.66)
    fc.forward(3)
    time.sleep(0.12 * dis * 0.33)

def left_45(dis):
    fc.turn_left(10)
    time.sleep(0.33)
    fc.forward(3)
    time.sleep(0.12 * dis * 0.33)

def forward_0(dis):
    fc.forward(3)
    time.sleep(0.12 * dis * 0.33)

def right_45(dis):
    fc.turn_right(10)
    time.sleep(0.33)
    fc.forward(3)
    time.sleep(0.12 * dis * 0.33)

def right_90(dis):
    fc.turn_right(10)
    time.sleep(0.66)
    fc.forward(3)
    time.sleep(0.12 * dis * 0.33)

action_dict = {
    -2: left_90,
    -1: left_45,
    0: forward_0,
    1: right_45,
    2: right_90
}

def calculate_servo_angle(x, y):
    """
    Calculate the angle to the target point (x, y) relative to the car.
    :param x: Horizontal coordinate relative to the car (forward is positive x).
    :param y: Vertical coordinate relative to the car (right is positive y).
    :return: Angle in degrees for the servo motor.
    """
    angle_rad = math.atan2(y, x)  # atan2 automatically handles quadrants
    angle_deg = math.degrees(angle_rad)  # Convert radians to degrees
    
    # Convert angle to servo range (0° to 180°)
    servo_angle = 90 + angle_deg  # 90° is straight ahead, adjust for range
    servo_angle = max(0, min(180, servo_angle))  # Clamp between 0° and 180°
    return servo_angle

def point_ultrasonic_to_target(x, y):
    """
    Point the ultrasonic sensor towards the target (x, y) coordinate.
    :param x: Horizontal coordinate relative to the car.
    :param y: Vertical coordinate relative to the car.
    """
    servo_angle = calculate_servo_angle(x, y)
    print(f"Calculated servo angle: {servo_angle:.2f} degrees")

    # Point the servo to the desired angle
    fc.set_angle = servo_angle
    time.sleep(0.5)  # Allow the servo to stabilize

    # Measure the distance using the ultrasonic sensor
    distance = fc.get_distance()  # Use Picar 4WD ultrasonic sensor function
    print(f"Distance to target: {distance:.2f} cm")
    return distance

def check_ultrasonic_obstacle(x, y, tolerance=5):
    """
    Check for obstacles using the ultrasonic sensor within a given tolerance (in centimeters).
    :param tolerance: Distance threshold to consider the spot occupied.
    :return: True if an obstacle is detected within the tolerance, otherwise False.
    """
    point_ultrasonic_to_target(x, y)
    obst_distance = fc.get_distance()
    target_distance = math.sqrt(x**2, y**2)

    if (obst_distance + tolerance) > target_distance and target_distance > (obst_distance - tolerance):
        print("Obstacle detected! Spot is occupied.")
        return True
    else:
        print("No obstacle detected. Spot is free.")
        return False


def navigate(rescan_step=25):
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1640, 1232)})
    picam2.configure(config)
    picam2.start()
    
    points = []
    for i in range(30):
        frame = picam2.capture_array()
        image = cv2.flip(frame, 0)
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        my_detector = HoughDetector()
        output_image = my_detector.process_image(image)
        target_point = my_detector.projection_2_ground()
        if target_point is not None:
            points.append(target_point)

    x, y = np.mean(np.array(points), axis=0)
    y = min(max(0, 99 - y * 100), 99)
    x = min(max(0, 55 + x * 100), 99)

    origin = (99, 49)
    target = (int(x), int(y))
    print(f"Target coordinates: {target}")

    # Use the ultrasonic sensor to check for obstacles at the target spot
    if check_ultrasonic_obstacle(target[0], target[1], tolerance=5):
        print("Target spot is full. Recalculating...")
        return  # Exit or handle re-navigation

    surrounding_map = np.zeros((100, 100), np.uint8)
    path = A_star(surrounding_map, origin, target)
    if not path:
        print("No path found. Exiting navigation.")
        return

    current_row, current_col = origin
    current_direction = 2
    fc.forward(-3)
    reach_target = False
    error = False

    while True:
        if error or reach_target or len(path) <= 4:
            break
        for next_i in range(1, len(path)):
            next_row, next_col = path[next_i]

            if next_row == current_row and next_col == current_col - 1:  # Full left
                next_direction = 0
                dis = 1
            elif next_row == current_row - 1 and next_col == current_col - 1:  # Diag left
                next_direction = 1
                dis = 1.5
            elif next_row == current_row - 1 and next_col == current_col:  # Up
                next_direction = 2
                dis = 1
            elif next_row == current_row - 1 and next_col == current_col + 1:  # Diag right
                next_direction = 3
                dis = 1.5
            elif next_row == current_row and next_col == current_col + 1:  # Full right
                next_direction = 4
                dis = 1
            else:
                print("Error: Invalid next direction")
                error = True
                break

            direction_diff = next_direction - current_direction
            if direction_diff in action_dict:
                action_dict[direction_diff](dis)
                current_row = next_row
                current_col = next_col
                current_direction = next_direction
            else:
                print("Error: Invalid direction change")
                error = True
                break

            if (current_row, current_col) == target:
                reach_target = True
                break
    
    fc.stop()
    picam2.stop()
    print("Navigation complete.")


def main():
    navigate(rescan_step=50)

if __name__ == "__main__":
    try: 
        main()
    finally: 
        fc.stop()
