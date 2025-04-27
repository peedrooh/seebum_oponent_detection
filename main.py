import atexit
import serial
import time

from picamera2 import Picamera2

from ultralytics import YOLO
import cv2 

cam = None
ser = None

def cleanup():
    print("\nCleaning up resources before exiting...")
    if cam:
        try:
            print("Stopping camera.")
            cam.stop()
        except Exception as e:
            print(f"Error stopping camera: {e}")
    else:
        print("Camera not initialized.")

    if ser:
        try:
            print("Flushing serial port.")
            ser.flush()
            print("Closing serial port.")
            ser.close()
        except Exception as e:
            print(f"Error closing serial port: {e}")
    else:
        print("Serial port not initialized.")
    print("Finished cleaning resources.\n")

if __name__ == "__main__":
    print("\nStart configuring camera...")
    try:
        cam = Picamera2()
        camera_config = cam.create_preview_configuration(main={"size": (640, 480)})  # Smaller = faster
        cam.configure(camera_config)
        cam.start()
        print("Finished configuring camera.\n\n")
    except Exception as e:
        print(f"Failed to configure camera: {e}")
        cam = None

    print("\nStarting YOLO model...")
    model = YOLO('/home/pedro/Pessoal/code/seebum_oponent_detection/runs/detect/train5/weights/last.pt')
    print("Finished loading YOLO model.\n\n")

    print("\nStarting configuring serial communication...")
    try:
        ser = serial.Serial("/dev/ttyS0", 115200)
        print("Finished configuring serial communication.\n\n")
    except serial.SerialException as e:
        print(f"Failed to initialize serial communication: {e}")
        ser = None

    atexit.register(cleanup)

    try:
        while True:
            start_time = time.time()
            frame = cam.capture_array()

            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            results = model.predict(frame, imgsz=(640, 480), device='cpu', verbose=False)[0].boxes.data.tolist()

            if len(results) == 0:
                continue

            x1, y1, x2, y2, score, class_id = results[0]
            width = int(x2 - x1)
            height = int(y2 - y1)
            x_center = x1 + width / 2
            y_center = y1 + height / 2

            # message = f"x_center: {int(x_center)}, y_center: {int(y_center)}, w: {width}, h: {height}\n"
            # print(message)

            end_time = time.time()  # End measuring time for the cycle
            cycle_time = end_time - start_time  # Calculate elapsed time
            print(f"Cycle time: {cycle_time:.4f} seconds")
            if ser:
                message = f"x_center: {int(x_center)}, y_center: {int(y_center)}, w: {width}, h: {height}\n"
                ser.write(message.encode())

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")


