import atexit
import serial

from picamera2 import Picamera2

from ultralytics import YOLO


cam = None;
ser = None;


def cleanup():
    print("\nCleaning up resources before exiting...");
    if cam:
        try:
            print("Stopping camera.");
            cam.stop();
        except Exception as e:
            print(f"Error stopping camera: {e}");
    else:
        print("Camera not initialized.");
    if ser:
        try:
            print("Flushing serial port.");
            ser.flush();
            print("Closing serial port.");
            ser.close();
        except Exception as e:
            print(f"Error closing serial port: {e}");
    else:
        print("Serial port not initialized.");
    print("Finished cleaning resources.\n");


if __name__ == "__main__":
    print("\nStart configuring camera...");
    try:
        cam = Picamera2();
        camera_config = cam.create_preview_configuration();
        cam.configure(camera_config);
        print("Finished configuring camera.\n\n");
    except Exception as e:
        print(f"Failed to configure camera: {e}");
        cam = None;

    print("\nStarting YOLO model...");
    model = YOLO('./runs/detect/train5/weights/last.pt');
    print("Finished loading YOLO model.\n\n");

    print("\nStarting configuring serial communication");
    try:
        ser = serial.Serial("/dev/ttyS0", 115200);
        print("Finished configuring serial communication.\n\n");
    except serial.SerialException as e:
        print(f"Failed to initialize serial communication: {e}");
        ser = None

    atexit.register(cleanup);

    try:
        while(1):
            cam.start();
            cam.capture_file("./assets/test_file.jpg");
            results = model("./assets/test_file.jpg")[0].boxes.data.tolist();

            if(len(results) == 0): continue;

            x1, y1, x2, y2, score, class_id = results[0];
            width = int(x2 - x1);
            height = int(y2 - y1);
            x_center = x1 + width / 2;
            y_center = y1 + height / 2;
            bytes_writen = ser.write(f"x_center: {int(x_center)}, y_center: {int(y_center)}, w: {width}, h: {height}\n".encode());

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...");

