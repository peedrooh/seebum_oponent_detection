from time import sleep

from picamera2 import Picamera2, Preview

from ultralytics import YOLO

cam = Picamera2();
camera_config = cam.create_preview_configuration();
cam.configure(camera_config);

model = YOLO('./runs/detect/train5/weights/last.pt');

if __name__ == "__main__":
    while(1):
        cam.start();
        cam.capture_file("./assets/test_file.jpg");
        results = model("./assets/test_file.jpg")[0];
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            print("Score: ", score);
