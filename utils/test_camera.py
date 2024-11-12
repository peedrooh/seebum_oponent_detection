def test_camera(period):
    """
        This function takes 10 picures every [period] seconds
    """

    from time import sleep
    from picamera2 import Picamera2, Preview
    cam = Picamera2();
    camera_config = cam.create_preview_configuration();
    cam.configure(camera_config);
    for i in range(10):
        cam.start();
        sleep(period);
        cam.capture_file(f"./assests/photos/photo_{i}.jpg");

if __name__ == "__main__":
    test_camera(2);
