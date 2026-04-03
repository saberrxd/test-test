import io
import sys
import unittest
from unittest.mock import patch

import smart_capture


class FakeCV2:
    def __init__(self):
        self.imread_calls = []
        self.imwrite_calls = []

    def imread(self, path):
        self.imread_calls.append(path)
        return "fake-image"

    def imwrite(self, path, image):
        self.imwrite_calls.append((path, image))
        return True

    def imshow(self, *_args, **_kwargs):
        return None

    def waitKey(self, *_args, **_kwargs):
        return 0

    def destroyAllWindows(self):
        return None


class SmartCaptureE2EMockTest(unittest.TestCase):
    def test_main_happy_path_saves_annotated_output(self):
        fake_cv2 = FakeCV2()
        result = smart_capture.CaptureResult(
            blur_score=250.0,
            brightness=120.0,
            is_blurry=False,
            lighting_issue=False,
            card_bbox=(10, 10, 100, 60),
        )

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            with patch.object(smart_capture, "analyze_image", return_value=result):
                with patch.object(smart_capture, "render_status", return_value="annotated-image"):
                    with patch.object(sys, "argv", ["smart_capture.py", "input.jpg", "--output", "result.jpg"]):
                        captured = io.StringIO()
                        with patch("sys.stdout", new=captured):
                            smart_capture.main()

        self.assertEqual(fake_cv2.imread_calls, ["input.jpg"])
        self.assertEqual(fake_cv2.imwrite_calls, [("result.jpg", "annotated-image")])

        output = captured.getvalue()
        self.assertIn("Blur score: 250.00", output)
        self.assertIn("Brightness value: 120.00", output)
        self.assertIn("Annotated output saved to: result.jpg", output)
        self.assertNotIn("Blurry Image", output)
        self.assertNotIn("Lighting Issue", output)

    def test_main_invalid_capture_prints_flags(self):
        fake_cv2 = FakeCV2()
        result = smart_capture.CaptureResult(
            blur_score=20.0,
            brightness=20.0,
            is_blurry=True,
            lighting_issue=True,
            card_bbox=None,
        )

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            with patch.object(smart_capture, "analyze_image", return_value=result):
                with patch.object(smart_capture, "render_status", return_value="annotated-image"):
                    with patch.object(sys, "argv", ["smart_capture.py", "input.jpg"]):
                        captured = io.StringIO()
                        with patch("sys.stdout", new=captured):
                            smart_capture.main()

        output = captured.getvalue()
        self.assertIn("Blurry Image", output)
        self.assertIn("Lighting Issue", output)
        self.assertIn("FASTag-like card not detected", output)


if __name__ == "__main__":
    unittest.main()
