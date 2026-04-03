#!/usr/bin/env python3
"""Smart Capture prototype for FASTag image quality checks using OpenCV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CaptureResult:
    blur_score: float
    brightness: float
    is_blurry: bool
    lighting_issue: bool
    card_bbox: Optional[Tuple[int, int, int, int]]

    @property
    def is_valid(self) -> bool:
        return not self.is_blurry and not self.lighting_issue and self.card_bbox is not None


def detect_blur(gray: np.ndarray, blur_threshold: float) -> Tuple[float, bool]:
    """Return Laplacian variance score and blurry decision."""
    import cv2
    import numpy as np

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(blur_score), blur_score < blur_threshold


def check_lighting(gray: np.ndarray, low_brightness: float, high_brightness: float) -> Tuple[float, bool]:
    """Return average brightness and whether it is outside acceptable range."""
    import numpy as np

    brightness = float(np.mean(gray))
    lighting_issue = brightness < low_brightness or brightness > high_brightness
    return brightness, lighting_issue


def detect_card_like_object(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find the best rectangular/card-like contour and return (x, y, w, h)."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        if area < 3000:
            continue

        aspect_ratio = w / float(h)
        if not 1.3 <= aspect_ratio <= 3.5:
            continue

        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)

    return best_rect


def analyze_image(
    image: np.ndarray,
    blur_threshold: float,
    low_brightness: float,
    high_brightness: float,
) -> CaptureResult:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score, is_blurry = detect_blur(gray, blur_threshold)
    brightness, lighting_issue = check_lighting(gray, low_brightness, high_brightness)
    bbox = detect_card_like_object(image)

    return CaptureResult(
        blur_score=blur_score,
        brightness=brightness,
        is_blurry=is_blurry,
        lighting_issue=lighting_issue,
        card_bbox=bbox,
    )


def render_status(image: np.ndarray, result: CaptureResult) -> np.ndarray:
    import cv2

    output = image.copy()

    color = (0, 255, 0) if result.is_valid else (0, 0, 255)  # GREEN if valid else RED

    if result.card_bbox is not None:
        x, y, w, h = result.card_bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    status_text = "VALID" if result.is_valid else "INVALID"
    cv2.putText(output, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(
        output,
        f"Blur: {result.blur_score:.2f} | Brightness: {result.brightness:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Capture prototype for FASTag image validation.")
    parser.add_argument("image", help="Path to input FASTag image")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Blur threshold (default: 100)")
    parser.add_argument("--low-brightness", type=float, default=60.0, help="Minimum acceptable brightness")
    parser.add_argument("--high-brightness", type=float, default=200.0, help="Maximum acceptable brightness")
    parser.add_argument("--output", default="smart_capture_output.jpg", help="Output annotated image path")
    parser.add_argument("--show", action="store_true", help="Display result window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    result = analyze_image(
        image,
        blur_threshold=args.blur_threshold,
        low_brightness=args.low_brightness,
        high_brightness=args.high_brightness,
    )

    print(f"Blur score: {result.blur_score:.2f}")
    print(f"Brightness value: {result.brightness:.2f}")

    if result.is_blurry:
        print("Blurry Image")

    if result.lighting_issue:
        print("Lighting Issue")

    if result.card_bbox is None:
        print("FASTag-like card not detected")

    annotated = render_status(image, result)
    cv2.imwrite(args.output, annotated)
    print(f"Annotated output saved to: {args.output}")

    if args.show:
        cv2.imshow("Smart Capture Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
