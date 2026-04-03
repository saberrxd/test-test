# Smart Capture Prototype (FASTag)

This prototype uses OpenCV to simulate a **Smart Capture** validation flow for FASTag images.

## Features

1. **Blur detection** using **Laplacian variance**.
   - Prints `Blurry Image` when blur score is below threshold.
2. **Lighting check** using average brightness.
   - Prints `Lighting Issue` when image is too dark or too bright.
3. **Basic FASTag detection** by finding a rectangular card-like contour.
   - Draws a bounding box around detected FASTag-like area.
4. **Status overlay**
   - **GREEN** when valid
   - **RED** when invalid
5. **Bonus metrics**
   - Prints blur score and brightness value.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 smart_capture.py /path/to/fastag_image.jpg --show
```

Without `--show`, the script still saves annotated output to `smart_capture_output.jpg`.

### Optional tuning

```bash
python3 smart_capture.py /path/to/fastag_image.jpg \
  --blur-threshold 100 \
  --low-brightness 60 \
  --high-brightness 200 \
  --output result.jpg
```

## Output behavior

The script prints:
- `Blur score: ...`
- `Brightness value: ...`
- `Blurry Image` (if blur score is too low)
- `Lighting Issue` (if brightness is out of range)
- `FASTag-like card not detected` (if no rectangular candidate is found)

Annotated image includes detected box and overall status text.

## Quick sanity checks

You can verify CLI wiring (without loading OpenCV processing) using:

```bash
python3 smart_capture.py --help
```

For a mocked end-to-end CLI demo test (no OpenCV install needed), run:

```bash
python3 -m unittest tests/test_smart_capture_mock_e2e.py -v
```
