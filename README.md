# FeatureMatcher – build & usage

## Build on Windows (Visual Studio 2022 + prebuilt OpenCV 4.x)
1. Download the OpenCV 4.x Windows release from https://opencv.org/releases/ and
   extract it, e.g. to `C:\opencv` (SIFT is included in OpenCV >= 4.4, no contrib needed).
2. From this folder:
   ```
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR=C:/opencv/build
   cmake --build build --config Release
   ```
3. The program is `build\Release\FeatureMatcher.exe`. Copy
   `C:\opencv\build\x64\vc16\bin\opencv_world4xxx.dll` next to it (see
   `Release\dll_requirements.txt`).

## Build on macOS (for development and testing)
```
brew install opencv cmake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/FeatureMatcher harris image.jpg
```
Camera: macOS asks for camera permission for Terminal on the first run. If it was
denied, enable it in System Settings > Privacy & Security > Camera.

OpenCV cannot read iPhone `.heic` photos. Convert them to JPG first:
```
for f in dataset/*/*.heic; do sips -s format jpeg "$f" --out "${f%.heic}.jpg"; done
```

## Build the Windows .exe without a Windows PC
Push this folder to a GitHub repo (as the repo root). `.github/workflows/build-windows.yml`
builds on a Windows runner; download the "Release" artifact from the Actions tab.
It contains FeatureMatcher.exe and the matching opencv_world DLL.

On Linux: `cmake -S . -B build && cmake --build build`.

## Usage
```
FeatureMatcher h
FeatureMatcher harris [image]
FeatureMatcher blob   [image]
FeatureMatcher dog    [image]
FeatureMatcher m <harris|blob|dog> <sift|lbp> [image1] [image2]
FeatureMatcher eval <dataset_dir> [maxDim]
```
No image -> camera (continuous capture). Keys and trackbars are listed by `h`.

## Evaluation dataset layout
```
dataset/
  object1/ 1.jpg 2.jpg 3.jpg
  object2/ 1.jpg 2.jpg 3.jpg
  ...
```
`FeatureMatcher eval dataset 1000` writes `eval_summary.csv`, `eval_pairs.csv`
and `eval_synthetic.csv` and prints the tables.

## Files
- `src/main.cpp`        command line, camera loop, trackbars, keys, help, drawing
- `src/features.cpp/.hpp` Harris / Blob / DoG detectors, orientation assignment,
  SIFT and LBP descriptors, KNN matcher + ratio test + RANSAC
- `src/evaluation.cpp/.hpp` batch evaluation used for the report
