// main.cpp - keypoint detection & matching assignment (OpenCV).
//
//   FeatureMatcher h
//   FeatureMatcher harris|blob|dog [image]
//   FeatureMatcher m <harris|blob|dog> <sift|lbp> [image1] [image2]
//   FeatureMatcher eval <dataset_dir> [maxDim]
//
// Without an image file the program uses the camera and processes frames
// continuously. Every operation is applied to the ORIGINAL image/frame,
// never to the result of a previous step.
#include "evaluation.hpp"
#include "features.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using std::string;
using std::vector;

namespace {

const char* WIN_RESULT = "Result";
const char* WIN_PARAMS = "Parameters";
const char* WIN_HELP = "Help";

const vector<string> kHelp = {
    "Keypoint detection and matching - Harris / Blob / DoG detectors,",
    "SIFT / LBP descriptors, KNN matcher with ratio test (+ optional RANSAC).",
    "",
    "Command line:",
    "  h                              show this help",
    "  harris [image]                 Harris corners on the image",
    "  blob   [image]                 blob keypoints (SimpleBlobDetector)",
    "  dog    [image]                 DoG keypoints (SIFT scale space)",
    "  m <det> <desc> img1 [img2]     match img1 with img2",
    "        det  = harris | blob | dog      desc = sift | lbp",
    "  eval <dataset_dir> [maxDim]    evaluate all 6 combinations (report)",
    "  No image -> the camera is used and frames are processed continuously.",
    "  In 'm' mode a missing img2 means live camera frames are matched",
    "  against img1; with no images at all, press 'c' to grab a reference.",
    "",
    "Keys (always applied to the ORIGINAL image):",
    "  h        show / hide this help       q, Esc   quit",
    "  1 2 3    detector Harris / Blob / DoG",
    "  4 5      descriptor SIFT / LBP (match mode)",
    "  o        orientation for Harris/Blob keypoints on/off",
    "  g        RANSAC geometric verification on/off",
    "  f        FLANN / brute-force KNN matcher",
    "  c        capture reference frame (camera match mode)",
    "  space    pause / resume the camera",
    "  s        save the current result image",
    "  r        reset all parameters to defaults",
    "Algorithm parameters are on trackbars in the 'Parameters' window.",
};

// ------------------------------------------------------- trackbar binding ---
struct Binding { int* value; };
std::deque<Binding> g_bindings;  // deque: element addresses stay valid
bool g_dirty = true;             // parameters changed -> recompute
bool g_paramsOpen = false;
bool g_helpOpen = false;

void onTrackbar(int pos, void* userdata) {
    *static_cast<Binding*>(userdata)->value = pos;
    g_dirty = true;
}

void addTrackbar(const string& name, int* value, int minV, int maxV) {
    const int init = std::clamp(*value, minV, maxV);
    g_bindings.push_back({value});
    createTrackbar(name, WIN_PARAMS, nullptr, maxV, onTrackbar, &g_bindings.back());
    try {
        setTrackbarMin(name, WIN_PARAMS, minV);  // not implemented by every GUI backend
    } catch (const cv::Exception&) {
        // fine: the algorithms clamp every parameter to its valid range anyway
    }
    setTrackbarPos(name, WIN_PARAMS, init);
    *value = init;
}

void buildParamsWindow(Params& p, Detector det, Descriptor desc, bool matchMode) {
    if (g_paramsOpen) destroyWindow(WIN_PARAMS);
    g_bindings.clear();
    namedWindow(WIN_PARAMS, WINDOW_AUTOSIZE);
    moveWindow(WIN_PARAMS, 0, 0);
    g_paramsOpen = true;

    string title = string(detectorName(det)) + " parameters";
    if (matchMode) title += string(" + ") + descriptorName(desc) + " matching";
    Mat banner(40, 460, CV_8UC3, Scalar(45, 45, 45));
    putText(banner, title, Point(10, 26), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1, LINE_AA);
    imshow(WIN_PARAMS, banner);

    switch (det) {
        case Detector::Harris:
            addTrackbar("Block size", &p.harris.blockSize, 2, 15);
            addTrackbar("Aperture 3/5/7", &p.harris.apertureIdx, 0, 2);
            addTrackbar("k x1000", &p.harris.kMilli, 1, 200);
            addTrackbar("Thresh x0.001max", &p.harris.threshPermille, 1, 500);
            addTrackbar("NMS radius", &p.harris.nmsRadius, 1, 20);
            addTrackbar("Max points", &p.harris.maxPoints, 10, 5000);
            addTrackbar("KP size", &p.harris.kpSize, 4, 64);
            break;
        case Detector::Blob:
            addTrackbar("Min thresh", &p.blob.minThreshold, 0, 254);
            addTrackbar("Max thresh", &p.blob.maxThreshold, 1, 255);
            addTrackbar("Thresh step", &p.blob.thresholdStep, 1, 50);
            addTrackbar("Min area", &p.blob.minArea, 1, 5000);
            addTrackbar("Max area", &p.blob.maxArea, 10, 100000);
            addTrackbar("Min circ %", &p.blob.minCircularityPct, 0, 100);
            addTrackbar("Min convex %", &p.blob.minConvexityPct, 0, 100);
            addTrackbar("Min inertia %", &p.blob.minInertiaPct, 0, 100);
            addTrackbar("Min dist", &p.blob.minDist, 1, 100);
            addTrackbar("Color 0dk 1br 2any", &p.blob.color, 0, 2);
            break;
        case Detector::DoG:
            addTrackbar("N features 0=all", &p.dog.nFeatures, 0, 5000);
            addTrackbar("Octave layers", &p.dog.nOctaveLayers, 1, 8);
            addTrackbar("Contrast x1000", &p.dog.contrastMilli, 1, 200);
            addTrackbar("Edge thresh", &p.dog.edgeThreshold, 1, 50);
            addTrackbar("Sigma x10", &p.dog.sigmaTenths, 5, 50);
            break;
    }
    if (matchMode) {
        addTrackbar("Ratio test %", &p.ratioPct, 10, 100);
        addTrackbar("RANSAC px", &p.ransacPx, 1, 20);
        if (desc == Descriptor::LBP) addTrackbar("LBP region x10", &p.lbpRegionTenths, 10, 80);
    }
    g_dirty = true;
}

// ---------------------------------------------------------------- display ---
void printHelp() {
    for (const auto& line : kHelp) std::cout << line << "\n";
    std::cout.flush();
}

void showHelpWindow() {
    Mat canvas((int)kHelp.size() * 22 + 20, 640, CV_8UC3, Scalar(30, 30, 30));
    for (size_t i = 0; i < kHelp.size(); ++i)
        putText(canvas, kHelp[i], Point(12, 26 + (int)i * 22), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(230, 230, 230), 1, LINE_AA);
    imshow(WIN_HELP, canvas);
    moveWindow(WIN_HELP, 40, 120);
    g_helpOpen = true;
}

void toggleHelp() {
    if (g_helpOpen) {
        destroyWindow(WIN_HELP);
        g_helpOpen = false;
    } else {
        printHelp();
        showHelpWindow();
    }
}

// Large images are shrunk for display only; processing uses full resolution.
Mat fitForDisplay(const Mat& img, int maxW = 1400, int maxH = 850) {
    const double s = std::min(1.0, std::min(maxW / (double)img.cols, maxH / (double)img.rows));
    Mat out;
    if (s < 1.0) resize(img, out, Size(), s, s, INTER_AREA);
    else out = img.clone();
    return out;
}

void drawStatus(Mat& img, const vector<string>& lines) {
    const double scale = 0.5;
    int width = 0;
    for (const auto& l : lines) {
        int base = 0;
        width = std::max(width, getTextSize(l, FONT_HERSHEY_SIMPLEX, scale, 1, &base).width);
    }
    Rect box(0, 0, std::min(img.cols, width + 16), std::min(img.rows, (int)lines.size() * 20 + 10));
    Mat roi = img(box);
    Mat dark(roi.size(), roi.type(), Scalar::all(0));
    addWeighted(roi, 0.35, dark, 0.65, 0, roi);
    for (size_t i = 0; i < lines.size(); ++i)
        putText(img, lines[i], Point(8, 20 + (int)i * 20), FONT_HERSHEY_SIMPLEX, scale,
                Scalar(255, 255, 255), 1, LINE_AA);
}

int lineThickness(const Mat& img) {
    return std::max(1, (int)std::lround(std::max(img.cols, img.rows) / 1000.0));
}

void drawKeypointsOn(Mat& img, const vector<KeyPoint>& kps, Detector det) {
    const int t = lineThickness(img);
    const Scalar color(0, 255, 0), dirColor(0, 0, 255);
    for (const auto& kp : kps) {
        const float radius = det == Detector::Harris ? 4.f * t : std::max(2.f, kp.size * 0.5f);
        circle(img, kp.pt, (int)std::lround(radius), color, t, LINE_AA);
        if (kp.angle >= 0) {
            const float a = kp.angle * (float)CV_PI / 180.f;
            const float len = std::max(radius, 6.f * t);
            line(img, kp.pt, kp.pt + Point2f(std::cos(a), std::sin(a)) * len, dirColor, t, LINE_AA);
        }
    }
}

Mat drawMatchResult(const Mat& img1, const Features& f1, const Mat& img2, const Features& f2,
                    const MatchResult& mr) {
    Mat vis;
    vector<char> outlierMask(mr.matches.size()), inlierMask(mr.matches.size());
    for (size_t i = 0; i < mr.matches.size(); ++i) {
        inlierMask[i] = mr.inlierMask[i];
        outlierMask[i] = !mr.inlierMask[i];
    }
    // all keypoints (blue), outliers (red), then inliers (green) on top
    drawMatches(img1, f1.keypoints, img2, f2.keypoints, mr.matches, vis, Scalar(0, 0, 255),
                Scalar(255, 128, 0), outlierMask, DrawMatchesFlags::DEFAULT);
    drawMatches(img1, f1.keypoints, img2, f2.keypoints, mr.matches, vis, Scalar(0, 255, 0),
                Scalar(255, 128, 0), inlierMask,
                DrawMatchesFlags::DRAW_OVER_OUTIMG | DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    if (!mr.homography.empty()) {  // outline of image 1 projected into image 2
        const float w = (float)img1.cols, h = (float)img1.rows;
        vector<Point2f> corners = {{0, 0}, {w, 0}, {w, h}, {0, h}}, proj;
        perspectiveTransform(corners, proj, mr.homography);
        vector<Point> poly;
        for (const auto& p : proj) poly.emplace_back((int)std::lround(p.x + w), (int)std::lround(p.y));
        polylines(vis, poly, true, Scalar(0, 255, 255), lineThickness(vis) + 1, LINE_AA);
    }
    return vis;
}

void saveResult(const Mat& img) {
    if (img.empty()) return;
    char name[64];
    const std::time_t now = std::time(nullptr);
    std::strftime(name, sizeof(name), "result_%Y%m%d_%H%M%S.png", std::localtime(&now));
    if (imwrite(name, img)) std::cout << "Saved " << name << "\n";
    else std::cerr << "Could not save " << name << "\n";
}

bool openCamera(VideoCapture& cap) {
    cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "No image file given and no camera could be opened.\n";
        return false;
    }
    std::cout << "Camera opened. Press h for help, q to quit.\n";
    return true;
}

Mat readColorImage(const string& file) {
    Mat img = imread(file, IMREAD_COLOR);  // always 3-channel BGR
    if (img.empty()) std::cerr << "Cannot read image: " << file << "\n";
    return img;
}

string fmt(const char* f, double a, double b = 0, double c = 0, double d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c, d);
    return buf;
}

bool windowClosed() {
    try {
        return getWindowProperty(WIN_RESULT, WND_PROP_VISIBLE) < 1;
    } catch (const cv::Exception&) {
        return true;
    }
}

// Keys shared by both modes. Returns false when the user wants to quit.
bool handleCommonKey(int key, Params& params, Detector& det, Descriptor& desc, bool matchMode,
                     bool& paused, const Mat& lastResult) {
    bool rebuild = false;
    switch (key) {
        case 'q': case 'Q': case 27: return false;
        case 'h': case 'H': toggleHelp(); break;
        case '1': det = Detector::Harris; rebuild = true; break;
        case '2': det = Detector::Blob; rebuild = true; break;
        case '3': det = Detector::DoG; rebuild = true; break;
        case '4': if (matchMode) { desc = Descriptor::SIFT; rebuild = true; } break;
        case '5': if (matchMode) { desc = Descriptor::LBP; rebuild = true; } break;
        case 'o': case 'O': params.assignOrientation = !params.assignOrientation; g_dirty = true; break;
        case 'g': case 'G': params.useRansac = !params.useRansac; g_dirty = true; break;
        case 'f': case 'F': params.useFlann = !params.useFlann; g_dirty = true; break;
        case ' ': paused = !paused; g_dirty = true; break;
        case 's': case 'S': saveResult(lastResult); break;
        case 'r': case 'R': params = Params(); rebuild = true; break;
        default: break;
    }
    if (rebuild) buildParamsWindow(params, det, desc, matchMode);
    return true;
}

// ------------------------------------------------------- detection mode ---
int runSingle(Detector det, const string& file) {
    Params params;
    Descriptor desc = Descriptor::SIFT;  // unused in this mode
    const bool camera = file.empty();
    VideoCapture cap;
    Mat original, lastResult;
    bool paused = false;

    if (!camera) {
        original = readColorImage(file);
        if (original.empty()) return 1;
    } else if (!openCamera(cap)) {
        return 1;
    }
    namedWindow(WIN_RESULT, WINDOW_AUTOSIZE);
    moveWindow(WIN_RESULT, 480, 0);
    buildParamsWindow(params, det, desc, false);

    for (;;) {
        if (camera && !paused) {
            Mat frame;
            if (!cap.read(frame) || frame.empty()) { std::cerr << "Camera stopped.\n"; break; }
            original = toBGR(frame);
            g_dirty = true;
        }
        if (g_dirty && !original.empty()) {
            g_dirty = false;
            const Features f = extractFeatures(original, det, desc, params, false);
            Mat vis = original.clone();  // always start from the original image
            drawKeypointsOn(vis, f.keypoints, det);

            string mode = detectorName(det);
            if (det != Detector::DoG) mode += params.assignOrientation ? " (orientation on)" : " (orientation off)";
            Mat display = fitForDisplay(vis);
            drawStatus(display, {mode + " | keypoints: " + std::to_string(f.keypoints.size()),
                                 fmt("detect %.1f ms | image %.0f x %.0f", f.detectMs, original.cols, original.rows),
                                 camera ? string(paused ? "camera (paused)" : "camera") : file,
                                 "h = help, q = quit"});
            lastResult = display;  // saved with the statistics overlay (report figures)
            imshow(WIN_RESULT, display);
        }
        const int key = waitKey(camera ? 1 : 20);
        if (key >= 0 && !handleCommonKey(key & 0xFF, params, det, desc, false, paused, lastResult)) break;
        if (windowClosed()) break;
    }
    destroyAllWindows();
    return 0;
}

// -------------------------------------------------------- matching mode ---
int runMatch(Detector det, Descriptor desc, const string& file1, const string& file2) {
    Params params;
    const bool camera = file2.empty();
    VideoCapture cap;
    Mat reference, live, lastResult;
    Features fRef;
    bool refDirty = true, paused = false;

    if (!file1.empty()) {
        reference = readColorImage(file1);
        if (reference.empty()) return 1;
    }
    if (!camera) {
        live = readColorImage(file2);
        if (live.empty()) return 1;
    } else if (!openCamera(cap)) {
        return 1;
    }
    namedWindow(WIN_RESULT, WINDOW_AUTOSIZE);
    moveWindow(WIN_RESULT, 480, 0);
    buildParamsWindow(params, det, desc, true);

    for (;;) {
        bool newFrame = false;
        if (camera && !paused) {
            Mat frame;
            if (!cap.read(frame) || frame.empty()) { std::cerr << "Camera stopped.\n"; break; }
            live = toBGR(frame);
            newFrame = true;
        }
        if (g_dirty) refDirty = true;
        if ((g_dirty || newFrame) && !live.empty()) {
            g_dirty = false;
            Mat display;
            if (reference.empty()) {
                display = fitForDisplay(live);
                drawStatus(display, {"No reference image: press 'c' to capture one", "h = help, q = quit"});
            } else {
                if (refDirty) {
                    fRef = extractFeatures(reference, det, desc, params);
                    refDirty = false;
                }
                const Features fLive = extractFeatures(live, det, desc, params);
                const MatchResult mr = matchFeatures(fRef, fLive, params);
                display = fitForDisplay(drawMatchResult(reference, fRef, live, fLive, mr));
                const double pct = mr.matches.empty() ? 0 : 100.0 * mr.inliers / mr.matches.size();
                drawStatus(display,
                           {string(detectorName(det)) + " + " + descriptorName(desc) + " | KNN " +
                                (params.useFlann ? "FLANN" : "brute-force") +
                                fmt(" | ratio %.2f", params.ratioPct / 100.0) +
                                " | RANSAC " + (params.useRansac ? "on" : "off") +
                                (det != Detector::DoG ? (params.assignOrientation ? " | orient on" : " | orient off") : ""),
                            fmt("keypoints %.0f / %.0f | matches %.0f | inliers %.0f", (double)fRef.keypoints.size(),
                                (double)fLive.keypoints.size(), (double)mr.matches.size(), (double)mr.inliers) +
                                fmt(" (%.0f%%)", pct),
                            fmt("time: detect %.1f + %.1f ms, describe %.1f + %.1f ms", fRef.detectMs,
                                fLive.detectMs, fRef.describeMs, fLive.describeMs) +
                                fmt(", match %.1f ms", mr.matchMs),
                            "green = inlier, red = rejected, yellow = projected outline | h = help"});
            }
            lastResult = display;  // saved with the statistics overlay (report figures)
            imshow(WIN_RESULT, display);
        }
        const int key = waitKey(camera ? 1 : 20);
        if (key >= 0) {
            const int k = key & 0xFF;
            if ((k == 'c' || k == 'C') && camera && !live.empty()) {
                reference = live.clone();
                refDirty = true;
                g_dirty = true;
                std::cout << "Reference frame captured.\n";
            } else if (!handleCommonKey(k, params, det, desc, true, paused, lastResult)) {
                break;
            }
        }
        if (windowClosed()) break;
    }
    destroyAllWindows();
    return 0;
}

string lower(string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printHelp();
        return 1;
    }
    const string cmd = lower(argv[1]);
    try {
        if (cmd == "h" || cmd == "-h" || cmd == "--help" || cmd == "help") {
            printHelp();
            showHelpWindow();
            waitKey(0);
            return 0;
        }
        Detector det;
        if (parseDetector(cmd, det)) return runSingle(det, argc >= 3 ? argv[2] : "");

        if (cmd == "m") {
            Descriptor desc;
            if (argc < 4 || !parseDetector(argv[2], det) || !parseDescriptor(argv[3], desc)) {
                std::cerr << "Usage: m <harris|blob|dog> <sift|lbp> [image1] [image2]\n";
                return 1;
            }
            return runMatch(det, desc, argc >= 5 ? argv[4] : "", argc >= 6 ? argv[5] : "");
        }
        if (cmd == "eval") {
            if (argc < 3) {
                std::cerr << "Usage: eval <dataset_dir> [maxDim=1000]\n";
                return 1;
            }
            return runEvaluation(argv[2], argc >= 4 ? std::atoi(argv[3]) : 1000, Params());
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "Unknown command '" << argv[1] << "'.\n\n";
    printHelp();
    return 1;
}
