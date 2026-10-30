// features.cpp - implementation of detectors, descriptors and matching.
#include "features.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>

using namespace cv;
using std::vector;

// ---------------------------------------------------------------- names ---
const char* detectorName(Detector d) {
    switch (d) {
        case Detector::Harris: return "Harris";
        case Detector::Blob: return "Blob";
        default: return "DoG";
    }
}

const char* descriptorName(Descriptor d) { return d == Descriptor::SIFT ? "SIFT" : "LBP"; }

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool parseDetector(const std::string& s, Detector& out) {
    const std::string t = toLower(s);
    if (t == "harris") { out = Detector::Harris; return true; }
    if (t == "blob") { out = Detector::Blob; return true; }
    if (t == "dog") { out = Detector::DoG; return true; }
    return false;
}

bool parseDescriptor(const std::string& s, Descriptor& out) {
    const std::string t = toLower(s);
    if (t == "sift") { out = Descriptor::SIFT; return true; }
    if (t == "lbp") { out = Descriptor::LBP; return true; }
    return false;
}

// -------------------------------------------------------------- helpers ---
Mat toGray(const Mat& img) {
    Mat g;
    if (img.channels() == 3) cvtColor(img, g, COLOR_BGR2GRAY);
    else if (img.channels() == 4) cvtColor(img, g, COLOR_BGRA2GRAY);
    else g = img.clone();
    if (g.depth() != CV_8U) normalize(g, g, 0, 255, NORM_MINMAX, CV_8U);
    return g;
}

Mat toBGR(const Mat& img) {
    Mat out;
    if (img.channels() == 3) out = img;
    else if (img.channels() == 4) cvtColor(img, out, COLOR_BGRA2BGR);
    else cvtColor(img, out, COLOR_GRAY2BGR);
    if (out.depth() != CV_8U) normalize(out, out, 0, 255, NORM_MINMAX, CV_8UC3);
    return out;
}

// ---------------------------------------------------------------- Harris ---
// R = det(M) - k * trace(M)^2 on the structure tensor M (cv::cornerHarris),
// followed by relative thresholding and non-maximum suppression (a pixel is a
// corner if it equals the maximum of its (2r+1)x(2r+1) neighbourhood).
static vector<KeyPoint> detectHarris(const Mat& gray, const HarrisParams& p) {
    const int block = std::max(2, p.blockSize);
    const int aperture = 3 + 2 * std::clamp(p.apertureIdx, 0, 2);
    const double k = std::max(1, p.kMilli) / 1000.0;

    Mat R;
    cornerHarris(gray, R, block, aperture, k, BORDER_DEFAULT);

    double maxV = 0;
    minMaxLoc(R, nullptr, &maxV);
    vector<KeyPoint> kps;
    if (maxV <= 0) return kps;

    const float thr = (float)(maxV * std::max(1, p.threshPermille) / 1000.0);
    const int r = std::max(1, p.nmsRadius);
    Mat dil;
    dilate(R, dil, getStructuringElement(MORPH_ELLIPSE, Size(2 * r + 1, 2 * r + 1)));

    const float size = (float)std::max(4, p.kpSize);
    for (int y = 0; y < R.rows; ++y) {
        const float* pr = R.ptr<float>(y);
        const float* pd = dil.ptr<float>(y);
        for (int x = 0; x < R.cols; ++x)
            if (pr[x] > thr && pr[x] >= pd[x])
                kps.emplace_back(Point2f((float)x, (float)y), size, -1.f, pr[x], 0, -1);
    }
    std::sort(kps.begin(), kps.end(),
              [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });
    if (p.maxPoints > 0 && (int)kps.size() > p.maxPoints) kps.resize(p.maxPoints);
    return kps;
}

// ------------------------------------------------------------------ Blob ---
// cv::SimpleBlobDetector: threshold the image at several levels, find
// connected components, keep the ones stable across levels and passing the
// shape filters (area, circularity, convexity, inertia).
static vector<KeyPoint> detectBlob(const Mat& gray, const BlobParams& p) {
    SimpleBlobDetector::Params bp;
    const int minT = std::clamp(p.minThreshold, 0, 254);
    const int maxT = std::max(p.maxThreshold, minT + 1);
    const int step = std::max(1, p.thresholdStep);
    bp.minThreshold = (float)minT;
    bp.maxThreshold = (float)maxT;
    bp.thresholdStep = (float)step;
    const int nLevels = (maxT - minT + step - 1) / step;
    bp.minRepeatability = (size_t)std::clamp(nLevels, 1, 2);
    bp.minDistBetweenBlobs = (float)std::max(1, p.minDist);

    bp.filterByArea = true;
    bp.minArea = (float)std::max(1, p.minArea);
    bp.maxArea = (float)std::max(p.maxArea, p.minArea + 1);

    const float inf = std::numeric_limits<float>::max();
    bp.filterByCircularity = p.minCircularityPct > 0;
    bp.minCircularity = p.minCircularityPct / 100.f;
    bp.maxCircularity = inf;
    bp.filterByConvexity = p.minConvexityPct > 0;
    bp.minConvexity = p.minConvexityPct / 100.f;
    bp.maxConvexity = inf;
    bp.filterByInertia = p.minInertiaPct > 0;
    bp.minInertiaRatio = p.minInertiaPct / 100.f;
    bp.maxInertiaRatio = inf;

    bp.filterByColor = p.color != 2;
    bp.blobColor = (uchar)(p.color == 1 ? 255 : 0);

    vector<KeyPoint> kps;
    SimpleBlobDetector::create(bp)->detect(gray, kps);
    for (auto& k : kps) {
        k.size = std::max(k.size, 8.f);  // tiny blobs give unusable descriptor regions
        k.angle = -1.f;
        k.octave = 0;
        k.response = k.size;
    }
    return kps;
}

// ------------------------------------------------------------------- DoG ---
// Difference-of-Gaussians extrema in scale space (the SIFT detector).
static vector<KeyPoint> detectDoG(const Mat& gray, const DoGParams& p) {
    Ptr<SIFT> sift = SIFT::create(std::max(0, p.nFeatures), std::max(1, p.nOctaveLayers),
                                  std::max(1, p.contrastMilli) / 1000.0,
                                  std::max(1, p.edgeThreshold),
                                  std::max(5, p.sigmaTenths) / 10.0);
    vector<KeyPoint> kps;
    sift->detect(gray, kps);
    return kps;
}

vector<KeyPoint> detectKeypoints(const Mat& gray, Detector det, const Params& p) {
    switch (det) {
        case Detector::Harris: return detectHarris(gray, p.harris);
        case Detector::Blob: return detectBlob(gray, p.blob);
        default: return detectDoG(gray, p.dog);
    }
}

// ------------------------------------------------------------ orientation ---
// Harris and Blob keypoints carry no orientation. We give them a SIFT-style
// dominant orientation: a 36-bin histogram of gradient directions weighted
// by gradient magnitude and a Gaussian window, smoothed, peak interpolated.
// The angle uses the same convention as cv::KeyPoint from SIFT (degrees,
// image coordinates), so SIFT and LBP descriptors become rotation invariant.
void assignDominantOrientation(const Mat& gray, vector<KeyPoint>& kps) {
    if (kps.empty()) return;
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);
    constexpr int BINS = 36;

    for (auto& kp : kps) {
        const int r = std::clamp((int)std::lround(kp.size * 0.75f), 3, 40);
        const float sigma = r * 0.5f;
        const float inv2s2 = 1.f / (2.f * sigma * sigma);
        const int cx = (int)std::lround(kp.pt.x), cy = (int)std::lround(kp.pt.y);
        float hist[BINS] = {0};

        for (int dy = -r; dy <= r; ++dy) {
            const int y = cy + dy;
            if (y < 0 || y >= gray.rows) continue;
            const float* px = gx.ptr<float>(y);
            const float* py = gy.ptr<float>(y);
            for (int dx = -r; dx <= r; ++dx) {
                const int x = cx + dx;
                const int d2 = dx * dx + dy * dy;
                if (x < 0 || x >= gray.cols || d2 > r * r) continue;
                const float fx = px[x], fy = py[x];
                const float mag = std::sqrt(fx * fx + fy * fy);
                if (mag <= 0) continue;
                const int b = std::min(BINS - 1, (int)(fastAtan2(fy, fx) * BINS / 360.f));
                hist[b] += mag * std::exp(-d2 * inv2s2);
            }
        }
        for (int it = 0; it < 2; ++it) {
            float tmp[BINS];
            for (int i = 0; i < BINS; ++i)
                tmp[i] = 0.25f * (hist[(i + BINS - 1) % BINS] + 2 * hist[i] + hist[(i + 1) % BINS]);
            std::copy(tmp, tmp + BINS, hist);
        }
        const int best = (int)(std::max_element(hist, hist + BINS) - hist);
        if (hist[best] <= 0) { kp.angle = 0; continue; }
        const float l = hist[(best + BINS - 1) % BINS], c = hist[best], rr = hist[(best + 1) % BINS];
        const float denom = l - 2 * c + rr;
        const float off = std::abs(denom) > 1e-12f ? 0.5f * (l - rr) / denom : 0.f;
        float angle = (best + 0.5f + off) * 360.f / BINS;
        if (angle < 0) angle += 360.f;
        if (angle >= 360.f) angle -= 360.f;
        kp.angle = angle;
    }
}

// -------------------------------------------------------------------- LBP ---
// Maps the 256 8-bit LBP codes to 59 bins: 58 "uniform" patterns (at most two
// 0/1 transitions around the circle) and one bin for all other patterns.
static const std::array<uchar, 256>& uniformLUT() {
    static const std::array<uchar, 256> lut = [] {
        std::array<uchar, 256> t{};
        int next = 0;
        for (int i = 0; i < 256; ++i) {
            int transitions = 0;
            for (int b = 0; b < 8; ++b)
                transitions += ((i >> b) & 1) != ((i >> ((b + 1) % 8)) & 1);
            t[i] = (uchar)(transitions <= 2 ? next++ : 58);
        }
        return t;
    }();
    return lut;
}

// LBP keypoint descriptor (own implementation, OpenCV has none):
//  1. sample a 34x34 patch around the keypoint, rotated by kp.angle and
//     scaled so it covers regionTenths/10 * kp.size pixels (sampled from the
//     Gaussian pyramid level matching the scale, to avoid aliasing);
//  2. compute the 8-neighbour LBP code of each inner pixel (32x32 codes);
//  3. split into a 4x4 grid, build a 59-bin uniform-LBP histogram per cell;
//  4. normalise each cell (L1 + square root = Hellinger kernel) and the
//     whole 944-D vector (L2) so it can be matched with Euclidean distance.
void computeLBPDescriptors(const Mat& gray, const vector<KeyPoint>& kps, Mat& desc,
                           int regionTenths) {
    constexpr int P = 32, G = 4, C = P / G, B = 59, S = P + 2;
    desc.create((int)kps.size(), G * G * B, CV_32F);
    desc.setTo(0);
    if (kps.empty()) return;

    Mat smooth;
    GaussianBlur(gray, smooth, Size(3, 3), 0);
    int maxLevel = 0;
    const int minDim = std::min(gray.rows, gray.cols);
    while (maxLevel < 8 && (minDim >> (maxLevel + 1)) >= S) ++maxLevel;
    vector<Mat> pyr;
    buildPyramid(smooth, pyr, maxLevel);

    const auto& lut = uniformLUT();
    const double regionFactor = std::max(1, regionTenths) / 10.0;
    const double c0 = (S - 1) * 0.5;
    Mat patch;

    for (int i = 0; i < (int)kps.size(); ++i) {
        const KeyPoint& kp = kps[i];
        const double region = std::max(8.0, kp.size * regionFactor);
        const double s = region / P;  // source pixels per patch pixel (level 0)
        int L = 0;
        while (L < maxLevel && s / (1 << (L + 1)) >= 1.0) ++L;
        const double f = 1.0 / (1 << L);
        const double sL = s * f;
        const double cx = (kp.pt.x + 0.5) * f - 0.5, cy = (kp.pt.y + 0.5) * f - 0.5;
        const double a = kp.angle >= 0 ? kp.angle * CV_PI / 180.0 : 0.0;
        const double ca = std::cos(a) * sL, sa = std::sin(a) * sL;
        // Maps patch coordinates -> image coordinates (used with WARP_INVERSE_MAP).
        const Matx23d M(ca, -sa, cx - ca * c0 + sa * c0,
                        sa, ca, cy - sa * c0 - ca * c0);
        warpAffine(pyr[L], patch, M, Size(S, S), INTER_LINEAR | WARP_INVERSE_MAP,
                   BORDER_REFLECT_101);

        float* d = desc.ptr<float>(i);
        for (int y = 1; y <= P; ++y) {
            const uchar* up = patch.ptr<uchar>(y - 1);
            const uchar* mid = patch.ptr<uchar>(y);
            const uchar* dn = patch.ptr<uchar>(y + 1);
            for (int x = 1; x <= P; ++x) {
                const uchar c = mid[x];
                // neighbours in circular order: UL, U, UR, R, DR, D, DL, L
                const int code = ((up[x - 1] >= c) << 7) | ((up[x] >= c) << 6) |
                                 ((up[x + 1] >= c) << 5) | ((mid[x + 1] >= c) << 4) |
                                 ((dn[x + 1] >= c) << 3) | ((dn[x] >= c) << 2) |
                                 ((dn[x - 1] >= c) << 1) | (int)(mid[x - 1] >= c);
                const int cell = ((y - 1) / C) * G + (x - 1) / C;
                d[cell * B + lut[code]] += 1.f;
            }
        }
        double sq = 0;
        for (int cell = 0; cell < G * G; ++cell) {
            float* h = d + cell * B;
            float sum = 0;
            for (int b = 0; b < B; ++b) sum += h[b];
            if (sum <= 0) continue;
            for (int b = 0; b < B; ++b) {
                h[b] = std::sqrt(h[b] / sum);
                sq += (double)h[b] * h[b];
            }
        }
        if (sq > 0) {
            const float inv = (float)(1.0 / std::sqrt(sq));
            for (int k = 0; k < G * G * B; ++k) d[k] *= inv;
        }
    }
}

void computeDescriptors(const Mat& gray, vector<KeyPoint>& kps, Descriptor d, const Params& p,
                        Mat& desc) {
    if (kps.empty()) { desc.release(); return; }
    if (d == Descriptor::SIFT) {
        for (auto& k : kps)
            if (k.angle < 0) k.angle = 0;  // SIFT needs a defined orientation
        // Same octave layers / sigma as the DoG detector so DoG octave codes stay valid.
        Ptr<SIFT> sift = SIFT::create(0, std::max(1, p.dog.nOctaveLayers), 0.04, 10,
                                      std::max(5, p.dog.sigmaTenths) / 10.0);
        sift->compute(gray, kps, desc);
    } else {
        computeLBPDescriptors(gray, kps, desc, p.lbpRegionTenths);
    }
}

Features extractFeatures(const Mat& img, Detector det, Descriptor desc, const Params& p,
                         bool describe) {
    Features f;
    const Mat gray = toGray(img);
    TickMeter tm;
    tm.start();
    f.keypoints = detectKeypoints(gray, det, p);
    if (det != Detector::DoG && p.assignOrientation) assignDominantOrientation(gray, f.keypoints);
    tm.stop();
    f.detectMs = tm.getTimeMilli();
    if (describe) {
        tm.reset();
        tm.start();
        computeDescriptors(gray, f.keypoints, desc, p, f.descriptors);
        tm.stop();
        f.describeMs = tm.getTimeMilli();
    }
    return f;
}

// --------------------------------------------------------------- matching ---
// KNN matching (k = 2) + Lowe's ratio test, optionally followed by RANSAC
// homography estimation to separate geometric inliers from outliers.
MatchResult matchFeatures(const Features& a, const Features& b, const Params& p) {
    MatchResult r;
    TickMeter tm;
    tm.start();
    if (a.descriptors.rows < 1 || b.descriptors.rows < 2) return r;

    vector<vector<DMatch>> knn;
    if (p.useFlann) {
        FlannBasedMatcher matcher;
        matcher.knnMatch(a.descriptors, b.descriptors, knn, 2);
    } else {
        BFMatcher matcher(NORM_L2);
        matcher.knnMatch(a.descriptors, b.descriptors, knn, 2);
    }
    const float ratio = std::clamp(p.ratioPct, 1, 100) / 100.f;
    for (const auto& m : knn)
        if (m.size() >= 2 && m[0].distance < ratio * m[1].distance) r.matches.push_back(m[0]);

    r.inlierMask.assign(r.matches.size(), 1);
    r.inliers = (int)r.matches.size();
    if (p.useRansac) {
        std::fill(r.inlierMask.begin(), r.inlierMask.end(), 0);
        r.inliers = 0;
        if (r.matches.size() >= 4) {
            vector<Point2f> p1, p2;
            for (const auto& m : r.matches) {
                p1.push_back(a.keypoints[m.queryIdx].pt);
                p2.push_back(b.keypoints[m.trainIdx].pt);
            }
            Mat mask;
            Mat H = findHomography(p1, p2, RANSAC, std::max(1, p.ransacPx), mask);
            if (!H.empty()) {
                r.homography = H;
                for (size_t i = 0; i < r.matches.size(); ++i) r.inlierMask[i] = mask.at<uchar>((int)i);
                r.inliers = countNonZero(mask);
            }
        }
    }
    tm.stop();
    r.matchMs = tm.getTimeMilli();
    return r;
}
