// features.hpp - keypoint detectors (Harris, Blob, DoG), descriptors (SIFT, LBP)
// and KNN matching used by the assignment program.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

enum class Detector { Harris = 0, Blob = 1, DoG = 2 };
enum class Descriptor { SIFT = 0, LBP = 1 };

const char* detectorName(Detector d);
const char* descriptorName(Descriptor d);
bool parseDetector(const std::string& s, Detector& out);
bool parseDescriptor(const std::string& s, Descriptor& out);

// Every parameter is an int so it can be bound directly to a trackbar.
// Scaled values are noted in the comments (e.g. kMilli = k * 1000).
struct HarrisParams {
    int blockSize = 3;        // neighbourhood size of the structure tensor
    int apertureIdx = 0;      // Sobel aperture: 0 -> 3, 1 -> 5, 2 -> 7
    int kMilli = 40;          // Harris k = kMilli / 1000 (typical 0.04 - 0.06)
    int threshPermille = 10;  // keep R > threshPermille/1000 * max(R)
    int nmsRadius = 4;        // non-maximum suppression radius (pixels)
    int maxPoints = 1000;     // keep the strongest N corners
    int kpSize = 16;          // support size given to each corner (for descriptors)
};

struct BlobParams {
    int minThreshold = 10;
    int maxThreshold = 220;
    int thresholdStep = 10;
    int minArea = 30;
    int maxArea = 20000;
    int minCircularityPct = 10;  // 0 disables the filter
    int minConvexityPct = 50;    // 0 disables the filter
    int minInertiaPct = 5;       // 0 disables the filter
    int minDist = 8;             // min distance between blob centres
    int color = 2;               // 0 = dark blobs, 1 = bright blobs, 2 = both
};

struct DoGParams {
    int nFeatures = 1000;      // 0 = keep all
    int nOctaveLayers = 3;
    int contrastMilli = 40;    // contrastThreshold = value / 1000
    int edgeThreshold = 10;
    int sigmaTenths = 16;      // sigma = value / 10
};

struct Params {
    HarrisParams harris;
    BlobParams blob;
    DoGParams dog;
    int ratioPct = 75;           // Lowe ratio test threshold = value / 100
    int ransacPx = 5;            // RANSAC reprojection threshold (pixels)
    int lbpRegionTenths = 30;    // LBP region diameter = value/10 * keypoint size
    bool assignOrientation = true;  // dominant-gradient orientation for Harris/Blob
    bool useRansac = true;          // geometric verification with a homography
    bool useFlann = false;          // FLANN KNN instead of brute-force KNN
};

struct Features {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;  // CV_32F, one row per keypoint
    double detectMs = 0;
    double describeMs = 0;
};

struct MatchResult {
    std::vector<cv::DMatch> matches;  // matches that passed the ratio test
    std::vector<char> inlierMask;     // 1 = geometric inlier (all 1 when RANSAC is off)
    int inliers = 0;
    cv::Mat homography;               // empty if not estimated
    double matchMs = 0;
};

cv::Mat toGray(const cv::Mat& img);
cv::Mat toBGR(const cv::Mat& img);

std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat& gray, Detector det, const Params& p);
void assignDominantOrientation(const cv::Mat& gray, std::vector<cv::KeyPoint>& kps);
void computeLBPDescriptors(const cv::Mat& gray, const std::vector<cv::KeyPoint>& kps,
                           cv::Mat& desc, int regionTenths);
void computeDescriptors(const cv::Mat& gray, std::vector<cv::KeyPoint>& kps, Descriptor d,
                        const Params& p, cv::Mat& desc);

Features extractFeatures(const cv::Mat& img, Detector det, Descriptor desc, const Params& p,
                         bool describe = true);
MatchResult matchFeatures(const Features& a, const Features& b, const Params& p);
