// evaluation.cpp - two experiments for the report:
//
//  A. Object matching on the student's dataset (N objects x 3 images).
//     Every image pair is matched (KNN + ratio test + RANSAC). Inlier counts
//     of same-object pairs should be high, of different-object pairs low.
//     Metrics: mean inliers (same / different), top-1 retrieval accuracy
//     (is the best-matching other image of the same object?) and AUC (the
//     probability that a same-object pair scores higher than a
//     different-object pair).
//
//  B. Robustness to known transformations. Each image is rotated, scaled,
//     warped in perspective, blurred and darkened with a KNOWN homography H.
//     A match is correct if H maps the first keypoint to within 3 px of the
//     second. Metrics: correct matches per image and precision.
#include "evaluation.hpp"

#include <opencv2/calib3d.hpp>  // OpenCV 5: getRotationMatrix2D/getPerspectiveTransform
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

namespace fs = std::filesystem;
using namespace cv;
using std::string;
using std::vector;

namespace {

struct Sample {
    string object, file;
    int label = 0;
    Mat img;
};

bool isImageFile(const fs::path& p) {
    string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp" || e == ".tif" ||
           e == ".tiff" || e == ".webp";
}

Mat limitSize(const Mat& img, int maxDim) {
    if (maxDim <= 0) return img;
    const double s = maxDim / (double)std::max(img.rows, img.cols);
    if (s >= 1.0) return img;
    Mat out;
    resize(img, out, Size(), s, s, INTER_AREA);
    return out;
}

vector<Sample> loadDataset(const string& dir, int maxDim) {
    vector<fs::path> objectDirs;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_directory()) objectDirs.push_back(e.path());
    std::sort(objectDirs.begin(), objectDirs.end());

    vector<Sample> out;
    int label = 0;
    for (const auto& od : objectDirs) {
        vector<fs::path> files;
        for (const auto& e : fs::directory_iterator(od))
            if (e.is_regular_file() && isImageFile(e.path())) files.push_back(e.path());
        std::sort(files.begin(), files.end());
        bool any = false;
        for (const auto& f : files) {
            Mat img = imread(f.string(), IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "  warning: cannot read " << f.string() << "\n";
                continue;
            }
            out.push_back({od.filename().string(), f.filename().string(), label, limitSize(img, maxDim)});
            any = true;
        }
        if (any) ++label;
    }
    return out;
}

double aucScore(const vector<double>& pos, const vector<double>& neg) {
    if (pos.empty() || neg.empty()) return std::nan("");
    double acc = 0;
    for (double p : pos)
        for (double n : neg) acc += p > n ? 1.0 : (p == n ? 0.5 : 0.0);
    return acc / ((double)pos.size() * (double)neg.size());
}

// Builds a transformed copy of src and the homography H (src -> dst).
bool makeSynthetic(const Mat& src, const string& kind, Mat& dst, Mat& H) {
    const float w = (float)src.cols, h = (float)src.rows;
    H = Mat::eye(3, 3, CV_64F);
    if (kind == "blur") { GaussianBlur(src, dst, Size(), 2.5); return true; }
    if (kind == "dark") { src.convertTo(dst, -1, 0.4, 0); return true; }

    if (kind == "scale0.5") {
        H = Mat(Matx33d(0.5, 0, 0, 0, 0.5, 0, 0, 0, 1));
    } else if (kind == "rot30" || kind == "rot90") {
        const Mat R = getRotationMatrix2D(Point2f(w * 0.5f, h * 0.5f), kind == "rot30" ? 30 : 90, 1.0);
        R.copyTo(H.rowRange(0, 2));
    } else if (kind == "persp") {
        const Point2f s[4] = {{0, 0}, {w, 0}, {w, h}, {0, h}};
        const Point2f d[4] = {{0.12f * w, 0.08f * h}, {0.92f * w, 0}, {w, h}, {0.05f * w, 0.9f * h}};
        H = getPerspectiveTransform(s, d);
    } else {
        return false;
    }
    // Enlarge the canvas so the whole transformed image is visible.
    vector<Point2f> corners = {{0, 0}, {w, 0}, {w, h}, {0, h}}, tc;
    perspectiveTransform(corners, tc, H);
    float minX = tc[0].x, minY = tc[0].y, maxX = tc[0].x, maxY = tc[0].y;
    for (const auto& p : tc) {
        minX = std::min(minX, p.x); minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x); maxY = std::max(maxY, p.y);
    }
    const Mat T(Matx33d(1, 0, -minX, 0, 1, -minY, 0, 0, 1));
    H = T * H;
    const Size outSize((int)std::ceil(maxX - minX), (int)std::ceil(maxY - minY));
    warpPerspective(src, dst, H, outSize, INTER_LINEAR, BORDER_CONSTANT, Scalar::all(0));
    return true;
}

struct SummaryRow {
    string combo;
    double kp = 0, detMs = 0, descMs = 0, matchMs = 0, inlSame = 0, inlDiff = 0, top1 = 0, auc = 0;
};

}  // namespace

int runEvaluation(const string& dataDir, int maxDim, const Params& params) {
    if (!fs::is_directory(dataDir)) {
        std::cerr << "Not a folder: " << dataDir << "\n";
        return 1;
    }
    std::cout << "Loading dataset from " << dataDir << " (max side " << maxDim << " px)...\n";
    const vector<Sample> samples = loadDataset(dataDir, maxDim);
    std::set<int> labels;
    for (const auto& s : samples) labels.insert(s.label);
    if (samples.size() < 2 || labels.size() < 2) {
        std::cerr << "Need at least 2 objects (one sub-folder per object, images inside).\n";
        return 1;
    }
    const int n = (int)samples.size();
    std::cout << n << " images, " << labels.size() << " objects\n";

    Params pObj = params;
    pObj.useRansac = true;
    Params pSyn = params;
    pSyn.useRansac = false;  // measure raw descriptor matching quality

    const vector<string> transforms = {"rot30", "rot90", "scale0.5", "persp", "blur", "dark"};
    vector<vector<Mat>> synImg(n, vector<Mat>(transforms.size()));
    vector<vector<Mat>> synH(n, vector<Mat>(transforms.size()));
    for (int i = 0; i < n; ++i)
        for (size_t t = 0; t < transforms.size(); ++t)
            makeSynthetic(samples[i].img, transforms[t], synImg[i][t], synH[i][t]);

    std::ofstream csvSummary("eval_summary.csv"), csvPairs("eval_pairs.csv"), csvSyn("eval_synthetic.csv");
    csvSummary << "combo,avg_keypoints,avg_detect_ms,avg_describe_ms,avg_match_ms,"
                  "mean_inliers_same,mean_inliers_diff,top1_accuracy,auc\n";
    csvPairs << "combo,object_a,image_a,object_b,image_b,same_object,ratio_matches,inliers\n";
    csvSyn << "combo,transform,avg_matches,avg_correct,precision\n";

    const Detector dets[] = {Detector::Harris, Detector::Blob, Detector::DoG};
    const Descriptor descs[] = {Descriptor::SIFT, Descriptor::LBP};
    vector<SummaryRow> rows;
    vector<vector<double>> synCorrect, synPrecision;

    for (Detector det : dets) {
        for (Descriptor ds : descs) {
            SummaryRow row;
            row.combo = string(detectorName(det)) + "+" + descriptorName(ds);
            std::cout << "Evaluating " << row.combo << "..." << std::flush;

            vector<Features> feats(n);
            for (int i = 0; i < n; ++i) {
                feats[i] = extractFeatures(samples[i].img, det, ds, pObj);
                row.kp += feats[i].keypoints.size();
                row.detMs += feats[i].detectMs;
                row.descMs += feats[i].describeMs;
            }
            row.kp /= n; row.detMs /= n; row.descMs /= n;

            // ---- A. object matching
            vector<vector<int>> score(n, vector<int>(n, -1));
            vector<double> same, diff;
            int pairs = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    const MatchResult r = matchFeatures(feats[i], feats[j], pObj);
                    score[i][j] = score[j][i] = r.inliers;
                    const bool isSame = samples[i].label == samples[j].label;
                    (isSame ? same : diff).push_back(r.inliers);
                    row.matchMs += r.matchMs;
                    ++pairs;
                    csvPairs << row.combo << ',' << samples[i].object << ',' << samples[i].file << ','
                             << samples[j].object << ',' << samples[j].file << ',' << (isSame ? 1 : 0)
                             << ',' << r.matches.size() << ',' << r.inliers << '\n';
                }
            }
            row.matchMs /= std::max(1, pairs);
            for (double v : same) row.inlSame += v;
            for (double v : diff) row.inlDiff += v;
            row.inlSame = same.empty() ? 0 : row.inlSame / same.size();
            row.inlDiff = diff.empty() ? 0 : row.inlDiff / diff.size();
            int correctTop1 = 0;
            for (int i = 0; i < n; ++i) {
                int best = -1;
                for (int j = 0; j < n; ++j)
                    if (j != i && (best < 0 || score[i][j] > score[i][best])) best = j;
                if (best >= 0 && score[i][best] > 0 && samples[best].label == samples[i].label) ++correctTop1;
            }
            row.top1 = (double)correctTop1 / n;
            row.auc = aucScore(same, diff);
            rows.push_back(row);
            csvSummary << row.combo << ',' << row.kp << ',' << row.detMs << ',' << row.descMs << ','
                       << row.matchMs << ',' << row.inlSame << ',' << row.inlDiff << ',' << row.top1
                       << ',' << row.auc << '\n';

            // ---- B. synthetic transforms
            vector<double> corr(transforms.size()), prec(transforms.size());
            for (size_t t = 0; t < transforms.size(); ++t) {
                double totalMatches = 0, totalCorrect = 0;
                for (int i = 0; i < n; ++i) {
                    const Features ft = extractFeatures(synImg[i][t], det, ds, pSyn);
                    const MatchResult r = matchFeatures(feats[i], ft, pSyn);
                    vector<Point2f> src, proj;
                    for (const auto& m : r.matches) src.push_back(feats[i].keypoints[m.queryIdx].pt);
                    if (!src.empty()) perspectiveTransform(src, proj, synH[i][t]);
                    for (size_t k = 0; k < r.matches.size(); ++k) {
                        const Point2f d = proj[k] - ft.keypoints[r.matches[k].trainIdx].pt;
                        if (d.x * d.x + d.y * d.y <= 9.f) totalCorrect += 1;
                    }
                    totalMatches += r.matches.size();
                }
                corr[t] = totalCorrect / n;
                prec[t] = totalMatches > 0 ? totalCorrect / totalMatches : 0;
                csvSyn << row.combo << ',' << transforms[t] << ',' << totalMatches / n << ','
                       << corr[t] << ',' << prec[t] << '\n';
            }
            synCorrect.push_back(corr);
            synPrecision.push_back(prec);
            std::cout << " done\n";
        }
    }

    std::printf("\n=== A. Object matching (KNN ratio %.2f + RANSAC %d px) ===\n",
                pObj.ratioPct / 100.0, pObj.ransacPx);
    std::printf("%-12s %7s %8s %8s %8s %9s %9s %7s %6s\n", "combo", "kp/img", "det ms", "desc ms",
                "match ms", "inl same", "inl diff", "top-1", "AUC");
    for (const auto& r : rows)
        std::printf("%-12s %7.0f %8.1f %8.1f %8.1f %9.1f %9.1f %6.0f%% %6.3f\n", r.combo.c_str(), r.kp,
                    r.detMs, r.descMs, r.matchMs, r.inlSame, r.inlDiff, r.top1 * 100, r.auc);

    std::printf("\n=== B. Robustness: correct matches per image (precision %%) ===\n%-12s", "combo");
    for (const auto& t : transforms) std::printf(" %14s", t.c_str());
    std::printf("\n");
    for (size_t r = 0; r < rows.size(); ++r) {
        std::printf("%-12s", rows[r].combo.c_str());
        for (size_t t = 0; t < transforms.size(); ++t)
            std::printf(" %7.0f (%3.0f%%)", synCorrect[r][t], synPrecision[r][t] * 100);
        std::printf("\n");
    }
    std::printf("\nCSV files written: eval_summary.csv, eval_pairs.csv, eval_synthetic.csv\n");
    return 0;
}
