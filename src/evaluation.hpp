// evaluation.hpp - batch evaluation of all detector/descriptor combinations.
#pragma once

#include "features.hpp"
#include <string>

// dataDir must contain one sub-folder per object, each holding its images, e.g.
//   data/cup/1.jpg data/cup/2.jpg data/cup/3.jpg data/book/1.jpg ...
// Images whose longer side exceeds maxDim are downscaled (0 = keep full size).
// Writes eval_summary.csv, eval_pairs.csv and eval_synthetic.csv to the
// current folder and prints the tables to the console.
int runEvaluation(const std::string& dataDir, int maxDim, const Params& params);
