// Parameters for KITTI dataset

// The number of superpixel
int superpixelTotal = 1000;

// The number of iterations
int outerIterationTotal = 10;
int innerIterationTotal = 10;

// Weight parameters
double lambda_pos = 500.0;
double lambda_depth = 2000.0;
double lambda_bou = 1000.0;
double lambda_smo = 400.0;

// Inlier threshold
double lambda_d = 3.0;

// Penalty values
double lambda_hinge = 5.0;
double lambda_occ = 15.0;
double lambda_pen = 30.0;
