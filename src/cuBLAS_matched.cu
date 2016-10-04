#include "cublas_v2.h"
#include <algorithm>
#include <cuda_runtime.h>

#include "utils.h"

int main(int argc, char *argv[]) {

  /***********************************************
  *  (1) initialize program's input parameters  *
  ***********************************************/
  double alpha = 1;
  double beta = 0;
  double norm = 0;

  h_vec_t<double> distance_1;
  int num_feat_1 = atoi(argv[2]);
  ReadMatrix(distance_1, argv[1], num_feat_1);

#ifdef ACCELERATE
  std::cout << "CUDA" << std::endl;
  d_vec_t<double> d_distance_1 = distance_1;
#endif

  h_vec_t<double> distance_2;
  int num_feat_2 = atoi(argv[4]);
  ReadMatrix(distance_2, argv[3], num_feat_2);

#ifdef ACCELERATE
  d_vec_t<double> d_distance_2 = distance_2;
#endif

  int match_len = atoi(argv[6]);
  h_vec_t<int> matched_feat_1(match_len);
  h_vec_t<int> matched_feat_2(match_len);
  ReadMatchedFeatures(matched_feat_1, matched_feat_2, argv[5], match_len);
#ifdef ACCELERATE
  d_vec_t<int> d_matched_feat_1 = matched_feat_1;
  d_vec_t<int> d_matched_feat_2 = matched_feat_2;
#endif

  int num_iters = 100;
  if (8 == argc)
    num_iters = atoi(argv[7]);

  double *distance1 = raw_pointer_cast(distance_1.data());
  double *distance2 = raw_pointer_cast(distance_2.data());

  int *h_matched_1 = raw_pointer_cast(matched_feat_1.data());
  int *h_matched_2 = raw_pointer_cast(matched_feat_2.data());

  double *affinity = new double[match_len * match_len];
  affinity = AffinityInitialMatches(distance1, distance2, num_feat_1, num_feat_2,
                                     h_matched_1, h_matched_2, match_len);

  d_vec_t<double> d_affinity(affinity,
                             affinity + match_len * match_len);

  ///////////////////////////////////////////////////////////////////////////////
  int len_eigen_vec = match_len;
  d_vec_t<double> d_eigen_new(len_eigen_vec);
  fill(d_eigen_new.begin(), d_eigen_new.end(), 0);

  d_vec_t<double> d_eigen_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(d_eigen_old.begin(), d_eigen_old.end(), norm);

  cublasHandle_t handle;
  cublasCreate(&handle);

  for (int iter = 0; iter < num_iters; ++iter) {
    cublasDgemv(handle, CUBLAS_OP_N, match_len,
                match_len, &alpha,
                raw_pointer_cast(d_affinity.data()), match_len,
                raw_pointer_cast(d_eigen_old.data()), 1, &beta,
                raw_pointer_cast(d_eigen_new.data()), 1);

    double init = 0;
    norm =
        std::sqrt(transform_reduce(d_eigen_new.begin(), d_eigen_new.end(),
                                   square(), init, thrust::plus<double>()));

    transform(d_eigen_new.begin(), d_eigen_new.end(), d_eigen_old.begin(),
              division(norm));

    fill(d_eigen_new.begin(), d_eigen_new.end(), 0);
  }

  for (int i = 0; i < d_eigen_old.size(); i++) {
    std::cout << "eigen new value = " << d_eigen_new[i] << "  ";
    std::cout << "eigen old value = " << d_eigen_old[i] << std::endl;
  }

  cublasDestroy(handle);

  return (0);
}
