#include "utils.h"
#include <mkl.h>
#include <time.h>  

int main(int argc, char *argv[]) {

  /***********************************************
  *    initialize program's input parameters     *
  ***********************************************/
  double alpha = 1;
  double beta = 1;
  double norm = 0;

  h_vec_t<double> distance_1;
  int num_feat_1 = atoi(argv[2]);
  ReadMatrix(distance_1, argv[1], num_feat_1);

  h_vec_t<double> distance_2;
  int num_feat_2 = atoi(argv[4]);
  ReadMatrix(distance_2, argv[3], num_feat_2);

  h_vec_t<double> distance_3;
  int num_feat_3 = atoi(argv[6]);
  ReadMatrix(distance_3, argv[5], num_feat_3);
  
  int match_len = atoi(argv[8]);
  h_vec_t<int> matched_feat_1(match_len);
  h_vec_t<int> matched_feat_2(match_len);
  h_vec_t<int> matched_feat_3(match_len);
  ReadMatchedFeatures(matched_feat_1, matched_feat_2, matched_feat_3,argv[7], match_len);
#ifdef ACCELERATE
  d_vec_t<int> d_matched_feat_1 = matched_feat_1;
  d_vec_t<int> d_matched_feat_2 = matched_feat_2;
  d_vec_t<int> d_matched_feat_3 = matched_feat_3;
#endif
  
  int num_iters = 20;
  
  if (10 == argc)
    num_iters = atoi(argv[9]);


  /**************************************************
  *            construct affinity  matrix            *
  ***************************************************/
  double *distance1 = raw_pointer_cast(distance_1.data());
  double *distance2 = raw_pointer_cast(distance_2.data());
  double *distance3 = raw_pointer_cast(distance_3.data());
  
  //  int *d_matched_1 = raw_pointer_cast(matched_feat_1.data());
  //  int *d_matched_2 = raw_pointer_cast(matched_feat_2.data());

  double *affinity = new double[match_len * match_len];

  const clock_t begin_time = clock();
  
  affinity = AffinityInitialMatches(
      distance1, distance2, distance3, num_feat_1, num_feat_2, num_feat_3,
      matched_feat_1.data(), matched_feat_2.data(), matched_feat_3.data(),
      match_len);

  std::cout << "affinity runtime: "
            << (clock() - begin_time) / double(CLOCKS_PER_SEC) * 1000 << std::endl;
  
 // int affinity_size = match_len * match_len;
 // std::cout << match_len << " " << match_len << std::endl;
 // for (int i = 0; i < match_len; ++i) {
 //   for (int j = 0; j < match_len; ++j) {
 //     std::cout << affinity[i * match_len + j] << " ";
 //   }
 //   std::cout << std::endl;
 // }
 // std::cout << std::endl;

  /************************************************
  *           initialize eigen vector            *
  ************************************************/
  int len_eigen_vec = match_len;
  h_vec_t<double> h_eigen_new(len_eigen_vec);
  fill(h_eigen_new.begin(), h_eigen_new.end(), 0);

  h_vec_t<double> h_eigen_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(h_eigen_old.begin(), h_eigen_old.end(), norm);

  /************************************************
  *           computing eigen vector            *
  ************************************************/
  const clock_t begin_time2 = clock();
  
  for (int iter = 0; iter < num_iters; ++iter) {

    cblas_dgemv(CblasRowMajor, CblasNoTrans, match_len, match_len, 1, affinity,
                match_len, raw_pointer_cast(h_eigen_old.data()), 1, 0,
                raw_pointer_cast(h_eigen_new.data()), 1);

    double init = 0;
    norm = std::sqrt(transform_reduce(h_eigen_new.begin(), h_eigen_new.end(),
                                      square(), init, thrust::plus<double>()));

    transform(h_eigen_new.begin(), h_eigen_new.end(), h_eigen_old.begin(),
              division(norm));

    fill(h_eigen_new.begin(), h_eigen_new.end(), 0);
  }
  
  std::cout << "Eigen runtime: "
            << (clock() - begin_time2) / double(CLOCKS_PER_SEC) * 1000 << std::endl;

  //std::cout << "eigen values" << std::endl;
  //for (int i = 0; i < h_eigen_old.size(); i++) {
  //  std::cout << "eigen new value = " << h_eigen_new[i] << "  ";
  //  std::cout << "eigen old value = " << h_eigen_old[i] << std::endl;
  //}

  return (0);
}
