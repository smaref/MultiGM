#include "utils.h"
#include <time.h>  

int main(int argc, char *argv[]) {

  /***********************************************
  *    initialize program's input parameters     *
  ***********************************************/
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
  
  int num_iters = 20;
  
  if (10 == argc)
    num_iters = atoi(argv[9]);

  /**************************************************
  *            construct affinity  matrix            *
  ***************************************************/
  double *distance1 = raw_pointer_cast(distance_1.data());
  double *distance2 = raw_pointer_cast(distance_2.data());
  double *distance3 = raw_pointer_cast(distance_3.data());

  double *affinity =
      new double[distance_1.size() * distance_2.size() * distance_3.size()];
  
  const clock_t begin_time = clock();

  affinity = AffinityMatrixCtor(distance1, distance2, distance3, num_feat_1,
                                num_feat_2, num_feat_3);

  std::cout << "affinity runtime: "
            << float(clock() - begin_time) / CLOCKS_PER_SEC * 1000 << std::endl;
  
  int affinity_row = num_feat_1 * num_feat_2 * num_feat_3;
  int affinity_size = distance_1.size() * distance_2.size() * distance_3.size();

//  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, affinity_row,
//              affinity_row, affinity_row, 1.0, A, k, B, n, 0.0, C, n);

 // std::cout << num_feat_1 * num_feat_2 * num_feat_3 << " "
 //           << num_feat_1 * num_feat_2 * num_feat_3 << std::endl;

 // for (int i = 0; i < num_feat_1 * num_feat_2 * num_feat_3; ++i) {
 //   for (int j = 0; j < num_feat_1 * num_feat_2 * num_feat_3; ++j) {
 //     std::cout << affinity[i * num_feat_1 * num_feat_2 * num_feat_3 + j]
 //               << "  ";
 //   }
 //   std::cout << std::endl;
 // }
 // std::cout << std::endl;

  /************************************************
  *           initialize eigen vector            *
  ************************************************/
  int len_eigen_vec = num_feat_1 * num_feat_2 * num_feat_3;
  h_vec_t<double> eigen_new(len_eigen_vec);
  fill(eigen_new.begin(), eigen_new.end(), 0);

  h_vec_t<double> eigen_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(eigen_old.begin(), eigen_old.end(), norm);

  /************************************************
  *           computing eigen vector            *
  ************************************************/
  const clock_t begin_time2 = clock();
  
  for (int iter = 0; iter < num_iters; ++iter) {

    for (int i = 0; i < len_eigen_vec; ++i)
      for (int j = 0; j < len_eigen_vec; ++j)
    
          eigen_new[i] += affinity[i * len_eigen_vec + j] * eigen_old[j];

    norm = VectorNorm(eigen_new.data(), len_eigen_vec);

    for (int j = 0; j < len_eigen_vec; ++j) {
      eigen_old[j] = eigen_new[j] / norm;
      }
      std::fill(eigen_new.begin(), eigen_new.end(), 0);
  }
    
  std::cout << "Eigen runtime: "
            << float(clock() - begin_time2) / CLOCKS_PER_SEC * 1000 << std::endl;

//  std::cout << "eigen values" << std::endl;
//  for (int i = 0; i < eigen_old.size(); i++) {
//    std::cout << "eigen new value = " << eigen_new[i] << "  ";
//    std::cout << "eigen old value = " << eigen_old[i] << std::endl;
//  }

  return (0);
}
