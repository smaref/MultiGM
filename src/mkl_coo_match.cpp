#include "utils.h"
#include <mkl.h>
#include <time.h>

int main(int argc, char *argv[]) {

  /***********************************************
  *    initialize program's input parameters     *
  ***********************************************/
  double norm = 0;
  double one = 1;
  double zero = 0;

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
  
  const clock_t begin_time = clock();

  stdvec_tuple_t aff_coo = AffinityInitialmatchesCoo(
      distance1, distance2, distance3, num_feat_1, num_feat_2, num_feat_3,
      matched_feat_1.data(), matched_feat_2.data(), matched_feat_3.data(), match_len);

  
  h_vec_t<double> value(get<0>(aff_coo));
  h_vec_t<int> column(get<1>(aff_coo));
  h_vec_t<int> row(get<2>(aff_coo));
  int nnz = value.size();
  
  std::cout << "affinity runtime: "
            << float(clock() - begin_time) / CLOCKS_PER_SEC * 1000 << std::endl;

 // for (int i = 0; i < value.size(); ++i) {
 //   std::cout << value[i] << "    " << column[i] << "       " << row[i]
 //             << std::endl;
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

    mkl_dcoomv("N", &match_len, &match_len, &one, "G**C", value.data(),
               row.data(), column.data(), &nnz,
               raw_pointer_cast(h_eigen_old.data()), &zero,
               raw_pointer_cast(h_eigen_new.data()));

    double init = 0;
    norm = std::sqrt(transform_reduce(h_eigen_new.begin(), h_eigen_new.end(),
                                      square(), init, thrust::plus<double>()));

    transform(h_eigen_new.begin(), h_eigen_new.end(), h_eigen_old.begin(),
              division(norm));

    fill(h_eigen_new.begin(), h_eigen_new.end(), 0);
  }

  std::cout << "Eigen runtime: "
            << float(clock() - begin_time2) / CLOCKS_PER_SEC * 1000 << std::endl;
  
  //std::cout << "eigen values" << std::endl;
  //for (int i = 0; i < h_eigen_old.size(); i++) {
  //  std::cout << "eigen new value = " << h_eigen_new[i] << "  ";
  //  std::cout << "eigen old value = " << h_eigen_old[i] << std::endl;
  //}

  return (0);
}
