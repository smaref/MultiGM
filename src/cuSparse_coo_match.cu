#include "cusparse.h"
#include <cuda_runtime.h>

#include "utils.h"
#include "time.h"

int main(int argc, char *argv[]) {
  /***********************************************
  *     initialize program's input parameters    *
  ***********************************************/
  double alpha = 1;
  double beta = 0;
  double norm = 0;

  cusparseHandle_t handle = 0;
  cusparseMatDescr_t descr = 0;
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descr);

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

  h_vec_t<double> distance_3;
  int num_feat_3 = atoi(argv[6]);
  ReadMatrix(distance_3, argv[5], num_feat_3);
#ifdef ACCELERATE
  d_vec_t<double> d_distance_3 = distance_3;
#endif

  int match_len = atoi(argv[8]);
  h_vec_t<int> matched_feat_1(match_len);
  h_vec_t<int> matched_feat_2(match_len);
  h_vec_t<int> matched_feat_3(match_len);
  ReadMatchedFeatures(matched_feat_1, matched_feat_2, matched_feat_3, argv[7],
                      match_len);
#ifdef ACCELERATE
  d_vec_t<int> d_matched_feat_1 = matched_feat_1;
  d_vec_t<int> d_matched_feat_2 = matched_feat_2;
  d_vec_t<int> d_matched_feat_3 = matched_feat_3;
#endif

  int num_iters = 20;
  if (10 == argc)
    num_iters = atoi(argv[9]);

  /**************************************************
  *            construct affinity COO matrix            *
  ***************************************************/
  double *distance1 = raw_pointer_cast(distance_1.data());
  double *distance2 = raw_pointer_cast(distance_2.data());
  double *distance3 = raw_pointer_cast(distance_3.data());

  int *matched_1 = raw_pointer_cast(matched_feat_1.data());
  int *matched_2 = raw_pointer_cast(matched_feat_2.data());
  int *matched_3 = raw_pointer_cast(matched_feat_3.data());

  const clock_t begin_time = clock();
  
  stdvec_tuple_t aff_coo = AffinityInitialmatchesCoo(
      distance1, distance2, distance3, num_feat_1, num_feat_2, num_feat_3,
      matched_1, matched_2, matched_3, match_len);

  //  AffinityInitialmatchesCoo(distance1, distance2, value, column, row,
  //  num_feat_1, num_feat_2, matched_1, matched_2, match_len);
  //
  //  h_vec_t<double> value;
  //  h_vec_t<int> column;
  //  h_vec_t<int> row;

  d_vec_t<double> d_value(get<0>(aff_coo));
  d_vec_t<int> d_column(get<1>(aff_coo));
  d_vec_t<int> d_row(get<2>(aff_coo));

  d_vec_t<int> d_csr_row(match_len + 1);

  cusparseXcoo2csr(handle, raw_pointer_cast(d_row.data()), d_row.size(),
                   match_len, raw_pointer_cast(d_csr_row.data()),
                   CUSPARSE_INDEX_BASE_ZERO);

  std::cout << "affinity runtime: "
            << float(clock() - begin_time) / CLOCKS_PER_SEC * 1000 << std::endl;
  
  // for (int i = 0; i < get<0>(aff_coo).size(); ++i) {
  //    std::cout << "val: " << get<0>(aff_coo)[i] << " columns: "<<
  //    get<1>(aff_coo)[i] << " rows: " << get<2>(aff_coo)[i] << std::endl;
  //}
  // std::cout << "affinity" << std::endl;
  // std::cout << "values "
  //          << "  "
  //          << "columns"
  //          << "  "
  //          << "rows" << std::endl;
 // std::cout << d_value.size() << std::endl;
 // for (int i = 0; i < d_value.size(); ++i) {
 //   std::cout << d_value[i] << "     " << d_column[i] << "      " << d_row[i]
 //             << std::endl;
 // }

  /************************************************
  *           initialize eigen vectors            *
  ************************************************/
  int len_eigen_vec = match_len;
  d_vec_t<double> d_eigen_new(len_eigen_vec);
  fill(d_eigen_new.begin(), d_eigen_new.end(), 0);

  d_vec_t<double> d_eigen_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(d_eigen_old.begin(), d_eigen_old.end(), norm);

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  /************************************************
  *           computing eigen vector            *
  ************************************************/
  const clock_t begin_time2 = clock();
  
  for (int i = 0; i < num_iters; ++i) {

    cusparseDcsrmv(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, match_len, match_len,
        d_value.size(), &alpha, descr, raw_pointer_cast(d_value.data()),
        raw_pointer_cast(d_csr_row.data()), raw_pointer_cast(d_column.data()),
        raw_pointer_cast(d_eigen_old.data()), &beta,
        raw_pointer_cast(d_eigen_new.data()));

    double init = 0;
    norm = std::sqrt(transform_reduce(d_eigen_new.begin(), d_eigen_new.end(),
                                      square(), init, thrust::plus<double>()));

    transform(d_eigen_new.begin(), d_eigen_new.end(), d_eigen_old.begin(),
              division(norm));

    fill(d_eigen_new.begin(), d_eigen_new.end(), 0);
  }

  std::cout << "Eigen runtime: "
            << float(clock() - begin_time2) / CLOCKS_PER_SEC * 1000 << std::endl;

//  std::cout << "eigen values" << std::endl;
//  for (int i = 0; i < d_eigen_old.size(); i++) {
//    std::cout << "eigen new value = " << d_eigen_new[i] << "  ";
//    std::cout << "eigen old value = " << d_eigen_old[i] << std::endl;
//  }

  cusparseDestroyMatDescr(descr);
  descr = 0;

  //    destroy handle
  cusparseDestroy(handle);
  handle = 0;

  return (0);
}

