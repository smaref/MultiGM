/* Using cuSPARSE for matrix vector multplication of completed affinity */

#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "utils.h"

int main(int argc, char *argv[]) {

  /***********************************************
  *  (1) initialize program's input parameters  *
  ***********************************************/
  double alpha = 1;
  double beta = 1;
  double norm = 0;

  int bin_width = 10;

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseMatDescr_t descr = 0;

  h_vec_t<int> distance_1;
  int num_feat_1 = atoi(argv[2]);
  ReadMatrix(distance_1, argv[1], num_feat_1);

  h_vec_t<double> distance_2;
  int num_feat_2 = atoi(argv[4]);
  ReadMatrix(distance_2, argv[3], num_feat_2);

  int match_len = atoi(argv[6]);
  h_vec_t<int> matched_feat_1(match_len);
  h_vec_t<int> matched_feat_2(match_len);
  ReadMatchedFeatures(matched_feat_1, matched_feat_2, argv[5], match_len);

  int num_iters = 1;
  if (8 == argc)
    num_iters = atoi(argv[7]);
  /*************************/
  /********** (1) **********/
  /*************************/

  h_vec_t<int> uniq_keys = FindUniques(distance_1);

  uniq_keys.erase(
      remove_if(uniq_keys.begin(), uniq_keys.end(), IsLessThan(bin_width)),
      uniq_keys.end());

  h_vec_t<int> *keys_idcs = new h_vec_t<int>[uniq_keys.size()];

  for (int i = 0; i < uniq_keys.size(); ++i) {
    keys_idcs[i].resize(distance_1.size());
  }

  counting_iterator<int> first_idx(0);
  counting_iterator<int> last_idx = first_idx + num_feat_1;

  h_zip_iter_t<int, int> first =
      make_zip_iterator(make_tuple(distance_1.begin(), first_idx));
  h_zip_iter_t<int, int> last =
      make_zip_iterator(make_tuple(distance_1.end(), last_idx));

  for (int i = 0; i < uniq_keys.size(); ++i) {
    transform(first, last, keys_idcs[i].begin(), IsEqual(uniq_keys[i]));
    keys_idcs[i].erase(remove(keys_idcs[i].begin(), keys_idcs[i].end(), -1),
                       keys_idcs[i].end());
  }

  int len_distance_2 = num_feat_2 * num_feat_2;

  /////// constructing coo sparse block affinity

  last_idx = first_idx + len_distance_2;
  h_vec_t<double> *coo_val = new h_vec_t<double>[uniq_keys.size()];
  h_vec_t<int> *coo_row = new h_vec_t<int>[uniq_keys.size()];
  h_vec_t<int> *coo_col = new h_vec_t<int>[uniq_keys.size()];

  d_vec_t<double> *d_coo_val = new d_vec_t<double>[uniq_keys.size()];
  d_vec_t<int> *d_coo_row = new d_vec_t<int>[uniq_keys.size()];
  d_vec_t<int> *d_coo_col = new d_vec_t<int>[uniq_keys.size()];
  d_vec_t<int> *d_csr_row = new d_vec_t<int>[uniq_keys.size()];

  for (int i = 0; i < uniq_keys.size(); ++i) {
    int key = uniq_keys[i];
    coo_val[i].resize(len_distance_2);
    coo_row[i].resize(len_distance_2);
    coo_col[i].resize(len_distance_2);

    for_each(
        ZIP2(ZIP2(distance_2.begin(), first_idx),
             ZIP3(coo_val[i].begin(), coo_row[i].begin(), coo_col[i].begin())),
        ZIP2(ZIP2(distance_2.end(), last_idx),
             ZIP3(coo_val[i].end(), coo_row[i].end(), coo_col[i].end())),
        createCOO(key, num_feat_2));

    coo_val[i].erase(
        remove_if(coo_val[i].begin(), coo_val[i].end(), IsLessThan(0)),
        coo_val[i].end());
    coo_row[i].erase(
        remove_if(coo_row[i].begin(), coo_row[i].end(), IsLessThan(0)),
        coo_row[i].end());
    coo_col[i].erase(
        remove_if(coo_col[i].begin(), coo_col[i].end(), IsLessThan(0)),
        coo_col[i].end());

    ///// converting COO to CSR
    d_coo_row[i] = coo_row[i];
    d_coo_col[i] = coo_col[i];
    d_coo_val[i] = coo_val[i];
    d_csr_row[i].resize(num_feat_2 + 1);

    cusparseXcoo2csr(handle, raw_pointer_cast(d_coo_row[i].data()),
                     d_coo_row[i].size(), num_feat_2,
                     raw_pointer_cast(d_csr_row[i].data()),
                     CUSPARSE_INDEX_BASE_ZERO);

    // std::cout << "\nval: " << d_coo_val[i].size() << std::endl;
    // for (int j = 0; j < d_coo_col[i].size(); ++j) {
    //  std::cout << d_coo_val[i][j] << " ";
    //}
    // std::cout << "\nrow: " << d_coo_row[i].size() << std::endl;
    // for (int j = 0; j < d_coo_col[i].size(); ++j) {
    //  std::cout << d_coo_row[i][j] << " ";
    //}
    // std::cout << "\ncol: " << d_coo_col[i].size() << std::endl;
    // for (int j = 0; j < d_coo_col[i].size(); ++j) {
    //  std::cout << d_coo_col[i][j] << " ";
    //}
    // std::cout << std::endl;
    // std::cout << "\nrow: " << d_csr_row[i].size() << std::endl;
    // for (int j = 0; j < d_csr_row[i].size(); ++j) {
    //  std::cout << d_csr_row[i][j] << " ";
    //}
    // std::cout << std::endl;
  }

  /*******************************
  *  Initialize eigen vectors.  *
  *******************************/

  int len_eigen_vec = num_feat_1 * num_feat_2;
  d_vec_t<double> eigen_vec_new(len_eigen_vec);
  // TODO: next line is probably unnecessary
  fill(eigen_vec_new.begin(), eigen_vec_new.end(), 0);

  d_vec_t<double> eigen_vec_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(eigen_vec_old.begin(), eigen_vec_old.end(), norm);

  for (int iter = 0; iter < num_iters; ++iter) {
    // Create a stream for each operation
    cudaStream_t *streams =
        (cudaStream_t *)malloc(uniq_keys.size() * sizeof(cudaStream_t));

    for (int i = 0; i < uniq_keys.size(); i++)
      cudaStreamCreate(&streams[i]);

    for (int i = 0; i < uniq_keys.size(); i++) {
      cusparseSetStream(handle, streams[i]);

      for (int j = 0; j < keys_idcs[i].size(); j++) {
        int row = keys_idcs[i][j] / num_feat_1;
        int col = keys_idcs[i][j] % num_feat_1;

        cusparseDcsrmv(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_feat_2, num_feat_2,
            d_coo_val[i].size(), &alpha, descr,
            raw_pointer_cast(d_coo_val[i].data()),
            raw_pointer_cast(d_csr_row[i].data()),
            raw_pointer_cast(d_coo_col[i].data()),
            raw_pointer_cast(eigen_vec_old.data()) + col * num_feat_2, &beta,
            raw_pointer_cast(eigen_vec_new.data()) + row * num_feat_2);
      }
    }

//    double init = 0;
//    norm =
//        std::sqrt(transform_reduce(eigen_vec_new.begin(), eigen_vec_new.end(),
//                                   square(), init, thrust::plus<double>()));
//
//    transform(eigen_vec_new.begin(), eigen_vec_new.end(), eigen_vec_old.begin(),
//              division(norm));
//
//    fill(eigen_vec_new.begin(), eigen_vec_new.end(), 0);
  }

//  for (int i = 0; i < eigen_vec_old.size(); i++) {
//    std::cout << "eigen new value = " << eigen_vec_new[i] << "
//                                                             ";
//        std::cout
//              << "eigen old value = " << eigen_vec_old[i] << std::endl;
//  }

  cusparseDestroy(handle);

  return (0);
}
