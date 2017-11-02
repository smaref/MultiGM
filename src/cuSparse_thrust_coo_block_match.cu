/* Using cuSPARSE for matrix vector multplication of completed affinity */
#include <cuda_runtime.h>
#include <cusparse.h>

#include "utils.h"

int main(int argc, char *argv[]) {

  /***********************************************
  *    initialize program's input parameters     *
  ***********************************************/
  double alpha = 1;
  double beta = 1;
  double norm = 0;
  int bin_width = 10;

  cusparseHandle_t handle = 0;
  cusparseMatDescr_t descr = 0;
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descr);

  h_vec_t<int> distance_1;
  int num_feat_1 = atoi(argv[2]);
  ReadMatrix(distance_1, argv[1], num_feat_1);
  //#ifdef ACCELERATE
  //  std::cout << "CUDA" << std::endl;
  //  d_vec_t<unsigned> d_distance_1 = distance_1;
  //#endif

  h_vec_t<double> distance_2;
  int num_feat_2 = atoi(argv[4]);
  ReadMatrix(distance_2, argv[3], num_feat_2);
  //#ifdef ACCELERATE
  //  d_vec_t<double> d_distance_2 = distance_2;
  //#endif

  int num_iters = 1;
  if (8 == argc)
    num_iters = atoi(argv[7]);

  /**************************************************
  * find unique values of distance1 and their indices
  ***************************************************/
  //#ifdef ACCELERATE
  //  d_vec_t<unsigned> d_uniq_keys = FindUniques(d_distance_1);
  //  d_uniq_keys.erase(
  //      remove_if(d_uniq_keys.begin(), d_uniq_keys.end(),
  //      IsLessThan(bin_width)),
  //      d_uniq_keys.end());
  //#else
  std::cout << "HOST" << std::endl;
  h_vec_t<unsigned> uniq_keys = FindUniques(distance_1);
  uniq_keys.erase(
      remove_if(uniq_keys.begin(), uniq_keys.end(), IsLessThan(bin_width)),
      uniq_keys.end());
  //#endif
  //
  //#ifdef ACCELERATE
  //  d_vec_t<int> *d_keys_idcs = new d_vec_t<int>[d_uniq_keys.size()];
  //  for (unsigned i = 0; i < d_uniq_keys.size(); ++i) {
  //    d_keys_idcs[i].resize(d_distance_1.size());
  //  }
  //#else
  h_vec_t<int> *keys_idcs = new h_vec_t<int>[uniq_keys.size()];
  for (unsigned i = 0; i < uniq_keys.size(); ++i) {
    keys_idcs[i].resize(distance_1.size());
  }
  //#endif

  counting_iterator<unsigned> first_idx(0);
  counting_iterator<unsigned> last_idx1 = first_idx + distance_1.size();

  //#ifdef ACCELERATE
  //  for (unsigned i = 0; i < d_uniq_keys.size(); ++i) {
  //    transform(ZIP2(d_distance_1.begin(), first_idx),
  //              ZIP2(d_distance_1.end(), last_idx), d_keys_idcs[i].begin(),
  //              IsEqual(d_uniq_keys[i]));
  //
  //    d_keys_idcs[i].erase(
  //        remove(d_keys_idcs[i].begin(), d_keys_idcs[i].end(), -1),
  //        d_keys_idcs[i].end());
  //  }
  //#else
  for (unsigned i = 0; i < uniq_keys.size(); ++i) {
    transform(ZIP2(distance_1.begin(), first_idx),
              ZIP2(distance_1.end(), last_idx1), keys_idcs[i].begin(),
              IsEqual(uniq_keys[i]));

    keys_idcs[i].erase(remove(keys_idcs[i].begin(), keys_idcs[i].end(), -1),
                       keys_idcs[i].end());
  }
  //#endif

  /***************************************************
  * construct COO sparse respresentative of affinity *
  ***************************************************/
  unsigned len_affinity_block = num_feat_2 * num_feat_2;
  counting_iterator<unsigned> last_idx2 = first_idx + len_affinity_block;

  h_vec_t<double> *h_coo_val = new h_vec_t<double>[uniq_keys.size()];
  h_vec_t<int> *h_coo_row = new h_vec_t<int>[uniq_keys.size()];
  h_vec_t<int> *h_coo_col = new h_vec_t<int>[uniq_keys.size()];

  d_vec_t<double> *d_coo_val = new d_vec_t<double>[uniq_keys.size()];
  d_vec_t<int> *d_coo_row = new d_vec_t<int>[uniq_keys.size()];
  d_vec_t<int> *d_coo_col = new d_vec_t<int>[uniq_keys.size()];
  d_vec_t<int> *d_csr_row = new d_vec_t<int>[uniq_keys.size()];

  for (int i = 0; i < uniq_keys.size(); ++i) {
    int key = uniq_keys[i];
    h_coo_val[i].resize(len_affinity_block);
    h_coo_row[i].resize(len_affinity_block);
    h_coo_col[i].resize(len_affinity_block);

    for_each(
        ZIP2(ZIP2(distance_2.begin(), first_idx),
             ZIP3(h_coo_val[i].begin(), h_coo_row[i].begin(),
                  h_coo_col[i].begin())),
        ZIP2(ZIP2(distance_2.end(), last_idx2),
             ZIP3(h_coo_val[i].end(), h_coo_row[i].end(), h_coo_col[i].end())),
        createCOO(key, num_feat_2));

    h_coo_val[i].erase(
        remove_if(h_coo_val[i].begin(), h_coo_val[i].end(), IsLessThan(0)),
        h_coo_val[i].end());
    h_coo_row[i].erase(
        remove_if(h_coo_row[i].begin(), h_coo_row[i].end(), IsLessThan(0)),
        h_coo_row[i].end());
    h_coo_col[i].erase(
        remove_if(h_coo_col[i].begin(), h_coo_col[i].end(), IsLessThan(0)),
        h_coo_col[i].end());

    d_coo_val[i] = h_coo_val[i];
    d_coo_row[i] = h_coo_row[i];
    d_coo_col[i] = h_coo_col[i];
    d_csr_row[i].resize(num_feat_2 + 1);

    cusparseXcoo2csr(handle, raw_pointer_cast(d_coo_row[i].data()),
                     d_coo_row[i].size(), num_feat_2,
                     raw_pointer_cast(d_csr_row[i].data()),
                     CUSPARSE_INDEX_BASE_ZERO);
  }

  // make_tuple(h_coo_val, h_coo_row, h_coo_col);
  std::cout << "affinity" << std::endl;
  for (int i = 0; i < uniq_keys.size(); ++i) {
    std::cout << " unq keys: " << uniq_keys[i] << std::endl;
    std::cout << " values " << " " << "columns" << "  " << "rows" << std::endl;
    for (int j = 0; j < h_coo_val[i].size(); ++j) {
      std::cout << h_coo_val[i][j] << "    " << h_coo_col[i][j]
                << "       " << h_coo_row[i][j] << std::endl;
    }
  }
  std::cout << std::endl;

  //  cusparseDestroy(handle);

  /******************************************************
  *             initialize eigen vectors                *
  ******************************************************/
  // cusparseCreate(&handle);
  int len_eigen_vec = num_feat_1 * num_feat_2;

  d_vec_t<double> d_eigen_vec_new(len_eigen_vec);
  d_vec_t<double> d_eigen_vec_old(len_eigen_vec);

  norm = 1.0 / sqrt(len_eigen_vec);
  fill(d_eigen_vec_old.begin(), d_eigen_vec_old.end(), norm);

  /******************************************************
  *               compute eigen vectors                 *
  ******************************************************/
  for (int iter = 0; iter < num_iters; ++iter) {
    // Create a stream for each operation
    cudaStream_t *streams =
        (cudaStream_t *)malloc(uniq_keys.size() * sizeof(cudaStream_t));

    for (int i = 0; i < uniq_keys.size(); i++)
      cudaStreamCreate(&streams[i]);

    for (int i = 0; i < uniq_keys.size(); ++i) {
      cusparseSetStream(handle, streams[i]);
      for (int j = 0; j < keys_idcs[i].size(); ++j) {
        int row = keys_idcs[i][j] / num_feat_1;
        int col = keys_idcs[i][j] % num_feat_1;

        cusparseDcsrmv(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_feat_2, num_feat_2,
            d_coo_val[i].size(), &alpha, descr,
            raw_pointer_cast(d_coo_val[i].data()),
            raw_pointer_cast(d_csr_row[i].data()),
            raw_pointer_cast(d_coo_col[i].data()),
            raw_pointer_cast(d_eigen_vec_old.data()) + col * num_feat_2, &beta,
            raw_pointer_cast(d_eigen_vec_new.data()) + row * num_feat_2);
      }
    }
    double init = 0;
    norm = std::sqrt(transform_reduce(d_eigen_vec_new.begin(),
                                      d_eigen_vec_new.end(), square(), init,
                                      thrust::plus<double>()));

    transform(d_eigen_vec_new.begin(), d_eigen_vec_new.end(),
              d_eigen_vec_old.begin(), division(norm));

    fill(d_eigen_vec_new.begin(), d_eigen_vec_new.end(), 0);
  }

  for (int i = 0; i < d_eigen_vec_old.size(); i++) {
    std::cout << "d_eigen new value = " << d_eigen_vec_new[i] << "  ";
    std::cout << "d_eigen old value = " << d_eigen_vec_old[i] << std::endl;
  }

  cusparseDestroy(handle);

  return 0;
}
