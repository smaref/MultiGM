/* Using cuSPARSE for matrix vector multplication of block_match technique. */

#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "utils.h"
#include "time.h"

int main(int argc, char *argv[]) {

  /***********************************************
  *     initialize program's input parameters    *
  ***********************************************/
  double alpha = 1;
  double beta = 1;
  double norm = 0;
  unsigned bin_width = 10;

  cusparseHandle_t handle = 0;
  cusparseMatDescr_t descr = 0;
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descr);

  h_vec_t<unsigned> h_distance_1;
  unsigned num_feat_1 = atoi(argv[2]);
  ReadMatrix(h_distance_1, argv[1], num_feat_1);
#ifdef ACCELERATE
  std::cout << "CUDA" << std::endl;
  d_vec_t<unsigned> d_distance_1 = distance_1;
#endif

  h_vec_t<double> h_distance_2;
  unsigned num_feat_2 = atoi(argv[4]);
  ReadMatrix(h_distance_2, argv[3], num_feat_2);
#ifdef ACCELERATE
  d_vec_t<double> d_distance_2 = distance_2;
#endif
  
  h_vec_t<double> h_distance_3;
  int num_feat_3 = atoi(argv[6]);
  ReadMatrix(h_distance_3, argv[5], num_feat_3);
#ifdef ACCELERATE
  d_vec_t<double> d_distance_3 = distance_3;
#endif

  int num_iters = 20;
  if (10 == argc)
    num_iters = atoi(argv[9]);


/**************************************************
* find unique values of distance1 and their indices
***************************************************/
#ifdef ACCELERATE
  d_vec_t<unsigned> d_uniq_keys = FindUniques(d_distance_1);
  d_uniq_keys.erase(
      remove_if(d_uniq_keys.begin(), d_uniq_keys.end(), IsLessThan(bin_width)),
      d_uniq_keys.end());
#else
  //std::cout << "HOST" << std::endl;
  h_vec_t<unsigned> h_uniq_keys = FindUniques(h_distance_1);
  h_uniq_keys.erase(
      remove_if(h_uniq_keys.begin(), h_uniq_keys.end(), IsLessThan(bin_width)),
      h_uniq_keys.end());
#endif

#ifdef ACCELERATE
  d_vec_t<int> *d_keys_idcs = new d_vec_t<int>[d_uniq_keys.size()];
  for (unsigned i = 0; i < d_uniq_keys.size(); ++i) {
    d_keys_idcs[i].resize(d_distance_1.size());
  }
#else
  h_vec_t<int> *h_keys_idcs = new h_vec_t<int>[h_uniq_keys.size()];
  for (unsigned i = 0; i < h_uniq_keys.size(); ++i) {
    h_keys_idcs[i].resize(h_distance_1.size());
  }
#endif

  counting_iterator<unsigned> first_idx(0);
  counting_iterator<unsigned> last_idx = first_idx + num_feat_1;

#ifdef ACCELERATE
  for (unsigned i = 0; i < d_uniq_keys.size(); ++i) {
    transform(ZIP2(d_distance_1.begin(), first_idx),
              ZIP2(d_distance_1.end(), last_idx), d_keys_idcs[i].begin(),
              IsEqual(d_uniq_keys[i]));

    d_keys_idcs[i].erase(
        remove(d_keys_idcs[i].begin(), d_keys_idcs[i].end(), -1),
        d_keys_idcs[i].end());
  }
#else
  for (unsigned i = 0; i < h_uniq_keys.size(); ++i) {
    transform(ZIP2(h_distance_1.begin(), first_idx),
              ZIP2(h_distance_1.end(), last_idx), h_keys_idcs[i].begin(),
              IsEqual(h_uniq_keys[i]));

    h_keys_idcs[i].erase(
        remove(h_keys_idcs[i].begin(), h_keys_idcs[i].end(), -1),
        h_keys_idcs[i].end());
  }
#endif

  /***************************************************
  *      construct CSR sparse affinity blocks        *
  ***************************************************/

  //#ifdef ACCELERATE
  //  d_vec_t<double> d_affinity_blocks(d_uniq_keys.size() *
  //  len_affinity_block);
  //#else
  //#endif
  //
  //#ifdef ACCELERATE
  //  d_vec_t<double> csr_val;
  //  d_vec_t<int> csr_col;
  //  d_vec_t<int> csr_row;
  //  d_vec_t<int> csr_blocked_len;
  //
  //  for (int i = 0; i < d_uniq_keys.size(); ++i) {
  //    transform(d_distance_2.begin(), d_distance_2.end(),
  //              d_affinity_blocks.begin() + i * len_affinity_block,
  //              Affinity(d_uniq_keys[i]));
  //
  //    CompressMatrix(csr_val, csr_col, csr_row,
  //                   raw_pointer_cast(d_affinity_blocks.begin()) +
  //                       i * len_affinity_block,
  //                   num_feat_2, num_feat_2);
  //
  //    csr_blocked_len.push_back(csr_val.size());
  //  }
  //#else
  double *distance2 = raw_pointer_cast(h_distance_2.data());
  double *distance3 = raw_pointer_cast(h_distance_3.data());

  unsigned len_affinity_block = num_feat_2 * num_feat_3 * num_feat_2 * num_feat_3;
  
  h_vec_t<double> *affinity_blocks = new h_vec_t<double>[h_uniq_keys.size()];
  
  h_vec_t<double> csr_val;
  h_vec_t<int> csr_col;
  h_vec_t<int> csr_row;
  h_vec_t<int> csr_blocked_len;
  csr_blocked_len.push_back(0);
  
  const clock_t begin_time = clock();
  
  for (int i = 0; i < h_uniq_keys.size(); ++i) {
    unsigned key = h_uniq_keys[i];
    affinity_blocks[i] =
        AffinityBlocks(distance2, distance3, key, num_feat_2, num_feat_3);

    CompressMatrix(csr_val, csr_col, csr_row,
                   raw_pointer_cast(affinity_blocks[i].data()),
                   num_feat_2 * num_feat_3, num_feat_2 * num_feat_3);

    csr_blocked_len.push_back(csr_val.size());
  }

  std::cout << "affinity runtime: "
            << float(clock() - begin_time) / CLOCKS_PER_SEC * 1000 << std::endl;
  
  d_vec_t<double> d_csr_val = csr_val;
  d_vec_t<int> d_csr_col = csr_col;
  d_vec_t<int> d_csr_row = csr_row;
  //#endif
    
  //std::cout << "values"
    //          << "  "
    //          << "columns" << std::endl;
 // for (int i = 0; i < h_uniq_keys.size(); ++i) {
 //   for (int j = csr_blocked_len[i]; j < csr_blocked_len[i + 1]; ++j) {
 //     std::cout << csr_val[j] << "   " << csr_col[j] << "  " << std::endl;
 //   }
 //   std::cout << std::endl;
 // }
 // std::cout << std::endl;

  /******************************************************
  *             initialize eigen vectors                *
  ******************************************************/
  unsigned len_eigen_vec = num_feat_1 * num_feat_2 * num_feat_3;

  d_vec_t<double> eigen_vec_new(len_eigen_vec);
  d_vec_t<double> eigen_vec_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(eigen_vec_old.begin(), eigen_vec_old.end(), norm);

  //#if ACCELERATE
  //  int num_keys = d_uniq_keys.size();
  //#else
  int num_keys = h_uniq_keys.size();
  //#endif

  /*******************************************************
  *                 compute eigen values                 *
  ********************************************************/
  const clock_t begin_time2 = clock();
  
  for (int iter = 0; iter < num_iters; ++iter) {
    // Create a stream for each operation
    cudaStream_t *streams =
        (cudaStream_t *)malloc(num_keys * sizeof(cudaStream_t));

    for (int i = 0; i < num_keys; i++)
      cudaStreamCreate(&streams[i]);

    for (int i = 0; i < num_keys; i++) {
      cusparseSetStream(handle, streams[i]);
      int csr_size = csr_blocked_len[i + 1] - csr_blocked_len[i];
      //#ifdef ACCELERATE
      //      for (int j = 0; j < d_keys_idcs[i].size(); j++) {
      //        int row = d_keys_idcs[i][j] / num_feat_1;
      //        int col = d_keys_idcs[i][j] % num_feat_1;
      //#else
      for (int j = 0; j < h_keys_idcs[i].size(); j++) {
        int row = h_keys_idcs[i][j] / num_feat_1;
        int col = h_keys_idcs[i][j] % num_feat_1;
        //#endif

        cusparseDcsrmv(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_feat_2 * num_feat_3, num_feat_2 * num_feat_3,
            csr_size, &alpha, descr,
            raw_pointer_cast(d_csr_val.data() + csr_blocked_len[i]),
            raw_pointer_cast(d_csr_row.data() + i * (num_feat_2 * num_feat_3 + 1)),
            raw_pointer_cast(d_csr_col.data() + csr_blocked_len[i]),
            raw_pointer_cast(eigen_vec_old.data()) + col * num_feat_2 * num_feat_3, &beta,
            raw_pointer_cast(eigen_vec_new.data()) + row * num_feat_2 * num_feat_3);
      }
    }

    double init = 0;
    norm =
        std::sqrt(transform_reduce(eigen_vec_new.begin(), eigen_vec_new.end(),
                                   square(), init, thrust::plus<double>()));

    transform(eigen_vec_new.begin(), eigen_vec_new.end(), eigen_vec_old.begin(),
              division(norm));

    fill(eigen_vec_new.begin(), eigen_vec_new.end(), 0);
  }

  std::cout << "Eigen runtime: "
            << float(clock() - begin_time2) / CLOCKS_PER_SEC * 1000 << std::endl;

//  std::cout << "eigen values" << std::endl;
//  for (int i = 0; i < eigen_vec_old.size(); i++) {
//    std::cout << "eigen new value = " << eigen_vec_new[i] << "";
//    std::cout << "eigen old value = " << eigen_vec_old[i] << std::endl;
//  }

    cusparseDestroy(handle);

  return 0;
}
