/* Using cuBlas library for matrix vector multplication of block_match
 * technique.*/
#include "cublas_v2.h"
#include <algorithm>
#include <cuda_runtime.h>

#include "utils.h"
#include <time.h>  

int main(int argc, char *argv[]) {

  /***********************************************
  *   initialize program's input parameters  *
  ***********************************************/
  double alpha = 1;
  double beta = 1;
  double norm = 0;
  unsigned bin_width = 10;

  h_vec_t<unsigned> distance_1;
  unsigned num_feat_1 = atoi(argv[2]);
  ReadMatrix(distance_1, argv[1], num_feat_1);
#ifdef ACCELERATE
  std::cout << "CUDA" << std::endl;
  d_vec_t<unsigned> d_distance_1 = distance_1;
#endif

  h_vec_t<double> distance_2;
  unsigned num_feat_2 = atoi(argv[4]);
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
  // std::cout << "HOST" << std::endl;
  h_vec_t<unsigned> uniq_keys = FindUniques(distance_1);
  uniq_keys.erase(
      remove_if(uniq_keys.begin(), uniq_keys.end(), IsLessThan(bin_width)),
      uniq_keys.end());
#endif

#ifdef ACCELERATE
  d_vec_t<int> *d_keys_idcs = new d_vec_t<int>[d_uniq_keys.size()];
  for (unsigned i = 0; i < d_uniq_keys.size(); ++i) {
    d_keys_idcs[i].resize(d_distance_1.size());
  }
#else
  h_vec_t<int> *keys_idcs = new h_vec_t<int>[uniq_keys.size()];
  for (unsigned i = 0; i < uniq_keys.size(); ++i) {
    keys_idcs[i].resize(distance_1.size());
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
  for (unsigned i = 0; i < uniq_keys.size(); ++i) {
    transform(ZIP2(distance_1.begin(), first_idx),
              ZIP2(distance_1.end(), last_idx), keys_idcs[i].begin(),
              IsEqual(uniq_keys[i]));

    keys_idcs[i].erase(remove(keys_idcs[i].begin(), keys_idcs[i].end(), -1),
                       keys_idcs[i].end());
  }
#endif

  /************************************************
  *         construct affinity blocks             *
  ************************************************/
  double *distance2 = raw_pointer_cast(distance_2.data());
  double *distance3 = raw_pointer_cast(distance_3.data());

  unsigned block_size = num_feat_2 * num_feat_3 * num_feat_2 * num_feat_3;
  d_vec_t<double> *d_affinity_blocks = new d_vec_t<double>[uniq_keys.size()];

#ifdef ACCELERATE
  double *d_distance2 = raw_pointer_cast(d_distance_2.data());
  double *d_distance3 = raw_pointer_cast(d_distance_3.data());
  d_vec_t<double> *d_affinity_blocks = new d_vec_t<double>[d_uniq_keys.size()];

  for (int i = 0; i < d_uniq_keys.size(); ++i) {
    unsigned key = d_uniq_keys[i];
    d_affinity_blocks[i] =
        AffinityBlocks(d_distance2, d_distance3, key, num_feat_2, num_feat_3);
  }
#else
  h_vec_t<double> *affinity_blocks = new h_vec_t<double>[uniq_keys.size()];

  const clock_t begin_time = clock();
  
  for (int i = 0; i < uniq_keys.size(); ++i) {
    unsigned key = uniq_keys[i];
    affinity_blocks[i] =
        AffinityBlocks(distance2, distance3, key, num_feat_2, num_feat_3);
    d_affinity_blocks[i] = affinity_blocks[i];
  }

  std::cout << "affinity runtime: "
            << float(clock() - begin_time) / CLOCKS_PER_SEC * 1000 << std::endl;
  
#endif

  //std::cout << uniq_keys.size() << " " << block_size << std::endl;
  //for (int i = 0; i < uniq_keys.size(); ++i) {
  //  for (int j = 0; j < affinity_blocks[i].size(); ++j)
  //    std::cout << affinity_blocks[i][j] << " ";
  //  std::cout << std::endl;
  //}
  //std::cout << std::endl;

  /************************************************
  *           initialize eigen vectors            *
  ************************************************/
  unsigned len_eigen_vec = num_feat_1 * num_feat_2 * num_feat_3;

  d_vec_t<double> d_eigen_vec_new(len_eigen_vec);
  d_vec_t<double> d_eigen_vec_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(d_eigen_vec_old.begin(), d_eigen_vec_old.end(), norm);

#if ACCELERATE
  int num_keys = d_uniq_keys.size();
#else
  int num_keys = uniq_keys.size();
#endif

  cublasHandle_t handle;
  cublasCreate(&handle);

  /*******************************
  *  compute eigen vectors.  *
  *******************************/
  const clock_t begin_time2 = clock();
  
  for (int iter = 0; iter < num_iters; ++iter) {
    // Create a stream for each operation
    cudaStream_t *streams =
        (cudaStream_t *)malloc(num_keys * sizeof(cudaStream_t));

    for (int i = 0; i < num_keys; i++)
      cudaStreamCreate(&streams[i]);

    for (int i = 0; i < num_keys; i++) {
      // Set CUDA stream
      cublasSetStream(handle, streams[i]);

#ifdef ACCELERATE
      for (int j = 0; j < d_keys_idcs[i].size(); j++) {
        int row = d_keys_idcs[i][j] / num_feat_1;
        int col = d_keys_idcs[i][j] % num_feat_1;
#else
      for (int j = 0; j < keys_idcs[i].size(); j++) {
        int row = keys_idcs[i][j] / num_feat_1;
        int col = keys_idcs[i][j] % num_feat_1;
#endif
  

        cublasDgemv(handle, CUBLAS_OP_N, num_feat_2 * num_feat_3,
                    num_feat_2 * num_feat_3, &alpha,
                    raw_pointer_cast(d_affinity_blocks[i].data()),
                    num_feat_2 * num_feat_3,
                    raw_pointer_cast(d_eigen_vec_old.data()) +
                        col * num_feat_2 * num_feat_3,
                    1, &beta, raw_pointer_cast(d_eigen_vec_new.data()) +
                                  row * num_feat_2 * num_feat_3,
                    1);
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
  
    std::cout << "Eigen runtime: "
            << float(clock() - begin_time2) / CLOCKS_PER_SEC * 1000 << std::endl;

//  std::cout << "eigen values" << std::endl;
//  for (int i = 0; i < d_eigen_vec_old.size(); i++) {
//    std::cout << "d_eigen new value = " << d_eigen_vec_new[i] << "  ";
//    std::cout << "d_eigen old value = " << d_eigen_vec_old[i] << std::endl;
//  }

  cublasDestroy(handle);

  return 0;
}
