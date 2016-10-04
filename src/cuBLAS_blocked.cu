#include "cublas_v2.h"
#include <algorithm>
#include <cuda_runtime.h>

#include "utils.h"

int main(int argc, char *argv[]) {

  /***********************************************
  *  (1) initialize program's input parameters  *
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
  d_vec_t<unsigned> d_distance_2 = distance_2;
#endif

  unsigned num_iters = 10;
  if (8 == argc)
    num_iters = atoi(argv[7]);

#ifdef ACCELERATE
  d_vec_t<unsigned> d_uniq_keys = FindUniques(d_distance_1);
  d_uniq_keys.erase(
      remove_if(d_uniq_keys.begin(), d_uniq_keys.end(), IsLessThan(bin_width)),
      d_uniq_keys.end());
#else
  std::cout << "HOST" << std::endl;
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
  d_zip_iter_t<unsigned, unsigned> first =
      make_zip_iterator(make_tuple(d_distance_1.begin(), first_idx));
  d_zip_iter_t<unsigned, unsigned> last =
      make_zip_iterator(make_tuple(d_distance_1.end(), last_idx));
#else
  h_zip_iter_t<unsigned, unsigned> first =
      make_zip_iterator(make_tuple(distance_1.begin(), first_idx));
  h_zip_iter_t<unsigned, unsigned> last =
      make_zip_iterator(make_tuple(distance_1.end(), last_idx));
#endif
#ifdef ACCELERATE
  for (unsigned i = 0; i < d_uniq_keys.size(); ++i) {
    transform(first, last, d_keys_idcs[i].begin(), IsEqual(d_uniq_keys[i]));
    d_keys_idcs[i].erase(
        remove(d_keys_idcs[i].begin(), d_keys_idcs[i].end(), -1),
        d_keys_idcs[i].end());
  }
#else
  for (unsigned i = 0; i < uniq_keys.size(); ++i) {
    transform(first, last, keys_idcs[i].begin(), IsEqual(uniq_keys[i]));
    keys_idcs[i].erase(remove(keys_idcs[i].begin(), keys_idcs[i].end(), -1),
                       keys_idcs[i].end());
  }
#endif

  unsigned len_distance_2 = num_feat_2 * num_feat_2;
#ifdef ACCELERATE

  d_vec_t<double> d_affinity_blocks(d_uniq_keys.size() * len_distance_2);
  for (int i = 0; i < d_uniq_keys.size(); ++i) {
    transform(d_distance_2.begin(), d_distance_2.end(),
              d_affinity_blocks.begin() + i * len_distance_2,
              Affinity(d_uniq_keys[i]));
  }
#else
  h_vec_t<double> affinity_blocks(uniq_keys.size() * len_distance_2);
  for (int i = 0; i < uniq_keys.size(); ++i) {
    transform(distance_2.begin(), distance_2.end(),
              affinity_blocks.begin() + i * len_distance_2,
              Affinity(uniq_keys[i]));
  }
#endif

  unsigned len_eigen_vec = num_feat_1 * num_feat_2;
  d_vec_t<double> eigen_vec_new(len_eigen_vec);
  fill(eigen_vec_new.begin(), eigen_vec_new.end(), 0);

  d_vec_t<double> eigen_vec_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(eigen_vec_old.begin(), eigen_vec_old.end(), norm);
#if ACCELERATE
  int num_keys = d_uniq_keys.size();
#else
  int num_keys = uniq_keys.size();
  d_vec_t<double> d_affinity_blocks = affinity_blocks;
#endif
  cublasHandle_t handle;
  cublasCreate(&handle);

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
        cublasDgemv(
            handle, CUBLAS_OP_N, num_feat_2, num_feat_2, &alpha,
            raw_pointer_cast(d_affinity_blocks.data()) + i * len_distance_2,
            num_feat_2,
            raw_pointer_cast(eigen_vec_old.data()) + col * num_feat_2, 1, &beta,
            raw_pointer_cast(eigen_vec_new.data()) + row * num_feat_2, 1);
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

//  for (int i = 0; i < eigen_vec_old.size(); i++) {
//    std::cout << "eigen new value = " << eigen_vec_new[i] << "  ";
//    std::cout << "eigen old value = " << eigen_vec_old[i] << std::endl;
//  }

  cublasDestroy(handle);

  return (0);
}
