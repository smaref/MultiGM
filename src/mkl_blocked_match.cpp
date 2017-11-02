#include "utils.h"
#include <mkl.h>
#include <time.h>

int main(int argc, char *argv[]) {

  /***********************************************
  *    initialize program's input parameters  *
  ***********************************************/
  double norm = 0;

  int bin_width = 10;

  h_vec_t<int> distance_1;
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

  /****************************************************
  * find unique values og distance1 and their indeces *
  *****************************************************/
  h_vec_t<int> uniq_keys = FindUniques(distance_1);

  uniq_keys.erase(
      remove_if(uniq_keys.begin(), uniq_keys.end(), IsLessThan(bin_width)),
      uniq_keys.end());

  h_vec_t<int> *h_keys_idcs = new h_vec_t<int>[uniq_keys.size()];
  for (int i = 0; i < uniq_keys.size(); ++i) {
    h_keys_idcs[i].resize(distance_1.size());
  }

  counting_iterator<int> first_idx(0);
  counting_iterator<int> last_idx = first_idx + num_feat_1;

  h_zip_iter_t<int, int> first =
      make_zip_iterator(make_tuple(distance_1.begin(), first_idx));
  h_zip_iter_t<int, int> last =
      make_zip_iterator(make_tuple(distance_1.end(), last_idx));
  for (int i = 0; i < uniq_keys.size(); ++i) {
    transform(first, last, h_keys_idcs[i].begin(), IsEqual(uniq_keys[i]));
    h_keys_idcs[i].erase(
        remove(h_keys_idcs[i].begin(), h_keys_idcs[i].end(), -1),
        h_keys_idcs[i].end());
  }

  // std::cout << uniq_keys.size() << std::endl;
  // for (int i = 0; i < uniq_keys.size(); ++i) {
  //     //std::cout << "keys: " << uniq_keys[i] << std::endl;
  //     for (int j = 0; j < h_keys_idcs[i].size(); ++j) {
  //       std::cout << h_keys_idcs[i][j] << " ";
  //     }
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;

  /**************************************************
  *     construct affinity matrix blocks            *
  ***************************************************/
  double *distance2 = raw_pointer_cast(distance_2.data());
  double *distance3 = raw_pointer_cast(distance_3.data());

  int block_size = num_feat_2 * num_feat_3 * num_feat_2 * num_feat_3;

  h_vec_t<double> *affinity_blocks = new h_vec_t<double>[uniq_keys.size()];
  h_vec_t<double> affinity(block_size);
  // double *affinity = new double[distance_2.size() * distance_3.size()];

  const clock_t begin_time = clock();
  
  for (int i = 0; i < uniq_keys.size(); ++i) {
    unsigned key = uniq_keys[i];
    //    std::cout << "key: " << key << std::endl;
    affinity_blocks[i] =
        AffinityBlocks(distance2, distance3, key, num_feat_2, num_feat_3);
  }

  std::cout << "affinity runtime: "
            << float(clock() - begin_time) / CLOCKS_PER_SEC * 1000 << std::endl;
  
 // std::cout << uniq_keys.size() << " " << block_size << std::endl;
 // for (int i = 0; i < uniq_keys.size(); ++i) {
 //   for (int j = 0; j < affinity_blocks[i].size(); ++j)
 //     std::cout << affinity_blocks[i][j] << " ";
 //   std::cout << std::endl;
 // }
 // std::cout << std::endl;

  /************************************************
  *           initialize eigen vectors            *
  ************************************************/
  int len_eigen_vec = num_feat_1 * num_feat_2 * num_feat_3;
  h_vec_t<double> h_eigen_new(len_eigen_vec);
  fill(h_eigen_new.begin(), h_eigen_new.end(), 0);
  int num_keys = uniq_keys.size();

  h_vec_t<double> h_eigen_old(len_eigen_vec);
  norm = 1.0 / sqrt(len_eigen_vec);
  fill(h_eigen_old.begin(), h_eigen_old.end(), norm);

  /************************************************
  *           computing eigen vector            *
  ************************************************/
  const clock_t begin_time2 = clock();
  
  for (int iter = 0; iter < num_iters; ++iter) {
    for (int i = 0; i < num_keys; i++) {
      for (int j = 0; j < h_keys_idcs[i].size(); j++) {
        int row = h_keys_idcs[i][j] / num_feat_1;
        int col = h_keys_idcs[i][j] % num_feat_1;

        cblas_dgemv(CblasRowMajor, CblasNoTrans, num_feat_2 * num_feat_3,
                    num_feat_2 * num_feat_3, 1,
                    raw_pointer_cast(affinity_blocks[i].data()),
                    num_feat_2 * num_feat_3, raw_pointer_cast(h_eigen_old.data()) +
                                    col * num_feat_2 * num_feat_3,
                    1, 1, raw_pointer_cast(h_eigen_new.data()) +
                              row * num_feat_2 * num_feat_3,
                    1);
      }
    }

    double init = 0;
    norm = std::sqrt(transform_reduce(h_eigen_new.begin(), h_eigen_new.end(),
                                      square(), init, thrust::plus<double>()));

    transform(h_eigen_new.begin(), h_eigen_new.end(), h_eigen_old.begin(),
              division(norm));

    fill(h_eigen_new.begin(), h_eigen_new.end(), 0);
  }
  
  std::cout << "Eigen runtime: "
            << float(clock() - begin_time2) / CLOCKS_PER_SEC * 1000 << std::endl;

//  std::cout << "eigen values" << std::endl;
//  for (int i = 0; i < h_eigen_old.size(); i++) {
//    std::cout << "eigen new value = " << h_eigen_new[i] << "  ";
//    std::cout << "eigen old value = " << h_eigen_old[i] << std::endl;
//  }

  return (0);
}
