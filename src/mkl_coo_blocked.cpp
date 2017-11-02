/* Using cuSPARSE for matrix vector multplication of completed affinity */
#include "utils.h"
#include <mkl.h>
#include <time.h>


int main(int argc, char *argv[]) {

  /***********************************************
  *    initialize program's input parameters     *
  ***********************************************/
  double norm = 0;
  double one = 1;
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

  /**************************************************
  * find unique values of distance1 and their indices
  ***************************************************/
  h_vec_t<unsigned> uniq_keys = FindUniques(distance_1);
  uniq_keys.erase(
      remove_if(uniq_keys.begin(), uniq_keys.end(), IsLessThan(bin_width)),
      uniq_keys.end());

  h_vec_t<int> *keys_idcs = new h_vec_t<int>[uniq_keys.size()];
  for (unsigned i = 0; i < uniq_keys.size(); ++i) {
    keys_idcs[i].resize(distance_1.size());
  }

  counting_iterator<unsigned> first_idx(0);
  counting_iterator<unsigned> last_idx1 = first_idx + distance_1.size();

  for (unsigned i = 0; i < uniq_keys.size(); ++i) {
    transform(ZIP2(distance_1.begin(), first_idx),
              ZIP2(distance_1.end(), last_idx1), keys_idcs[i].begin(),
              IsEqual(uniq_keys[i]));

    keys_idcs[i].erase(remove(keys_idcs[i].begin(), keys_idcs[i].end(), -1),
                       keys_idcs[i].end());
  }

  /***************************************************
  * construct COO sparse respresentative of affinity *
  ***************************************************/
  double *distance2_ptr = raw_pointer_cast(distance_2.data());
  double *distance3_ptr = raw_pointer_cast(distance_3.data());
  unsigned len_affinity_block =
      num_feat_2 * num_feat_3 * num_feat_2 * num_feat_3;

  h_vec_t<double> *h_coo_value = new h_vec_t<double>[uniq_keys.size()];
  h_vec_t<int> *h_coo_row = new h_vec_t<int>[uniq_keys.size()];
  h_vec_t<int> *h_coo_column = new h_vec_t<int>[uniq_keys.size()];

  const clock_t begin_time = clock();
  
  for (int i = 0; i < uniq_keys.size(); ++i) {
    unsigned key = uniq_keys[i];
    stdvec_tuple_t affanity_coo = AffinityBlocksCoo(
        distance2_ptr, distance3_ptr, key, num_feat_2, num_feat_3);

    h_coo_value[i] = get<0>(affanity_coo);
    h_coo_column[i] = (get<1>(affanity_coo));
    h_coo_row[i] = (get<2>(affanity_coo));
  }
  std::cout << "affinity runtime: "
            << float(clock() - begin_time) / CLOCKS_PER_SEC * 1000 << std::endl;

  // make_tuple(h_coo_val, h_coo_row, h_coo_col);

  // std::cout << "affinity" << std::endl;
  // std::cout << "values"
  //           << "  "
  //           << "columns"
  //           << "  "
  //           << "rows" << std::endl;

 // std::cout << uniq_keys.size() << std::endl;
 // for (int i = 0; i < uniq_keys.size(); ++i) {
 //   // std::cout << " unq keys: " << uniq_keys[i] << std::endl;
 //   for (int j = 0; j < h_coo_value[i].size(); ++j) {
 //     std::cout << h_coo_value[i][j] << "   " << h_coo_column[i][j] << "      "
 //               << h_coo_row[i][j] << std::endl;
 //   }
 //   std::cout << std::endl;
 // }

 // std::cout << std::endl;

  /******************************************************
  *             initialize eigen vectors                *
  ******************************************************/
  // cusparseCreate(&handle);
  int len_eigen_vec = num_feat_1 * num_feat_2 * num_feat_3;

  h_vec_t<double> h_eigen_vec_new(len_eigen_vec);
  h_vec_t<double> h_eigen_vec_old(len_eigen_vec);

  norm = 1.0 / sqrt(len_eigen_vec);
  fill(h_eigen_vec_old.begin(), h_eigen_vec_old.end(), norm);

  /*******************************************************
  *                 compute eigen values                 *
  ********************************************************/
  const clock_t begin_time2 = clock();
  
  for (int iter = 0; iter < num_iters; ++iter) {
    for (int i = 0; i < uniq_keys.size(); ++i) {
      for (int j = 0; j < keys_idcs[i].size(); ++j) {
        int row = keys_idcs[i][j] / num_feat_1;
        int col = keys_idcs[i][j] % num_feat_1;

        int nnz = h_coo_value[i].size();
        mkl_dcoomv("N", &len_eigen_vec, &len_eigen_vec, &one, "G**C",
                   h_coo_value[i].data(), h_coo_row[i].data(),
                   h_coo_column[i].data(), &nnz,
                   raw_pointer_cast(h_eigen_vec_old.data()) + col * num_feat_2 * num_feat_3, &one,
                   raw_pointer_cast(h_eigen_vec_new.data()) + row * num_feat_2 * num_feat_3);
      }
    }
    double init = 0;
    norm = std::sqrt(transform_reduce(h_eigen_vec_new.begin(),
                                      h_eigen_vec_new.end(), square(), init,
                                      thrust::plus<double>()));

    transform(h_eigen_vec_new.begin(), h_eigen_vec_new.end(),
              h_eigen_vec_old.begin(), division(norm));

    fill(h_eigen_vec_new.begin(), h_eigen_vec_new.end(), 0);
  }

  std::cout << "Eigen runtime: "
            << float(clock() - begin_time2) / CLOCKS_PER_SEC * 1000 << std::endl;
  
 // std::cout << "eigen values" << std::endl;
 // for (int i = 0; i < h_eigen_vec_old.size(); i++) {
 //   std::cout << "h_eigen new value = " << h_eigen_vec_new[i] << "  ";
 //   std::cout << "h_eigen old value = " << h_eigen_vec_old[i] << std::endl;
 // }

  return 0;
}
