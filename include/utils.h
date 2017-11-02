/**
 * \file utils.h
 * \brief Utility functions.
 * \authors S. Navaz, R. Tohid
 */

#ifndef INCLUDE_UTILS_H_
#define INCLUDE_UTILS_H_

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
using namespace thrust;

#define SIGMA 3.0

#define ZIP2(X, Y) make_zip_iterator(make_tuple(X, Y))
#define ZIP3(X, Y, Z) make_zip_iterator(make_tuple(X, Y, Z))

template <class T> using h_vec_t = host_vector<T>;
template <class T> using d_vec_t = device_vector<T>;
template <class T> using duplex_tuple_t = tuple<T, T>;
template <class T> using h_iter_t = typename h_vec_t<T>::iterator;
template <class T> using d_iter_t = typename d_vec_t<T>::iterator;
template <typename T1, typename T2>
using h_zip_iter_t =
    zip_iterator<tuple<typename h_vec_t<T1>::iterator, counting_iterator<T2>>>;
template <typename T1, typename T2>
using d_zip_iter_t =
    zip_iterator<tuple<typename d_vec_t<T1>::iterator, counting_iterator<T2>>>;
typedef tuple<double, int, int> triple_tuple_t;

typedef tuple<h_vec_t<double>, h_vec_t<int>, h_vec_t<int>> vec_tuple_t;
typedef std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>
    stdvec_tuple_t;

struct IsZero {
  __host__ __device__ int operator()(const triple_tuple_t &x) const {
    return (x.get<0>() == 0);
  }
};

struct IsEqual : public unary_function<duplex_tuple_t<int>, int> {
  int key;
  IsEqual(int _key) : key(_key) {}
  __host__ __device__ int operator()(const duplex_tuple_t<int> &x) const {
    if (x.get<0>() == key)
      return x.get<1>();
    else
      return -1;
  }
};

struct IsLessThan {
  int value;
  IsLessThan(int _value) : value(_value) {}
  __host__ __device__ int operator()(const int &x) const { return (x < value); }
};

template <class T>
T *AffinityMatrixCtor(T *distance1, T *distance2, T *distance3, int nrows1,
                      int nrows2, int nrows3) {

  T *affinity = new T[nrows1 * nrows1 * nrows2 * nrows2 * nrows3 * nrows3];

  int count = 0;

  for (int l = 0; l < nrows1 * nrows2 * nrows3; ++l) {
    for (int i = 0; i < nrows1; ++i) {
      for (int j = 0; j < nrows2; ++j) {
        for (int k = 0; k < nrows3; ++k) {

          // affinity[count] =
          //    fabs(distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
          //         distance2[j + ((l / nrows3) % nrows2) * nrows2] -
          //         distance3[k + (l % nrows3) * nrows3]);

          if (distance1[i + (l / (nrows2 * nrows3)) * nrows1] -
                  distance2[j + ((l / nrows3) % nrows2) * nrows2] <
              distance3[k + (l % nrows3) * nrows3] <
              distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
                  distance2[j + ((l / nrows3) % nrows2) * nrows2]) {
            affinity[count] =
                (4.5 *
                 sqrt(3 * fabs(distance1[i + (l / (nrows2 * nrows3)) * nrows1] -
                               distance2[j + ((l / nrows3) % nrows2) * nrows2] +
                               distance3[k + (l % nrows3) * nrows3]) *
                      fabs(distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
                           distance2[j + ((l / nrows3) % nrows2) * nrows2] -
                           distance3[k + (l % nrows3) * nrows3]) *
                      fabs(-distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
                           distance2[j + ((l / nrows3) % nrows2) * nrows2] +
                           distance3[k + (l % nrows3) * nrows3]) *
                      fabs(distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
                           distance2[j + ((l / nrows3) % nrows2) * nrows2] +
                           distance3[k + (l % nrows3) * nrows3]))) /
                (pow(distance1[i + (l / (nrows2 * nrows3)) * nrows1], 2) +
                 pow(distance2[j + ((l / nrows3) % nrows2) * nrows2], 2) +
                 pow(distance3[k + (l % nrows3) * nrows3], 2));
          } else
            affinity[count] = 0;
          count++;
        }
      }
    }
  }
  return affinity;
}

template <class T>
stdvec_tuple_t AffinityOrigCoo(T *distance1, T *distance2, T *distance3,
                               unsigned nrows1, unsigned nrows2,
                               unsigned nrows3) {
  std::vector<double> values;
  std::vector<int> columns;
  std::vector<int> rows;

  int count = 0;

  for (int l = 0; l < nrows1 * nrows2 * nrows3; ++l) {
    for (int i = 0; i < nrows1; ++i) {
      for (int j = 0; j < nrows2; ++j) {
        for (int k = 0; k < nrows3; ++k) {

          if (distance1[i + (l / (nrows2 * nrows3)) * nrows1] -
                  distance2[j + ((l / nrows3) % nrows2) * nrows2] <
              distance3[k + (l % nrows3) * nrows3] <
              distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
                  distance2[j + ((l / nrows3) % nrows2) * nrows2]) {
            double val =
                (4.5 *
                 sqrt(3 * fabs(distance1[i + (l / (nrows2 * nrows3)) * nrows1] -
                               distance2[j + ((l / nrows3) % nrows2) * nrows2] +
                               distance3[k + (l % nrows3) * nrows3]) *
                      fabs(distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
                           distance2[j + ((l / nrows3) % nrows2) * nrows2] -
                           distance3[k + (l % nrows3) * nrows3]) *
                      fabs(-distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
                           distance2[j + ((l / nrows3) % nrows2) * nrows2] +
                           distance3[k + (l % nrows3) * nrows3]) *
                      fabs(distance1[i + (l / (nrows2 * nrows3)) * nrows1] +
                           distance2[j + ((l / nrows3) % nrows2) * nrows2] +
                           distance3[k + (l % nrows3) * nrows3]))) /
                (pow(distance1[i + (l / (nrows2 * nrows3)) * nrows1], 2) +
                 pow(distance2[j + ((l / nrows3) % nrows2) * nrows2], 2) +
                 pow(distance3[k + (l % nrows3) * nrows3], 2));
            if (val != 0) {
              values.push_back(val);
              rows.push_back(l);
              columns.push_back(i * nrows2 * nrows3 + j * nrows3 + k);
            }
          }
          ++count;
        }
      }
    }
  }
  return std::make_tuple(values, columns, rows);
}

template <class T>
T *AffinityInitialMatches(T *distance1, T *distance2, T *distance3,
                          unsigned nrows1, unsigned nrows2, unsigned nrows3,
                          int *matches1, int *matches2, int *matches3,
                          unsigned match_len) {

  T *affinity_matches = new T[match_len * match_len];
  int count = 0;
  for (int i = 0; i < match_len; i++) {
    for (int j = 0; j < match_len; j++) {
      if (i == j) {
        affinity_matches[count] = 0;
      } else {
        int idx_mat1 = matches1[i] * nrows1 + matches1[j];
        int idx_mat2 = matches2[i] * nrows2 + matches2[j];
        int idx_mat3 = matches3[i] * nrows3 + matches3[j];

        if (distance1[idx_mat1] - distance2[idx_mat2] < distance3[idx_mat3] <
            distance1[idx_mat1] + distance2[idx_mat2]) {
          affinity_matches[count] =
              (4.5 * sqrt(3 * fabs(distance1[idx_mat1] - distance2[idx_mat2] +
                                   distance3[idx_mat3]) *
                          fabs(distance1[idx_mat1] + distance2[idx_mat2] -
                               distance3[idx_mat3]) *
                          fabs(-distance1[idx_mat1] + distance2[idx_mat2] +
                               distance3[idx_mat3]) *
                          fabs(distance1[idx_mat1] + distance2[idx_mat2] +
                               distance3[idx_mat3]))) /
              (pow(distance1[idx_mat1], 2) + pow(distance2[idx_mat2], 2) +
               pow(distance3[idx_mat3], 2));
        } else
          affinity_matches[count] = 0;
      }
      ++count;
    }
  }
  return affinity_matches;
}

template <class T>
stdvec_tuple_t AffinityInitialmatchesCoo(T *distance1, T *distance2,
                                         T *distance3, unsigned nrows1,
                                         unsigned nrows2, unsigned nrows3,
                                         int *matches1, int *matches2,
                                         int *matches3, unsigned match_len) {

  std::vector<double> values;
  std::vector<int> columns;
  std::vector<int> rows;

  for (int i = 0; i < match_len; i++) {
    for (int j = 0; j < match_len; j++) {

      int idx_mat1 = matches1[i] * nrows1 + matches1[j];
      int idx_mat2 = matches2[i] * nrows2 + matches2[j];
      int idx_mat3 = matches3[i] * nrows3 + matches3[j];

      if (i!=j && distance1[idx_mat1] - distance2[idx_mat2] < distance3[idx_mat3] <
          distance1[idx_mat1] + distance2[idx_mat2]) {
        double val =
            (4.5 * sqrt(3 * fabs(distance1[idx_mat1] - distance2[idx_mat2] +
                                 distance3[idx_mat3]) *
                        fabs(distance1[idx_mat1] + distance2[idx_mat2] -
                             distance3[idx_mat3]) *
                        fabs(-distance1[idx_mat1] + distance2[idx_mat2] +
                             distance3[idx_mat3]) *
                        fabs(distance1[idx_mat1] + distance2[idx_mat2] +
                             distance3[idx_mat3]))) /
            (pow(distance1[idx_mat1], 2) + pow(distance2[idx_mat2], 2) +
             pow(distance3[idx_mat3], 2));
        if (val != 0) {
          values.push_back(val);
          rows.push_back(i);
          columns.push_back(j);
        }
      }
    }
  }
  return std::make_tuple(values, columns, rows);
}

template <class T>
void CompressMatrix(h_vec_t<T> &values, h_vec_t<int> &columns,
                    h_vec_t<int> &row_index, T *matrix, int nrows, int ncols) {
  unsigned offset = 0;
  row_index.push_back(offset);

  for (int i = 0; i < nrows; ++i) {
    offset = 0;
    //++idx_r;
    for (int j = 0; j < ncols; ++j) {
      if (0 != matrix[i * ncols + j]) {
        values.push_back(matrix[i * ncols + j]);
        columns.push_back(j);
        ++offset;
      }
    }
    row_index.push_back(offset + row_index.back());
  }
}

template <class T>
h_vec_t<T> AffinityBlocks(T *distance2, T *distance3, unsigned key, int nrows2,
                          int nrows3) {
  h_vec_t<double> affinity;
  // T *affinity = new T[nrows2 * nrows2 * nrows3 * nrows3];
  int count = 0;
  double value;
  for (int l = 0; l < nrows2 * nrows3; ++l) {
    for (int j = 0; j < nrows2; ++j) {
      for (int k = 0; k < nrows3; ++k) {
        if (count != l * nrows2 * nrows3 + l &&
            key - distance2[j + (l / nrows3) * nrows2] <
                distance3[k + (l % nrows3) * nrows3] <
                key + distance2[j + (l / nrows3) * nrows2]) {

          value =
              (4.5 * sqrt(3 * fabs(key - distance2[j + (l / nrows3) * nrows2] +
                                   distance3[k + (l % nrows3) * nrows3]) *
                          fabs(key + distance2[j + (l / nrows3) * nrows2] -
                               distance3[k + (l % nrows3) * nrows3]) *
                          fabs(-key + distance2[j + (l / nrows3) * nrows2] +
                               distance3[k + (l % nrows3) * nrows3]) *
                          fabs(key + distance2[j + (l / nrows3) * nrows2] +
                               distance3[k + (l % nrows3) * nrows3]))) /
              (pow(key, 2) + pow(distance2[j + (l / nrows3) * nrows2], 2) +
               pow(distance3[k + (l % nrows3) * nrows3], 2));
        } else
          value = 0;
        // std::cout << value << " ";
        affinity.push_back(value);

        count++;
      }
    }
    // std::cout << std::endl;
  }
  return affinity;
}

template <class T>
stdvec_tuple_t AffinityBlocksCoo(T *distance2, T *distance3, const unsigned key,
                                 const unsigned nrows2, const unsigned nrows3) {

  std::vector<double> values;
  std::vector<int> columns;
  std::vector<int> rows;
  int count = 0;
  for (int l = 0; l < nrows2 * nrows3; ++l) {
    for (int j = 0; j < nrows2; ++j) {
      for (int k = 0; k < nrows3; ++k) {
        if (count != l * nrows2 * nrows3 + l &&
            key - distance2[j + (l / nrows3) * nrows2] <
                distance3[k + (l % nrows3) * nrows3] <
                key + distance2[j + (l / nrows3) * nrows2]) {

          double val =
              (4.5 * sqrt(3 * fabs(key - distance2[j + (l / nrows3) * nrows2] +
                                   distance3[k + (l % nrows3) * nrows3]) *
                          fabs(key + distance2[j + (l / nrows3) * nrows2] -
                               distance3[k + (l % nrows3) * nrows3]) *
                          fabs(-key + distance2[j + (l / nrows3) * nrows2] +
                               distance3[k + (l % nrows3) * nrows3]) *
                          fabs(key + distance2[j + (l / nrows3) * nrows2] +
                               distance3[k + (l % nrows3) * nrows3]))) /
              (pow(key, 2) + pow(distance2[j + (l / nrows3) * nrows2], 2) +
               pow(distance3[k + (l % nrows3) * nrows3], 2));
          if (val != 0) {
            values.push_back(val);
            rows.push_back(l);
            columns.push_back(j * nrows3 + k );
          }
        }
        count++;
      }
    }
  }

  return std::make_tuple(values, columns, rows);
}

struct createCOO {
  int key;
  int row_len;

  createCOO(int _key, int _row_len) : key(_key), row_len(_row_len) {}
  template <typename T> __host__ __device__ void operator()(T x) {

    double difference = key - get<0>(get<0>(x));
    if (key != 0 && get<0>(get<0>(x)) != 0 && fabs(difference) < 3 * SIGMA) {
      double val = 4.5 - pow(difference, 2) / (2 * pow(SIGMA, 2));
      if (0 != val) {
        get<0>(get<1>(x)) = val;
        get<1>(get<1>(x)) = get<1>(get<0>(x)) / row_len;
        get<2>(get<1>(x)) = get<1>(get<0>(x)) % row_len;
        return;
      }
    }
    get<0>(get<1>(x)) = -1;
    get<1>(get<1>(x)) = -1;
    get<2>(get<1>(x)) = -1;
    return;
  }
};

struct Affinity {
  int key;
  Affinity(int _key) : key(_key) {}
  __host__ __device__ double operator()(double x) {
    if (x != 0 && key != 0 && fabs(key - x) < 3 * SIGMA) {
      return (4.5 - pow(fabs(key - x), 2) / (2 * pow(SIGMA, 2)));
    } else
      return 0;
  }
};

// square<T> computes the square of a number f(x) -> x*x
struct square {
  __host__ __device__ double operator()(const double &x) const { return x * x; }
};

// division<value> computes the division of f(x) -> x/value
struct division {
  const double value;

  division(double _value) : value(_value) {}

  __host__ __device__ double operator()(const double &x) const {
    if (0 != value)
      return x / value;
    else
      return 1;
  }
};
/**
 * \brief Read a matrix `mtrx` from a file.
 *
 * \param mtrx
 *  A vector, containing the matrix values- flattened in row-major order.
 * \param file_name
 *  Name of the file containing the matrix data.
 * \param num_rows
 *  Number of rows in of matrix.
 * \param num_cols
 *  Number of columns in of matrix. It is set to num_rows if not defined
 * (i.e.,
 *  assumed as a symmetric matrix.)
 */
template <class T>
void ReadMatrix(h_vec_t<T> &matrix, std::string file_name, int num_rows,
                int num_cols = 0) {

  // if matrix is symmetric and num_cols was not passed to the function.
  if (0 == num_cols)
    num_cols = num_rows;

  // make sure input arguments are valid.
  if (!num_rows) {
    std::cerr << "Number rows must be greater than 0!" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::ifstream file_content(file_name.c_str(), std::ios::in);
  if (!file_content) {
    std::cerr << "Could not open file: " << file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  matrix.resize(num_rows * num_cols);

  for (int i = 0; i < num_rows * num_cols; i++)
    file_content >> matrix[i];
}

/**
 * \brief Print a symmetic matrix `mtrx` on the screen.
 *
 * \param mtrx
 *  A pointer, containing the matrix values in row-major order.
 * \param num_rows
 *  Number of rows in the matrix.
 * \param num_cols
 *  Number of columns in of matrix. It is set to num_rows if not defined
 * (i.e.,
 *  assumed as a symmetric matrix.)
 */
template <class T>
void PrintMatrix(h_vec_t<T> mtrx, int num_rows, int num_cols = 0) {

  if (0 == num_cols)
    num_cols = num_rows;

  // make sure input arguments are valid.
  if (!num_rows) {
    std::cerr << "Number rows must be greater than 0!" << std::endl;
    exit(1);
  }

  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      std::cout << mtrx[i * num_cols + j] << "\t";
    }
    std::cout << std::endl;
  }
}

template <class T>
void ReadMatchedFeatures(host_vector<T> &features_1, host_vector<T> &features_2,
                         host_vector<T> &features_3, std::string file_name,
                         int num_matched) {

  std::ifstream file_content(file_name.c_str(), std::ios::in);
  if (!file_content) {
    std::cerr << "Could not open file: " << file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < num_matched; ++i) {
    file_content >> features_1[i];
    file_content >> features_2[i];
    file_content >> features_3[i];
  }
}

#ifdef ACCELERATE
template <class T> d_vec_t<T> FindUniques(d_vec_t<T> vec) {
  d_vec_t<T> vec_unique = vec;
#else
template <class T> h_vec_t<T> FindUniques(h_vec_t<T> vec) {
  h_vec_t<T> vec_unique = vec;
#endif
  sort(vec_unique.begin(), vec_unique.end());
  vec_unique.resize(unique(vec_unique.begin(), vec_unique.end()) -
                    vec_unique.begin());
  return vec_unique;
}

template <typename Itr> class StridedRange {
public:
  struct StrideFunctor : public unary_function<int, int> {
    int stride;

    StrideFunctor(int stride) : stride(stride) {}

    __host__ __device__ int operator()(const int &i) const {
      return stride * i;
    }
  };

  template <class T>
  using transform_iter_t =
      transform_iterator<StrideFunctor, counting_iterator<T>>;

  template <class T>
  using permutation_iter_t = permutation_iterator<Itr, transform_iter_t<T>>;

  // construct StridedRange for the range [first,last)
  StridedRange(Itr first, Itr last, int stride)
      : first(first), last(last), stride(stride) {}

  permutation_iter_t<int> begin(void) const {
    return permutation_iter_t<int>(
        first, transform_iter_t<int>(counting_iterator<int>(0),
                                     StrideFunctor(stride)));
  }

  permutation_iter_t<int> end(void) const {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

protected:
  Itr first;
  Itr last;
  int stride;
};


template <class T> T VectorNorm(T *v, const unsigned length) {
  T norm = 0;

  for (int i = 0; i < length; ++i) {
    norm += v[i] * v[i];
  }
  norm = sqrt(norm);

  return norm;
}

#endif // INCLUDE_UTILS_H_
