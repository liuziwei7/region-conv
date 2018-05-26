#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/region_im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void region_im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const Dtype* index_1, const Dtype* index_2,
    const int mask_cnt, Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int m_index = index % mask_cnt;
    const int h_col = index_1[m_index];
    const int w_col = index_2[m_index];
    const int c_im = index / mask_cnt;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col - pad_h;
    const int w_offset = w_col - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += c_col * mask_cnt + m_index;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += mask_cnt;
      }
    }
  }
}

template <typename Dtype>
void region_im2col_gpu(const Dtype* data_im,
    const Dtype* index_1, const Dtype* index_2, const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.

  int num_kernels = channels * mask_cnt;
  // NOLINT_NEXT_LINE(whitespace/operators)
  region_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, dilation_h, dilation_w, index_1,
      index_2, mask_cnt, data_col);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void compression_region_im2col_gpu_kernel(const int n, const Dtype* data_im, const Dtype* data_mask,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const Dtype* index_1, const Dtype* index_2,
    const int mask_cnt, Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int m_index = index % mask_cnt;
    const int h_col = index_1[m_index];
    const int w_col = index_2[m_index];
    const int c_im = index / mask_cnt;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col - pad_h;
    const int w_offset = w_col - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += c_col * mask_cnt + m_index;
    const Dtype* data_im_ptr = data_im + c_im * mask_cnt;
    const Dtype* data_mask_ptr = data_mask + h_offset * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width && static_cast<int>(data_mask_ptr[i * dilation_h * width + j * dilation_w]) >= 0) ?
        data_im_ptr[static_cast<int>(data_mask_ptr[i * dilation_h * width + j * dilation_w])] : 0;
        data_col_ptr += mask_cnt;
      }
    }
  }
}

template <typename Dtype>
void compression_region_im2col_gpu(const Dtype* data_im, const Dtype* data_mask,
    const Dtype* index_1, const Dtype* index_2, const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.

  int num_kernels = channels * mask_cnt;
  // NOLINT_NEXT_LINE(whitespace/operators)
  compression_region_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, data_mask, height, width, kernel_h, kernel_w, pad_h,
      pad_w, dilation_h, dilation_w, index_1,
      index_2, mask_cnt, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void region_im2col_gpu<float>(const float* data_im, 
    const float* index_1, const float* index_2, const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, float* data_col);
template void region_im2col_gpu<double>(const double* data_im,
    const double* index_1, const double* index_2, const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, double* data_col);

template void compression_region_im2col_gpu<float>(const float* data_im, const float* data_mask,
    const float* index_1, const float* index_2, const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, float* data_col);
template void compression_region_im2col_gpu<double>(const double* data_im, const double* data_mask,
    const double* index_1, const double* index_2, const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, double* data_col);


template <typename Dtype>
__global__ void region_col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    const Dtype* index_1, const Dtype* index_2, const Dtype* data_mask,
    const int mask_cnt, Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int m_index = index % mask_cnt;
    const int h_im = index_1[m_index] + pad_h;
    const int w_im = index_2[m_index] + pad_w;
    const int c_im = index / (mask_cnt);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) + 1;
    const int w_col_end = min(w_im + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) + 1;
    const int h_col_end = min(h_im + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = h_im - h_col;
        int w_k = w_im - w_col;
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          const int temp = static_cast<int>(data_mask[h_col * width + w_col]);
          if (temp != -1)
          {
            h_k /= dilation_h;
            w_k /= dilation_w;
            int data_col_index = ((c_im * kernel_h + h_k) * kernel_w + w_k) * mask_cnt + temp;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_im[(c_im * height + (h_im - pad_h)) * width + (w_im - pad_w)] = val;
  }
}

template <typename Dtype>
void region_col2im_gpu(const Dtype* data_col, 
    const Dtype* index_1, const Dtype* index_2, const Dtype* data_mask,
    const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) + 1;
  int num_kernels = channels * mask_cnt;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)

  region_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, dilation_h, dilation_w,
      height_col, width_col, index_1, index_2, data_mask, mask_cnt, data_im);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void compression_region_col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    const Dtype* index_1, const Dtype* index_2, const Dtype* data_mask,
    const int mask_cnt, Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int m_index = index % mask_cnt;
    const int h_im = index_1[m_index] + pad_h;
    const int w_im = index_2[m_index] + pad_w;
    const int c_im = index / (mask_cnt);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) + 1;
    const int w_col_end = min(w_im + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) + 1;
    const int h_col_end = min(h_im + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = h_im - h_col;
        int w_k = w_im - w_col;
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          const int temp = static_cast<int>(data_mask[h_col * width + w_col]);
          if (temp != -1)
          {
            h_k /= dilation_h;
            w_k /= dilation_w;
            int data_col_index = ((c_im * kernel_h + h_k) * kernel_w + w_k) * mask_cnt + temp;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_im[c_im * mask_cnt + m_index] = val;
  }
}

template <typename Dtype>
void compression_region_col2im_gpu(const Dtype* data_col, 
    const Dtype* index_1, const Dtype* index_2, const Dtype* data_mask,
    const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) + 1;
  int num_kernels = channels * mask_cnt;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)

  compression_region_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, dilation_h, dilation_w,
      height_col, width_col, index_1, index_2, data_mask, mask_cnt, data_im);

  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void region_col2im_gpu<float>(const float* data_col,
    const float* index_1, const float* index_2, const float* data_mask,
    const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void region_col2im_gpu<double>(const double* data_col,
    const double* index_1, const double* index_2, const double* data_mask,
    const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    double* data_im);
template void compression_region_col2im_gpu<float>(const float* data_col,
    const float* index_1, const float* index_2, const float* data_mask,
    const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void compression_region_col2im_gpu<double>(const double* data_col,
    const double* index_1, const double* index_2, const double* data_mask,
    const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    double* data_im);
}  // namespace caffe
