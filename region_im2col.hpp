#ifndef _CAFFE_UTIL_REGION_IM2COL_HPP_
#define _CAFFE_UTIL_REGION_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
void region_im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col);

template <typename Dtype>
void region_col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);

template <typename Dtype>
void region_im2col_gpu(const Dtype* data_im,
    const Dtype* index_1, const Dtype* index_2, const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col);

template <typename Dtype>
void compression_region_im2col_gpu(const Dtype* data_im, const Dtype* data_mask,
    const Dtype* index_1, const Dtype* index_2, const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col);

template <typename Dtype>
void region_col2im_gpu(const Dtype* data_col, 
    const Dtype* index_1, const Dtype* index_2, const Dtype* data_mask,
    const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);

template <typename Dtype>
void compression_region_col2im_gpu(const Dtype* data_col, 
    const Dtype* index_1, const Dtype* index_2, const Dtype* data_mask,
    const int mask_cnt, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_REGION_IM2COL_HPP_
