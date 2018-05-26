#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/region_im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Blob<Dtype>* RegionConvolutionLayer<Dtype>::col_buffer_ = new Blob<Dtype>();

template <typename Dtype>
Blob<Dtype>* RegionConvolutionLayer<Dtype>::top_buffer_ = new Blob<Dtype>();

template <typename Dtype>
Blob<Dtype>* RegionConvolutionLayer<Dtype>::mask_buffer_[2] = {new Blob<Dtype>(), new Blob<Dtype>()};

template <typename Dtype>
Blob<Dtype>* RegionConvolutionLayer<Dtype>::index_buffer_[2] = {new Blob<Dtype>(), new Blob<Dtype>()};


template <typename Dtype>
void RegionConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - (this->dilation_h_ * (this->kernel_h_ - 1) + 1)) + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - (this->dilation_w_ * (this->kernel_w_ - 1) + 1)) + 1;
}


template <typename Dtype>
void RegionConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK(!conv_param.has_stride() && !conv_param.has_stride_h() && !conv_param.has_stride_w()) << "Do not support stride now";

  input_compression_ = this->layer_param_.region_convolution_param().input_compression();
  output_compression_ = this->layer_param_.region_convolution_param().output_compression();
  mask_idx_ = this->layer_param_.region_convolution_param().mask_idx();


  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_dilation_h()) {
    dilation_h_ = dilation_w_ = conv_param.dilation();
  } else {
    dilation_h_ = conv_param.dilation_h();
    dilation_w_ = conv_param.dilation_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  CHECK_EQ(this->layer_param_.convolution_param().group(), 1)<< "Do not support group now";
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void RegionConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  if (bottom.size() == 3)
  {
    RegionConvolutionLayer<Dtype>::mask_buffer_[mask_idx_]->ReshapeLike(*bottom[1]);
    caffe_copy(bottom[1]->count(), bottom[1]->cpu_data(),  mask_buffer_[mask_idx_]->mutable_cpu_data());
    RegionConvolutionLayer<Dtype>::index_buffer_[mask_idx_]->ReshapeLike(*bottom[2]);
    caffe_copy(bottom[2]->count(), bottom[2]->cpu_data(),  index_buffer_[mask_idx_]->mutable_cpu_data());
  }

  num_ = bottom[0]->num();
  height_ = mask_buffer_[mask_idx_]->height();
  width_ = mask_buffer_[mask_idx_]->width();

  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";

  spatial_dim_  = height_ * width_;
  if (!input_compression_)
  {
    CHECK_EQ(bottom[0]->height(), height_);
    CHECK_EQ(bottom[0]->width(), width_);
  }
  else
  {
    CHECK_EQ(bottom[0]->height(), 1);
    CHECK_EQ(bottom[0]->width(), spatial_dim_);
  }

  CHECK_EQ(mask_buffer_[mask_idx_]->num(), num_);
  CHECK_EQ(mask_buffer_[mask_idx_]->channels(), 1);

  CHECK_EQ(index_buffer_[mask_idx_]->num(), num_);
  CHECK_EQ(index_buffer_[mask_idx_]->channels(), 1);
  CHECK_EQ(index_buffer_[mask_idx_]->height(), 2);
  CHECK_EQ(index_buffer_[mask_idx_]->width(), spatial_dim_ + 1);

  // Shape the tops.
  compute_output_shape();

  // only support in == out;
  CHECK_EQ(height_, height_out_);
  CHECK_EQ(width_, width_out_);

  if (!output_compression_)
    top[0]->Reshape(num_, num_output_, height_out_, width_out_);
  else
    top[0]->Reshape(num_, num_output_, 1, spatial_dim_);
    
  conv_in_height_ = height_;
  conv_in_width_ = width_;
  conv_out_spatial_dim_ = height_out_ * width_out_;
  
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.


  if (col_buffer_->count() < kernel_dim_ * height_out_ * width_out_)
    col_buffer_->Reshape(1, kernel_dim_, height_out_, width_out_);

  if (top_buffer_->count() < conv_out_channels_ * height_out_ * width_out_)
    top_buffer_->Reshape(1, conv_out_channels_, height_out_, width_out_);
  
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

}



template <typename Dtype>
void RegionConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void RegionConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RegionConvolutionLayer);
#endif

INSTANTIATE_CLASS(RegionConvolutionLayer);
REGISTER_LAYER_CLASS(RegionConvolution);

}  // namespace caffe
