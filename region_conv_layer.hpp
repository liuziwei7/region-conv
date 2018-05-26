
template <typename Dtype>
class RegionConvolutionLayer : public Layer<Dtype> {
 public:
  explicit RegionConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline const char* type() const { return "RegionConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void compute_output_shape();

  int kernel_h_, kernel_w_;
  int dilation_h_, dilation_w_;
  int num_;
  int channels_;
  int pad_h_, pad_w_;
  int height_, width_;
  int spatial_dim_;
  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  bool is_1x1_;

  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int conv_in_height_;
  int conv_in_width_;
  int kernel_dim_;

  bool input_compression_;
  bool output_compression_;

  int mask_idx_;

  static Blob<Dtype> *col_buffer_;
  static Blob<Dtype> *top_buffer_;
  static Blob<Dtype> *mask_buffer_[2];
  static Blob<Dtype> *index_buffer_[2];
  Blob<Dtype> bias_multiplier_;
};
