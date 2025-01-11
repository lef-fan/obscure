#include <torch/script.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <obs-module.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <tuple>
#include <algorithm>
#include <ATen/ATen.h>


OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obscure", "en-US")

const char *obs_module_description(void) {
    return "Obscure: Say goodbye to boring camera backgrounds!";
}

std::string getHomeDirectory() {
    const char* homeDir = getenv("HOME");
    if (!homeDir) {
        throw std::runtime_error("HOME environment variable is not set");
    }
    return std::string(homeDir);
}

std::string getModelPath() {
    std::string homeDir = getHomeDirectory();
    return homeDir + "/.config/obs-studio/plugins/obscure/bin/64bit/yolov8l-seg_288-480.torchscript";
}

static obs_source_frame *last_frame = NULL;

struct ObscureData {
    obs_source_t *source;
    cv::Mat frame;
    torch::jit::script::Module model;
    bool model_loaded;
    torch::Device device;
    const float confidence_threshold = 0.25f;
    const int person_class_id = 0;
    const float nms_threshold = 0.45f;
    const bool enable_temporal_smoothing = true;
    
    ObscureData() : model_loaded(false), device(torch::kCUDA) {
        try {            
            blog(LOG_INFO, "Loading model on GPU");
            std::string modelPath = getModelPath();
            blog(LOG_INFO, "Loading model from %s", modelPath.c_str());
            model = torch::jit::load(modelPath, device);
            model.eval();
            model_loaded = true;
            blog(LOG_INFO, "Model loaded successfully on GPU");
        } catch (const std::exception& e) {
            blog(LOG_ERROR, "Error loading model: %s", e.what());
        }
    }
};

static const char *obscure_get_name(void *type_data) {
    return "Obscure";
}

static void *obscure_create(obs_data_t *settings, obs_source_t *source) {
    auto *filter = new ObscureData();
    filter->source = source;
    return filter;
}

static void obscure_destroy(void *data) {
    if (data) {
        auto *filter = static_cast<ObscureData *>(data);
        try {
            if (filter->model_loaded) {
                blog(LOG_INFO, "Starting model cleanup...");
                filter->model.to(torch::kCPU);
                filter->model = torch::jit::Module();
                filter->model_loaded = false;
                c10::cuda::CUDACachingAllocator::emptyCache();
                blog(LOG_INFO, "Model unloaded successfully");
            }
        } catch (const std::exception& e) {
            blog(LOG_ERROR, "Error during model cleanup: %s", e.what());
        }
        delete filter;
        blog(LOG_INFO, "Filter destroyed");
    }
}

float calculate_iou(const torch::Tensor& box1, const torch::Tensor& box2) {
    float x1 = std::max(box1[0].item<float>(), box2[0].item<float>());
    float y1 = std::max(box1[1].item<float>(), box2[1].item<float>());
    float x2 = std::min(box1[2].item<float>(), box2[2].item<float>());
    float y2 = std::min(box1[3].item<float>(), box2[3].item<float>());
    
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float box1_area = (box1[2].item<float>() - box1[0].item<float>()) * 
                     (box1[3].item<float>() - box1[1].item<float>());
    float box2_area = (box2[2].item<float>() - box2[0].item<float>()) * 
                     (box2[3].item<float>() - box2[1].item<float>());
    
    return intersection / (box1_area + box2_area - intersection);
}

std::vector<size_t> nms(const torch::Tensor& boxes, const torch::Tensor& scores, float iou_threshold) {
    auto scores_cpu = scores.cpu();
    auto boxes_cpu = boxes.cpu();
    
    std::vector<size_t> indices(scores_cpu.size(0));
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
              [&scores_cpu](size_t i1, size_t i2) {
                  return scores_cpu[i1].item<float>() > scores_cpu[i2].item<float>();
              });
    
    std::vector<size_t> keep;
    while (!indices.empty()) {
        size_t current = indices[0];
        keep.push_back(current);
        
        std::vector<size_t> temp;
        for (size_t i = 1; i < indices.size(); i++) {
            float iou = calculate_iou(boxes_cpu[current], boxes_cpu[indices[i]]);
            if (iou <= iou_threshold) {
                temp.push_back(indices[i]);
            }
        }
        indices = std::move(temp);
    }
    
    return keep;
}


static obs_source_frame *obscure_video_render(void *data, obs_source_frame *frame) {
    auto *filter = static_cast<ObscureData *>(data);
    if (!filter || !frame || !frame->data || !filter->model_loaded) {
        blog(LOG_ERROR, "Invalid filter, frame data, or model not loaded");
        return frame;
    }

    cv::Mat input_frame;

    try {
        switch (frame->format) {
            case VIDEO_FORMAT_UYVY: {
                cv::Mat uyvy(frame->height, frame->width, CV_8UC2, 
                            frame->data[0], frame->linesize[0]);
                cv::cvtColor(uyvy, input_frame, cv::COLOR_YUV2RGB_UYVY);
            } break;
            case VIDEO_FORMAT_I422: {
                cv::Mat y_plane(frame->height, frame->width, CV_8UC1, frame->data[0], frame->linesize[0]);
                cv::Mat u_plane(frame->height, frame->width / 2, CV_8UC1, frame->data[1], frame->linesize[1]);
                cv::Mat v_plane(frame->height, frame->width / 2, CV_8UC1, frame->data[2], frame->linesize[2]);

                cv::Mat u_resized, v_resized;
                cv::resize(u_plane, u_resized, cv::Size(frame->width, frame->height), 0, 0, cv::INTER_LINEAR);
                cv::resize(v_plane, v_resized, cv::Size(frame->width, frame->height), 0, 0, cv::INTER_LINEAR);

                std::vector<cv::Mat> yuv_planes = {y_plane, u_resized, v_resized};
                cv::Mat yuv;
                cv::merge(yuv_planes, yuv);

                cv::cvtColor(yuv, input_frame, cv::COLOR_YUV2RGB);
            } break;
            case VIDEO_FORMAT_I420: {
                cv::Mat y_plane(frame->height, frame->width, CV_8UC1, frame->data[0], frame->linesize[0]);
                cv::Mat u_plane(frame->height / 2, frame->width / 2, CV_8UC1, frame->data[1], frame->linesize[1]);
                cv::Mat v_plane(frame->height / 2, frame->width / 2, CV_8UC1, frame->data[2], frame->linesize[2]);

                cv::Mat u_resized, v_resized;
                cv::resize(u_plane, u_resized, y_plane.size(), 0, 0, cv::INTER_LINEAR);
                cv::resize(v_plane, v_resized, y_plane.size(), 0, 0, cv::INTER_LINEAR);

                std::vector<cv::Mat> yuv_planes = {y_plane, u_resized, v_resized};
                cv::Mat yuv;
                cv::merge(yuv_planes, yuv);

                cv::cvtColor(yuv, input_frame, cv::COLOR_YUV2RGB);
            } break;
            default:
                blog(LOG_ERROR, "Unsupported input format: %d", frame->format);
                return frame;
        }

        // cv::Mat blue_frame = cv::Mat::zeros(input_frame.size(), input_frame.type());
        // blue_frame.setTo(cv::Scalar(0, 0, 255));
        cv::Mat transparent_frame(input_frame.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));

        cv::Mat resized_frame;
        cv::resize(input_frame, resized_frame, cv::Size(480, 288));
        
        torch::Tensor input_tensor = torch::from_blob(
            resized_frame.data,
            {1, resized_frame.rows, resized_frame.cols, resized_frame.channels()},
            torch::kByte
        ).to(filter->device);        
      
        input_tensor = input_tensor.permute({0, 3, 1, 2})
                                 .to(torch::kFloat32)
                                 .div(255.0);

        auto output = filter->model.forward({input_tensor}).toTuple();
        auto detection = output->elements()[0].toTensor();
        auto proto = output->elements()[1].toTensor();

        auto pred = detection.squeeze(0).transpose(0, 1);
        auto box_pred = pred.slice(1, 0, 4);
        auto cls_pred = pred.slice(1, 4, 84);
        auto mask_pred = pred.slice(1, 84, 116);
        
        auto [scores, class_ids] = cls_pred.max(1);
        
        auto person_mask_ = class_ids == filter->person_class_id;
        auto conf_mask = scores > filter->confidence_threshold;
        auto valid_mask = person_mask_ & conf_mask;

        bool skip_this_frame = false;
        
        if (!valid_mask.any().item<bool>()) {
            skip_this_frame = true;
        }

        if (skip_this_frame == false) {
            auto filtered_boxes = box_pred.index({valid_mask});
            auto filtered_scores = scores.index({valid_mask});
            auto filtered_masks = mask_pred.index({valid_mask});
            
            auto keep_indices = nms(filtered_boxes, filtered_scores, filter->nms_threshold);
            
            if (keep_indices.empty()) {
                skip_this_frame = true;
            }

            if (skip_this_frame == false) { 
                size_t best_idx = keep_indices[0];
                auto best_box = filtered_boxes[best_idx];
                auto best_mask = filtered_masks[best_idx];
                
                auto proto_out = proto.squeeze(0);
                
                auto mask_flat = proto_out.reshape({proto_out.size(0), -1});
                auto masks = torch::matmul(best_mask, mask_flat)
                                .reshape({-1, proto_out.size(1), proto_out.size(2)});
                
                auto mask_cpu = masks[0].cpu().contiguous();
                cv::Mat current_mask(masks.size(1), masks.size(2), CV_32F,
                                    mask_cpu.data_ptr<float>());
                
                static cv::Mat prev_mask;
                if (!prev_mask.empty() && filter->enable_temporal_smoothing) {
                    cv::addWeighted(current_mask, 0.7, prev_mask, 0.3, 0, current_mask);
                }
                prev_mask = current_mask.clone();
                
                cv::GaussianBlur(current_mask, current_mask, cv::Size(3, 3), 0);
                
                cv::Mat person_mask;
                cv::resize(current_mask, person_mask,
                        cv::Size(input_frame.cols, input_frame.rows),
                        0, 0, cv::INTER_LINEAR);
                
                cv::Mat person_mask_binary;
                cv::threshold(person_mask, person_mask_binary, 0.5, 1.0, cv::THRESH_BINARY);
                person_mask_binary.convertTo(person_mask_binary, CV_8U, 255);
                
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
                cv::morphologyEx(person_mask_binary, person_mask_binary, cv::MORPH_CLOSE, kernel);
                // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
                // cv::erode(person_mask_binary, person_mask_binary, kernel, cv::Point(-1,-1), 5);
                
                cv::Mat person_segment;
                cv::bitwise_and(input_frame, input_frame, person_segment, person_mask_binary);
                //person_segment.copyTo(blue_frame, person_mask_binary);

                cv::Mat person_segment_rgba;
                //cv::cvtColor(person_segment, person_segment_rgba, cv::COLOR_RGB2BGRA);
                cv::cvtColor(person_segment, person_segment_rgba, cv::COLOR_RGB2RGBA);
               
                person_segment_rgba.copyTo(transparent_frame(cv::Rect(0, 0, person_segment_rgba.cols, person_segment_rgba.rows)), person_mask_binary);
                cv::Mat alpha_channel = transparent_frame(cv::Rect(0, 0, person_segment_rgba.cols, person_segment_rgba.rows));
                alpha_channel.setTo(cv::Scalar(255), person_mask_binary);

                cv::Mat rgb_channels = transparent_frame(cv::Rect(0, 0, person_segment_rgba.cols, person_segment_rgba.rows));
                person_segment_rgba.copyTo(rgb_channels, person_mask_binary);

            }
        }

        if (last_frame) {
            obs_source_frame_free(last_frame);
            last_frame = NULL;
        }

        obs_source_frame *out_frame = obs_source_frame_create(VIDEO_FORMAT_RGBA, 
                                                            transparent_frame.cols, 
                                                            transparent_frame.rows);
        if (!out_frame) {
            return frame;
        }

        size_t frame_size = transparent_frame.rows * transparent_frame.cols * 4;
        memcpy(out_frame->data[0], transparent_frame.data, frame_size);
        if (frame) {
            obs_source_frame_free(frame);
        }
     
        last_frame = out_frame;

        //blog(LOG_DEBUG, "Frame processed successfully");
        
        return out_frame;
        
    } catch (const c10::Error& e) {
        blog(LOG_ERROR, "PyTorch error: %s", e.what());
    } catch (const cv::Exception& e) {
        blog(LOG_ERROR, "OpenCV error: %s", e.what());
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "Standard exception: %s", e.what());
    } catch (...) {
        blog(LOG_ERROR, "Unknown error during frame processing");
    }

    //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return frame;
}

struct obs_source_info obscure = {
    .id = "obscure",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = obscure_get_name,
    .create = obscure_create,
    .destroy = obscure_destroy,
    .filter_video = obscure_video_render,
};

bool obs_module_load(void) {
    obs_register_source(&obscure);
    blog(LOG_INFO, "Obscure loaded successfully!");
    return true;
}

void obs_module_unload(void) {
    blog(LOG_INFO, "Obscure unloaded.");
}