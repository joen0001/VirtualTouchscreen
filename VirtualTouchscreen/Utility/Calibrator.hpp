#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

#include "Abstractions/Webcam.hpp"

namespace vt
{
	// Base class for a generic camera-projector calibration 
	class Calibrator
	{
	protected:
		static void calibrate_exposure(
			Webcam& webcam,
			const double brightness_target,
			const std::string& window_name,
			const bool auto_destroy_window = false
		);

		static void capture_colour(
			Webcam& webcam,
			cv::UMat& dst,
			const cv::Scalar& colour,
			const int settle_time_ms,
			const int capture_samples,
			const std::string& window_name,
			const bool auto_destroy_window = false
		);

		static void capture_image(
			Webcam& webcam,
			cv::UMat& dst,
			const cv::Mat& image,
			const int settle_time_ms,
			const int capture_samples,
			const std::string& window_name,
			const bool auto_destroy_window = false
		);

		static void show_feedback(
			Webcam& webcam,
			const cv::String& top_text,
			const cv::String& bot_text,
			const std::string& window_name,
			const bool auto_destroy_window = false
		);

		static cv::Rect make_fullscreen_window(
			const std::string& window_name
		);
	};

}