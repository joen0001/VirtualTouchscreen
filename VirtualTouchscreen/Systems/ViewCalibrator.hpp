#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

#include "Abstractions/Webcam.hpp"
#include "Utility/Calibrator.hpp"

namespace vt
{

	// This is just a way of transferring a calibration 
	// between two OpenCL contexts on different threads. 
	struct ViewProperties
	{
		// Geometric Calibration
		cv::Mat view_homography;
		cv::UMat correction_map;
		cv::Size output_resolution;
		std::vector<cv::Point2f> screen_contour;

		// Photometric calibration
		std::array<cv::Vec3f, 8 * 8 * 8> colour_map;
		cv::Mat reflectance_map;
	};


	class ViewCalibrator : protected Calibrator
	{
	public:
		
		ViewCalibrator(const cv::Size& target_resolution);

		ViewCalibrator(const ViewProperties& context);

		const cv::Size& output_resolution() const; 

		float ambient_intensity() const;
		
		// Calibrate to the current view
		void calibrate(
			Webcam& webcam,
			const float min_coverage,
			const int settle_time_ms = 500
		);

		// Correct frame based on the calibration. 
		void correct(
			const cv::UMat& src,
			cv::UMat& dst
		) const;

		// Predict the output of the projector.
		// NOTE: dst is in CV_32FC3 with range [0,255].
		void predict(
			const cv::Mat& src,
			cv::Mat& dst
		) const;

		ViewProperties context() const;

	private:

		std::optional<std::vector<cv::Point2f>> find_geometric_model(
			const std::vector<cv::Scalar>& colours,
			const std::vector<cv::UMat>& samples,
			const cv::UMat& chessboard_sample,
			const cv::Size& chessboard_size
		);

		void find_photometric_model(
			Webcam& webcam,
			const int settle_time_ms,
			const std::string& window_name,
			const cv::UMat& white_sample
		);

		std::optional<std::vector<cv::Point2f>> detect_screen(
			const std::vector<cv::Scalar>& colours,
			const std::vector<cv::UMat>& samples
		) const;

		std::optional<std::vector<cv::Point2f>> detect_chessboard(
			const std::vector<cv::Point2f>& screen_bounds,
			const cv::UMat& chessboard_sample,
			const cv::Size& chessboard_size
		) const;
		
	private:
		const cv::Size m_OutputResolution;
		
		// Geometric calibration
		cv::UMat m_CorrectionMap;
		cv::Mat m_ViewHomography;
		std::vector<cv::Point2f> m_ScreenContour;

		// Photometric calibration
		// Map Size: 8x8x8 = 512 samples
		// Colour Step: 1/7 = 0.142
		// Colour Mapping: x = B, y = G, z = R
		std::array<cv::Vec3f, 8 * 8 * 8> m_ColourMap;
		cv::Mat m_ReflectanceMap;
	};


}

