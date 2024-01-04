#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

namespace vt
{

	struct Webcam
	{
		// Webcam Properties
		const int framerate;
		const int latency_ms;
		const int width, height;


		// Webcam Functions
		static std::optional<Webcam> TryCreate(
			const int id,
			const cv::Size& target_size,     // Not guaranteed
			const int target_framerate = 30  // Not guaranteed
		);

		~Webcam();


		void drop_frame();

		bool next_frame(cv::UMat& dst);


		bool is_open() const;

		cv::VideoCapture& raw();

		const cv::VideoCapture& raw() const;

	private:
		Webcam(cv::VideoCapture&& stream);

		cv::VideoCapture m_Stream;
	};


}


