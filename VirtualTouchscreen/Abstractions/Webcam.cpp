#include "Webcam.hpp"

#include <cstdlib>

namespace vt
{

	//---------------------------------------------------------------------------------------------------------------------

	std::optional<Webcam> Webcam::TryCreate(const int id, const cv::Size& target_size, const int target_framerate)
	{
		CV_Assert(target_size.width > 0 && target_size.height > 0);
		CV_Assert(target_framerate > 0);

		// Fixes MSMF backend taking a long time to initialize. 
		_putenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0");

		// Open the webcam stream
		cv::VideoCapture webcam_stream(id, cv::CAP_DSHOW);
		if (!webcam_stream.isOpened())
			return {};

		// Update stream size and FPS properties
		webcam_stream.set(cv::CAP_PROP_FPS, target_framerate);
		webcam_stream.set(cv::CAP_PROP_FRAME_WIDTH, target_size.width);
		webcam_stream.set(cv::CAP_PROP_FRAME_HEIGHT, target_size.height);

		// Open settings menu (DSHOW only)
		webcam_stream.set(cv::CAP_PROP_SETTINGS, -1);

		return Webcam(std::move(webcam_stream));
	}

//---------------------------------------------------------------------------------------------------------------------
	
	Webcam::Webcam(cv::VideoCapture&& stream)
		: framerate(static_cast<int>(stream.get(cv::CAP_PROP_FPS))),
		  latency_ms(std::round(1000.0f / static_cast<float>(framerate))),
		  width(static_cast<int>(stream.get(cv::CAP_PROP_FRAME_WIDTH))),
		  height(static_cast<int>(stream.get(cv::CAP_PROP_FRAME_HEIGHT)))
	{
		CV_Assert(stream.isOpened());
		m_Stream = std::move(stream);
	}

//---------------------------------------------------------------------------------------------------------------------

	Webcam::~Webcam()
	{
		m_Stream.release();
	}

//---------------------------------------------------------------------------------------------------------------------

	bool Webcam::is_open() const
	{
		return m_Stream.isOpened();
	}

//---------------------------------------------------------------------------------------------------------------------

	void Webcam::drop_frame()
	{
		// Burn a frame. 
		m_Stream.grab();
	}

//---------------------------------------------------------------------------------------------------------------------

	bool Webcam::next_frame(cv::UMat& dst)
	{
		return m_Stream.read(dst);
	}

//---------------------------------------------------------------------------------------------------------------------

	cv::VideoCapture& Webcam::raw()
	{
		return m_Stream;
	}
	
//---------------------------------------------------------------------------------------------------------------------

	const cv::VideoCapture& Webcam::raw() const
	{
		return m_Stream;
	}

//---------------------------------------------------------------------------------------------------------------------


}
