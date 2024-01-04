#include "MaskGenerator.hpp"

#include <ScreenVision.h>

#include "../Configuration.hpp"
#include "../Utility/Common.hpp"


namespace vt
{

	using namespace std::chrono;

//---------------------------------------------------------------------------------------------------------------------

	// Objects less than threshold are classified as a shadow.
	constexpr auto SHADOW_OFFSET = 50;
	constexpr auto NOISE_OFFSET = 15;
	
	constexpr auto PREDICTION_RATE_HZ = 60;
	constexpr auto PREDICTION_RATE_MS = 1000 / PREDICTION_RATE_HZ;
	 
//---------------------------------------------------------------------------------------------------------------------
	
	MaskGenerator::MaskGenerator() 
		: m_Runflag(false)
	{
		// Initialize light sharpening kernel.
		cv::Mat({3,3}, {
			0.00f, -0.25f,  0.00f,
		   -0.25f,  2.00f, -0.25f,
			0.00f, -0.25f,  0.00f
		}).copyTo(m_SharpeningKernel);

		// Initialize morph kernel for noise erosion. 
		cv::getStructuringElement(
			cv::MORPH_RECT,
			cv::Size(3, 3)
		).copyTo(m_MorphKernel);
	}

//---------------------------------------------------------------------------------------------------------------------

	void MaskGenerator::start(const Webcam& webcam, const ViewCalibrator& calibration)
	{
		const auto& input_size = calibration.output_resolution();
		m_NextPrediction.create(input_size, CV_32FC3);
		m_ForegroundView.create(input_size, CV_8UC3);
		m_BorderMask.create(input_size, CV_8UC1);
		m_RawFrame.create(input_size, CV_8UC3);

		m_BorderMask.setTo(cv::Scalar::zeros());
		const auto [w, h] = input_size - cv::Size(1, 1);
		cv::line(m_BorderMask, {0,0}, {w,0}, cv::Scalar(255), 3);
		cv::line(m_BorderMask, {w,0}, {w,h}, cv::Scalar(255), 3);
		cv::line(m_BorderMask, {w,h}, {0,h}, cv::Scalar(255), 3);
		cv::line(m_BorderMask, {0,h}, {0,0}, cv::Scalar(255), 3);

		m_AmbientIntensity = calibration.ambient_intensity();

		// Fill in frame queue
		m_FrameQueue.resize(prediction_delay);
		for(auto& buffer : m_FrameQueue)
		{
			buffer.create(input_size, CV_32FC3);
			buffer.setTo(cv::Scalar::zeros());
		}
		m_WriteIndex = 0;

		// Start the prediction thread. 
		m_Runflag = true;
		m_PredictionThread = std::thread(
			&MaskGenerator::predictor_process,
			this,
			calibration.context()
		);
	}

//---------------------------------------------------------------------------------------------------------------------

	void MaskGenerator::segment(const cv::UMat& view, cv::UMat& foreground_mask, cv::UMat& shadow_mask)
	{
		CV_Assert(m_Runflag);

		// Sharpen the input view.
		cv::filter2D(view, m_View, CV_32FC3, m_SharpeningKernel);

		// Read the predicted background. 
		read_prediction(m_Background);

		// Perform dynamic background subtraction via the
		// difference between the prediction and webcam view.
		cv::absdiff(m_Background, m_View, m_Difference);
		cv::transform(m_Difference, m_Score, cv::Matx13f(0.75f, 0.75f, 1.00f));

		// Assume minimal differences belong to background and remove. 
		const auto noise_floor = cv::mean(m_Score, m_BackgroundMask);
		cv::threshold(m_Score, m_Score, noise_floor[0] + NOISE_OFFSET, 255, cv::THRESH_BINARY);
		m_Score.convertTo(foreground_mask, CV_8UC1);

		// Uncomment to see prediction and background side by side.
		if constexpr (show_output_prediction)
		{
			thread_local cv::UMat n1, n2, n3;
			m_View.convertTo(n1, CV_8UC3);
			m_Background.convertTo(n2, CV_8UC3);
			cv::cvtColor(foreground_mask, n3, cv::COLOR_GRAY2BGR);
			vt::imshow_3x1("View vs. Prediction vs. Raw Mask", n1, n2, n3);
			cv::pollKey();
		}

		// Erode the mask to remove remove small noises and thin lines. 
		cv::erode(foreground_mask, foreground_mask, m_MorphKernel, {-1,-1}, 2);

		// Remove any noise that is not connected to the edge of the screen. 
		cv::add(foreground_mask, m_BorderMask, m_NoiseMask);
		cv::floodFill(m_NoiseMask, {0,0}, cv::Scalar(0));
		cv::subtract(foreground_mask, m_NoiseMask, foreground_mask);
		cv::subtract(foreground_mask, m_BorderMask, foreground_mask);

		// Dilate the mask and smooth it to remove jagged edges. 
		cv::dilate(foreground_mask, foreground_mask, m_MorphKernel, {-1,-1}, 2);
		cv::boxFilter(foreground_mask, foreground_mask, -1, cv::Size(5, 5));
		cv::threshold(foreground_mask, foreground_mask, 192, 255, cv::THRESH_BINARY);

		// Find the shadow mask
		cv::bitwise_not(foreground_mask, m_BackgroundMask);
		cv::cvtColor(view, m_ForegroundView, cv::COLOR_BGR2GRAY);
		m_ForegroundView.setTo(cv::Scalar::all(255), m_BackgroundMask);
		cv::threshold(m_ForegroundView, shadow_mask, m_AmbientIntensity + SHADOW_OFFSET, 255, cv::THRESH_BINARY_INV);

		if constexpr (show_backsub_outputs)
		{
			cv::imshow("Foreground Mask", foreground_mask);
			cv::imshow("Shadow Mask", shadow_mask);
			cv::pollKey();
		}
	}

//---------------------------------------------------------------------------------------------------------------------

	void MaskGenerator::stop()
	{
		m_Runflag = false;
		m_PredictionThread.join();
	}

//---------------------------------------------------------------------------------------------------------------------

	void MaskGenerator::predictor_process(ViewProperties calibration)
	{
		auto screen_capture = sv::ScreenCapture::Open(
			MonitorFromPoint(POINT{MONITOR_OFFSET}, MONITOR_DEFAULTTONEAREST)
		);

		if(!screen_capture.has_value())
		{
			std::cerr << "Failed to start screen capture!" << std::endl;
			exit(-1);
		}

		// Create a view calibrator for use with our unique OpenCL context
		ViewCalibrator calibrator(calibration);
		

		// Initialize buffer resources
		const auto buffer_size = calibration.output_resolution;
		cv::UMat raw_capture(buffer_size, CV_8UC4), resize_buffer(buffer_size, CV_8UC3);
		cv::Mat prediction_buffer(buffer_size, CV_32FC3), frame_buffer(buffer_size, CV_8UC3);

		while(m_Runflag)
		{
			// Capture the screen buffer of the monitor. 
			const auto start_time = high_resolution_clock::now();
			if(bool new_frame = screen_capture->read(raw_capture, PREDICTION_RATE_MS - 1); new_frame)
			{
				// If we have a new frame (screen buffer changed) then 
				// downsample and predict its projector-camera output. 
				cv::cvtColor(raw_capture, resize_buffer, cv::COLOR_BGRA2BGR);
				cv::resize(resize_buffer, frame_buffer, calibration.output_resolution);
				
				calibrator.predict(frame_buffer, prediction_buffer);
			}

			// Ensure we always meet the prediction rate timing.   
			while(duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count() < PREDICTION_RATE_MS)
				std::this_thread::yield();

			// NOTE: cv::Mat is needed to transfer between OpenCL contexts.
			{
				std::unique_lock lock(m_PredictionMutex);

				// Push latest frame onto the frame queue
				prediction_buffer.copyTo(m_FrameQueue[m_WriteIndex]);
				m_WriteIndex = (m_WriteIndex + 1) % m_FrameQueue.size();

				frame_buffer.copyTo(m_RawFrame);
			}
		}
	}

//---------------------------------------------------------------------------------------------------------------------

	void MaskGenerator::read_prediction(cv::UMat& dst)
	{
		std::unique_lock lock(m_PredictionMutex);

		// NOTE: read index is write index due to 
		// other thread incrementing it after writing 
		m_FrameQueue[m_WriteIndex].copyTo(dst);

		if constexpr (show_raw_projector_input)
		{
			cv::imshow("Raw Frame", m_RawFrame);
			cv::pollKey();
		}
	}

//---------------------------------------------------------------------------------------------------------------------

}