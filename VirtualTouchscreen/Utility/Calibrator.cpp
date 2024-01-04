#include "Calibrator.hpp"

#include "../Configuration.hpp"

namespace vt
{

//---------------------------------------------------------------------------------------------------------------------

	void Calibrator::calibrate_exposure(
		Webcam& webcam,
		const double brightness_target,
		const std::string& window_name,
		const bool auto_destroy_window
	)
	{
		CV_Assert(brightness_target > 0 && brightness_target < 255);

		auto& cam = webcam.raw();

		// Lock the camera focus - assume it is already in focus. 
		cam.set(cv::CAP_PROP_AUTOFOCUS, false);
		cam.set(cv::CAP_PROP_FOCUS, cam.get(cv::CAP_PROP_FOCUS));

		// Lock the camera white balance to neutral.
		// NOTE: this is unsupported by all Windows backends. 
		cam.set(cv::CAP_PROP_AUTO_WB, false);
		cam.set(cv::CAP_PROP_WB_TEMPERATURE, 4500);

		// Disable auto-exposure and gain
		cam.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
		cam.set(cv::CAP_PROP_GAIN, 0);

		// solve for the exposure which doesn't blow out the 
		// projector. We do this by looking at the brightest
		// pixel in the image each exposure level. 
		int exposure_level = 0;
		cv::UMat webcam_sample, intensity;
		double min_brightness, max_brightness;
		do
		{
			cam.set(cv::CAP_PROP_EXPOSURE, exposure_level--);

			capture_colour(webcam, webcam_sample, cv::Scalar::all(255), webcam.latency_ms * 2, 3, window_name);
			cv::cvtColor(webcam_sample, intensity, cv::COLOR_BGR2GRAY);
			cv::minMaxLoc(intensity, &min_brightness, &max_brightness);

			if constexpr (show_auto_exposure_samples)
			{
				cv::imshow(
					"Exposure: " + std::to_string(exposure_level) + "  Max Intensity: " + std::to_string((int)max_brightness + 1),
					intensity
				);
			}
		} while (max_brightness > brightness_target);

		if(auto_destroy_window) cv::destroyWindow(window_name);
	}

//---------------------------------------------------------------------------------------------------------------------

	void Calibrator::capture_colour(
		Webcam& webcam,
		cv::UMat& dst,
		const cv::Scalar& colour,
		const int settle_time_ms,
		const int capture_samples,
		const std::string& window_name,
		const bool auto_destroy_window
	)
	{
		cv::Mat colour_image(1, 1, CV_8UC3);
		colour_image.setTo(colour);

		capture_image(
			webcam,
			dst,
			colour_image,
			settle_time_ms,
			capture_samples,
			window_name,
			auto_destroy_window
		);

		//cv::imshow(
		//	"Sample Colour: (" + std::to_string((int)colour[0]) + "," 
		//	                   + std::to_string((int)colour[1]) + "," 
		//	                   + std::to_string((int)colour[2]) + ")",
		//	dst
		//);
	}

//---------------------------------------------------------------------------------------------------------------------

	void Calibrator::capture_image(
		Webcam& webcam,
		cv::UMat& dst,
		const cv::Mat& image,
		const int settle_time_ms,
		const int capture_samples,
		const std::string& window_name,
		const bool auto_destroy_window
	)
	{
		CV_Assert(settle_time_ms >= 0);
		CV_Assert(capture_samples >= 1);

		// Ensure the output window exists and is in fullscreen.
		make_fullscreen_window(window_name);

		// Show the image on the screen. 
		cv::imshow(window_name, image);
		cv::pollKey();

		// Sleep for the settle time.
		std::this_thread::sleep_for(std::chrono::milliseconds(settle_time_ms));

		// Burn a few frames to remove old buffered frames.  
		webcam.drop_frame();
		webcam.drop_frame();
		webcam.drop_frame();
		webcam.next_frame(dst);
		webcam.next_frame(dst);
		webcam.next_frame(dst);

		// Grab the webcam capture of the image. 
		if (capture_samples > 1)
		{
			// Average out multiple captures
			cv::Mat average(webcam.height, webcam.width, CV_64FC3);
			average.setTo(cv::Scalar::all(0));

			for (int i = 0; i < capture_samples; i++)
			{
				webcam.next_frame(dst);
				cv::accumulate(dst, average);
			}
			cv::divide(average, cv::Scalar::all(capture_samples), average);
			average.convertTo(dst, CV_8UC3);

			//static int i = 0; 
			//cv::imshow(std::to_string(i++), dst);
		}

		if (auto_destroy_window) cv::destroyWindow(window_name);
	}

//---------------------------------------------------------------------------------------------------------------------

	void Calibrator::show_feedback(
		Webcam& webcam,
		const cv::String& top_text,
		const cv::String& bot_text,
		const std::string& window_name,
		const bool auto_destroy_window
	)
	{
		// Ensure the feedback window exists and is in fullscreen.
		auto window_region = make_fullscreen_window(window_name);
		const cv::Size window_size = window_region.size();

		// Find the ideal webcam scaling to fit in the centre of the feedback window. 
		constexpr float HEADER_SIZE = 80, FOOTER_SIZE = 80;
		const float VERTICAL_SPACE = window_size.height - HEADER_SIZE - FOOTER_SIZE;

		const float hs = VERTICAL_SPACE / static_cast<float>(webcam.height);
		const float ws = static_cast<float>(window_size.width) / static_cast<float>(webcam.width);
		const float scaling = std::min(hs, ws);

		const cv::Size webcam_size = cv::Size2f(webcam.width, webcam.height) * scaling;
		const cv::Rect webcam_slot((window_size - webcam_size) / 2, webcam_size);

		// Show the feedback to the user until they press any key. 
		cv::UMat window_frame, webcam_frame, webcam_scaled_frame;
		window_frame.create(window_size, CV_8UC3);
		while (cv::waitKey(webcam.latency_ms) == -1)
		{
			// Reset the draw buffer
			window_frame.setTo(cv::Scalar::all(255));

			// Grab webcam view and copy it onto the frame
			webcam.next_frame(webcam_frame);
			cv::resize(webcam_frame, webcam_scaled_frame, webcam_size);
			webcam_scaled_frame.copyTo(window_frame(webcam_slot));

			// Draw feedback text onto the window frame
			cv::putText(
				window_frame,
				top_text,
				cv::Point(10, 50),
				cv::FONT_HERSHEY_COMPLEX_SMALL,
				2,
				cv::Scalar(0, 0, 0),
				3,
				cv::LINE_AA
			);

			cv::putText(
				window_frame,
				bot_text,
				cv::Point(10, window_size.height - 50),
				cv::FONT_HERSHEY_COMPLEX_SMALL,
				2,
				cv::Scalar(0, 0, 0),
				3,
				cv::LINE_AA
			);

			cv::imshow(window_name, window_frame);
		}

		if (auto_destroy_window) cv::destroyWindow(window_name);
	}

//---------------------------------------------------------------------------------------------------------------------

	cv::Rect Calibrator::make_fullscreen_window(const std::string& window_name)
	{
		// Create the fullscreen window
		cv::namedWindow(window_name, cv::WINDOW_GUI_NORMAL);
		cv::moveWindow(window_name, MONITOR_OFFSET); // TODO: remove when shipping product
		cv::setWindowProperty(window_name, cv::WND_PROP_TOPMOST, true);
		cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, true);
		return cv::getWindowImageRect(window_name);
	}

//---------------------------------------------------------------------------------------------------------------------

}