#include <opencv2/opencv.hpp>
#include <iostream>
#include <optional>

#include "Abstractions/Mouse.hpp"
#include "Abstractions/Webcam.hpp"
#include "Systems/ViewCalibrator.hpp"
#include "Systems/MaskGenerator.hpp"
#include "Systems/FingerTracker.hpp"

#include "Configuration.hpp"

//---------------------------------------------------------------------------------------------------------------------


// Forward Declaration
std::optional<std::tuple<cv::Point, bool>> find_touch_action(
	const std::vector<vt::FingerTracker::Fingertip>& fingertips,
	const cv::UMat& foreground_mask, const cv::UMat& shadow_mask,
	const cv::UMat& camera_view
);

//---------------------------------------------------------------------------------------------------------------------

int main(int argc, const char* argv[])
{
	// Obtain webcam hardware ID. 
	int webcam_id = (argc == 2) ? atoi(argv[1]) : WEBCAM_ID;


	// Initialize the webcam
	auto webcam = vt::Webcam::TryCreate(webcam_id, cv::Size(WEBCAM_WIDTH, WEBCAM_HEIGHT), 30);
	if(!webcam.has_value())
	{
		std::cerr << "Failed to load webcam with hardware ID: " << webcam_id << std::endl;
		return -1;
	}
	else std::cout << cv::format("Loaded webcam (%dx%d@%d)\n", webcam->width, webcam->height, webcam->framerate);


	// Calibrate the webcam view
	const cv::Size output_resolution(CALIB_OUTPUT_WIDTH, CALIB_OUTPUT_HEIGHT);
	vt::ViewCalibrator calibrator(output_resolution);
	calibrator.calibrate(*webcam, CALIB_MIN_COVERAGE, CALIB_SETTLE_TIME_MS);

	// Initialize touchscreen systems.
	vt::MaskGenerator mask_generator;
	vt::FingerTracker finger_tracker;
	vt::Mouse mouse(output_resolution);

	// Begin the mask generator
	mask_generator.start(*webcam, calibrator);

	// Run the main processing loop
	cv::UMat raw_frame, screen_frame;
	cv::UMat foreground_mask, shadow_mask;
	auto start_frame = std::chrono::high_resolution_clock::now();
	auto start_process = std::chrono::high_resolution_clock::now();
	while(webcam->next_frame(raw_frame))
	{
		start_process = std::chrono::high_resolution_clock::now();

		if constexpr (show_raw_webcam_view)
		{
			cv::imshow("Raw Capture", raw_frame);
			cv::pollKey();
		}
		
		calibrator.correct(raw_frame, screen_frame);
		
		// Find foreground and shadow masks
		mask_generator.segment(
			screen_frame,
			foreground_mask,
			shadow_mask
		);

		// Detect fingertips in the foreground mask and handle touch registration.
		auto fingertips = finger_tracker.detect(foreground_mask, shadow_mask);
		if(const auto action = find_touch_action(fingertips, foreground_mask, shadow_mask, screen_frame); action.has_value())
		{
			const auto& [point, touch] = *action;

			finger_tracker.focus(point, cv::Size(256, 256));
			mouse.move(point, true);
			
			if(touch) mouse.hold_left();
		}
		else mouse.release_hold();

		// Report total processing latency
		if constexpr (show_latencies)
		{
			const auto elapsed_frame = std::chrono::high_resolution_clock::now() - start_frame;
			const auto elapsed_process = std::chrono::high_resolution_clock::now() - start_process;
			const auto frame_ms = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_frame).count() / 1000.0f;
			const auto process_ms = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_process).count() / 1000.0f;
			std::cout << "Latency: " << process_ms << "/" << frame_ms << "ms (" << process_ms / frame_ms * 100.0f << "%)\n";
			start_frame = std::chrono::high_resolution_clock::now();
		}
	}

	mask_generator.stop();
}

//---------------------------------------------------------------------------------------------------------------------


std::optional<std::tuple<cv::Point, bool>> find_touch_action(
	const std::vector<vt::FingerTracker::Fingertip>& fingertips,
	const cv::UMat& foreground_mask, const cv::UMat& shadow_mask,
	const cv::UMat& camera_view
)
{
	// Process the list of fingertips to find either the 
	// last used fingertip or the oldest new fingertip. 
	// We assume that noise not consistent so does not 
	// have a large age, while a solid fingertip should
	// be able to easily live on for multiple frames. 

	thread_local vt::FingerTracker::Fingertip last_fingertip;
	std::optional<vt::FingerTracker::Fingertip> chosen_fingertip;

	constexpr size_t MIN_FINGER_AGE = 5; 
	size_t oldest_age = MIN_FINGER_AGE;
	
	for(const auto& fingertip : fingertips)
	{
		if(fingertip.id == last_fingertip.id)
		{
			chosen_fingertip = fingertip;
			break;
		}
		
		// Prefer fingertip with higher age. 
		if(fingertip.age >= oldest_age)
		{
			oldest_age = fingertip.age;
			chosen_fingertip = fingertip;
		}
	}

	// If a suitable finger was found, test for touch.  
	if(chosen_fingertip.has_value())
	{
		const auto point = chosen_fingertip->point;
		const auto com = chosen_fingertip->com;
		last_fingertip = *chosen_fingertip;

		// Find the ratio of shadow to foreground in a region
		// around the fingertip. The shadow will coincide with
		// the object that casts it if there is a touch, meaning
		// that the ratio should be minimal, but never zero, as 
		// the shadow will outline the contour of the hand.  
		const int radius = cv::norm(com - point) + 7;
		cv::Rect roi(
			cv::Point(
				std::max(com.x - radius, 0),
				std::max(com.y - radius, 0)
			),
			cv::Point(
				std::min(com.x + radius, shadow_mask.cols - 2),
				std::min(com.y + radius, shadow_mask.rows - 2)
			)
		);


		// Perform touch registration on the finger via ratio test. 
		const auto shadow = cv::countNonZero(shadow_mask(roi));
		const auto foreground = cv::countNonZero(foreground_mask(roi));
		const float ratio = static_cast<float>(shadow) / static_cast<float>(foreground);

		int touch_thresh = 20, hover_thresh = 30;
		if constexpr (show_ratio_patch)
		{
			cv::namedWindow("Ratio Patch", cv::WINDOW_NORMAL);

			static bool trackbar_initialized = false;
			if (!trackbar_initialized)
			{
				cv::createTrackbar("Touch", "Ratio Patch", &touch_thresh, 100);
				cv::createTrackbar("Hover", "Ratio Patch", &hover_thresh, 100);
				cv::resizeWindow("Ratio Patch", cv::Size(640, 480));
				trackbar_initialized = true;
			}

			thread_local cv::UMat patch(612, 512, CV_8UC3);
			patch.setTo(cv::Scalar::zeros());

			cv::resize(camera_view(roi), patch(cv::Rect(0,0,512,512)), {512, 512});
			cv::putText(
				patch,
				std::to_string(ratio),
				{0,600},
				cv::FONT_HERSHEY_COMPLEX_SMALL,
				3, cv::Scalar(255, 255, 255), 2
			);
			cv::imshow("Ratio Patch", patch);
			cv::pollKey();
		}

		// Test for touch
		if(ratio <= static_cast<float>(touch_thresh) / 100.0f)
		{
			return std::make_tuple(point, true);
		}

		// Test for hover
		if(ratio <= static_cast<float>(hover_thresh) / 100.0f)
		{
			return std::make_tuple(point, false);
		}
	}
	return std::nullopt;
}

//---------------------------------------------------------------------------------------------------------------------

