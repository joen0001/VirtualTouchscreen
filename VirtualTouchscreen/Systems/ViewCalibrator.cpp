#include "ViewCalibrator.hpp"

#include <thread>
#include <chrono>
#include <bitset>
#include <limits>
#include <numeric>
#include <opencv2/core/ocl.hpp>

#include "../Configuration.hpp"
#include "../Utility/Common.hpp"

namespace vt
{

//---------------------------------------------------------------------------------------------------------------------

	ViewCalibrator::ViewCalibrator(const cv::Size& output_resolution)
		: m_CorrectionMap(output_resolution, CV_32FC2, cv::Scalar::zeros()),
		  m_ViewHomography(cv::Mat::eye(3, 3, CV_32FC1)),
		  m_OutputResolution(output_resolution)
	{
		CV_Assert(output_resolution.width > 0 && output_resolution.height > 0);
	}

//---------------------------------------------------------------------------------------------------------------------

	ViewCalibrator::ViewCalibrator(const ViewProperties& context)
		: m_ViewHomography(context.view_homography),
		  m_CorrectionMap(context.correction_map),
		  m_OutputResolution(context.output_resolution),
		  m_ScreenContour(context.screen_contour),
		  m_ColourMap(context.colour_map)
	{
		context.reflectance_map.copyTo(m_ReflectanceMap);
	}

//---------------------------------------------------------------------------------------------------------------------

	const cv::Size& ViewCalibrator::output_resolution() const
	{
		return m_OutputResolution;
	}

//---------------------------------------------------------------------------------------------------------------------

	float ViewCalibrator::ambient_intensity() const
	{
		const auto ambient_colour = m_ColourMap[0];

		return (1.0f/3.0f) * (ambient_colour[0] + ambient_colour[1] + ambient_colour[2]);
	}

//---------------------------------------------------------------------------------------------------------------------

	void ViewCalibrator::calibrate(
		Webcam& webcam,
		const float min_coverage,
		const int settle_time_ms
	)
	{
		CV_Assert(webcam.is_open());

		// Initialize the fullscreen calibration window for the user. 
		const cv::String window_name = "Screen Calibrator";
		const auto screen_region = make_fullscreen_window(window_name);

		// Get the user to position their camera correctly.
		if constexpr (!auto_start_calibration)
		{
			show_feedback(
				webcam,
				"Please ensure the entire screen is visible and in focus!",
				"Press any key to start the calibration...",
				window_name
			);
		}
		
		// The calibration colours used for detecting the screen and later
		// photometric calibration. They are chosen based on their apparent
		// brightness and high green component, which is robust in practice. 
		const std::vector<cv::Scalar> calibration_colours = {
			 cv::Scalar(255,255,255), cv::Scalar(000,255,000),
			 cv::Scalar(255,255,000), cv::Scalar(000,255,255)
		};

		std::vector<cv::UMat> colour_samples(calibration_colours.size());
		cv::UMat chessboard_sample;

		// Run interactive calibration
		while(true)
		{
			// Calibrate the webcam properties. 
			if constexpr (!skip_auto_exposure)
			{
				calibrate_exposure(webcam, 250, window_name);
			}

			// Capture all required colour samples
			for(int i = 0; i < colour_samples.size(); i++)
			{
				capture_colour(
					webcam,
					colour_samples[i],
					calibration_colours[i],
					settle_time_ms,
					CAPTURE_SAMPLES,
					window_name
				);
				
				// Uncomment to show colour samples
				//cv::imshow(std::to_string(i), colour_samples[i]);
				//cv::pollKey();
			}

			// Capture chessboard pattern for geometric lens distortion calibration. 
			const cv::Size chessboard_size(CHESSBOARD_SIZE);
			cv::Mat chessboard_pattern = make_chessboard(
				chessboard_size, cv::Vec3b::all(0), cv::Vec3b::all(255)
			);

			capture_image(
				webcam,
				chessboard_sample,
				chessboard_pattern,
				settle_time_ms,
				CAPTURE_SAMPLES,
				window_name
			);

			// Find the geometric calibration model using the chessboard and colour samples. 
			const auto screen_corners = find_geometric_model(
				calibration_colours, 
				colour_samples,
				chessboard_sample,
				chessboard_size
			);
			
			if (!screen_corners.has_value())
			{
				// TODO: proper feedback message
				show_feedback(
					webcam,
					"Failed to find screen or chessboard corners",
					"Press any key to try again",
					window_name
				);
				continue;
			}
			else m_ScreenContour = *screen_corners;


			// Check that the detected screen region meets the minimum coverage constraints. 
			if(cv::contourArea(m_ScreenContour) < min_coverage * m_OutputResolution.area())
			{
				show_feedback(
					webcam,
					"Please move the camera closer",
					"Press any key to try again",
					window_name
				);
				continue;
			}


			// Correct all the colour samples using the geometric calibration. 
			std::vector<cv::UMat> corrected_wgcy_samples(calibration_colours.size());
			for(size_t i = 0; i < colour_samples.size(); i++)
			{
				correct(colour_samples[i], corrected_wgcy_samples[i]);
				
				// Uncomment to see corrected colour samples. 
				//cv::imshow(std::to_string(i), corrected_gcy_samples[i]);
				//cv::pollKey();
			}

			// Find the photometric model using all our captured colour samples. 
			find_photometric_model(
				webcam,
				settle_time_ms,
				window_name,
				corrected_wgcy_samples[0]
			);

			break;
		}

		// Show results by drawing the screen outline on the chessboard sample. 
		cv::Point2f last_point = m_ScreenContour.back();
		for(const auto& point : m_ScreenContour)
		{
			cv::line(chessboard_sample, last_point, point, cv::Scalar(255, 0, 255), 2);
			last_point = point;
		}
		
		// Show the result to the user for a few seconds, or until a key is pressed.
		cv::imshow(window_name, chessboard_sample);
		cv::waitKey(2000);

		cv::destroyWindow(window_name);
	}

//---------------------------------------------------------------------------------------------------------------------

	std::optional<std::vector<cv::Point2f>> ViewCalibrator::find_geometric_model(
		const std::vector<cv::Scalar>& colours,
		const std::vector<cv::UMat>& samples,
		const cv::UMat& chessboard_sample,
		const cv::Size& chessboard_size
	)
	{
		CV_Assert(chessboard_size.width > 2 && chessboard_size.height > 2);

		const auto webcam_resolution = chessboard_sample.size();

		// Find raw screen contour using the given colour samples.
		const auto screen_corners = detect_screen(colours, samples);
		if (!screen_corners.has_value())
		{
			std::cout << "Failed to find screen contour \n";
			return std::nullopt;
		}

		// Use screen contour to find chessboard corners in the chessboard sample.
		auto chessboard_corners = detect_chessboard(*screen_corners, chessboard_sample, chessboard_size);
		if (!chessboard_corners.has_value())
		{
			std::cout << "Failed to find chessboard corners \n";
			return std::nullopt;
		}

		// Use the chessboard corners to calculate a rough lens correction map.
		const cv::Size2f chessboard_square_size(
			static_cast<float>(m_OutputResolution.width) / static_cast<float>(chessboard_size.width),
			static_cast<float>(m_OutputResolution.height) / static_cast<float>(chessboard_size.height)
		);

		std::vector<cv::Point2f> ideal_chessboard_corners;
		for (int r = 1; r < chessboard_size.height; r++)
		{
			for (int c = 1; c < chessboard_size.width; c++)
			{
				ideal_chessboard_corners.emplace_back(
					c * chessboard_square_size.width,
					r * chessboard_square_size.height
				);
			}
		}

		// Collect chessboard samples in a vector of point vectors. 
		std::vector<std::vector<cv::Point3f>> ideal_chessboard_corner_samples(1);
		for (auto& p : ideal_chessboard_corners) ideal_chessboard_corner_samples.back().emplace_back(p.x, p.y, 0);

		std::vector<std::vector<cv::Point2f>> chessboard_corner_samples(1);
		chessboard_corner_samples.back() = *chessboard_corners;

		// Find the intrinsic camera properties using the points. 
		cv::Mat camera_matrix;
		std::vector<double> distortion_coefficients;
		cv::calibrateCamera(
			ideal_chessboard_corner_samples,
			chessboard_corner_samples,
			webcam_resolution,
			camera_matrix,
			distortion_coefficients,
			cv::noArray(),
			cv::noArray()
		);

		// Optimize the camera matrix for better scaling.
		cv::Mat optimal_camera_matrix = cv::getOptimalNewCameraMatrix(
			camera_matrix,
			distortion_coefficients,
			webcam_resolution,
			1.0,
			webcam_resolution,
			nullptr,
			false
		);

		// Initialize lens correction map.
		cv::UMat lens_correction_map;
		cv::initUndistortRectifyMap(
			camera_matrix,
			distortion_coefficients,
			cv::noArray(),
			optimal_camera_matrix,
			webcam_resolution,
			CV_32FC2,
			lens_correction_map,
			cv::noArray()
		);


		// Apply lens correction on the samples.
		cv::UMat corrected_chessboard;
		cv::remap(chessboard_sample, corrected_chessboard, lens_correction_map, cv::noArray(), cv::INTER_LANCZOS4);

		std::vector<cv::UMat> corrected_samples(samples.size());
		for (int i = 0; i < corrected_samples.size(); i++)
		{
			cv::remap(samples[i], corrected_samples[i], lens_correction_map, cv::noArray(), cv::INTER_LANCZOS4);

			//cv::imshow("Remaped - " + std::to_string(i), corrected_samples[i]);
			//cv::pollKey();
		}

		// Re-detect screen using the lens corrected samples
		auto corrected_screen_corners = detect_screen(colours, corrected_samples);
		if (!corrected_screen_corners.has_value())
		{
			std::cout << "Failed to find screen contour in lens corrected samples \n";
			return std::nullopt;
		}

		// Find chessboard corners again - this time we use them for the view correction 
		auto corrected_chessboard_corners = detect_chessboard(*corrected_screen_corners, corrected_chessboard, chessboard_size);
		if (!corrected_chessboard_corners.has_value())
		{
			std::cout << "Failed to find corners in lens corrected chessboard \n";
			return std::nullopt;
		}


		// Add the chessboard and screen corners to our homography screen sample points. 
		std::vector<cv::Point2f> screen_points;
		screen_points.insert(screen_points.end(), corrected_screen_corners->begin(), corrected_screen_corners->end());
		screen_points.insert(screen_points.end(), corrected_chessboard_corners->begin(), corrected_chessboard_corners->end());

		// Create ideal screen contour
		const cv::Point2f tl(0, 0), br = cv::Point2f(m_OutputResolution);
		std::vector<cv::Point2f> ideal_screen_corners;
		ideal_screen_corners.emplace_back(tl.x, tl.y);
		ideal_screen_corners.emplace_back(tl.x, br.y);
		ideal_screen_corners.emplace_back(br.x, br.y);
		ideal_screen_corners.emplace_back(br.x, tl.y);

		std::vector<cv::Point2f> ideal_corners;
		ideal_corners.insert(ideal_corners.end(), ideal_screen_corners.begin(), ideal_screen_corners.end());
		ideal_corners.insert(ideal_corners.end(), ideal_chessboard_corners.begin(), ideal_chessboard_corners.end());

		// Generate a homography to rectify the webcam view to only the screen. 
		cv::UsacParams usac_params;
		usac_params.confidence = 0.999;
		usac_params.threshold = 3;
		usac_params.maxIterations = 1000;
		usac_params.sampler = cv::SAMPLING_UNIFORM;
		usac_params.score = cv::SCORE_METHOD_MAGSAC;
		usac_params.final_polisher = cv::MAGSAC;
		usac_params.final_polisher_iterations = 10;
		usac_params.loMethod = cv::LOCAL_OPTIM_SIGMA;
		usac_params.loIterations = 10;
		usac_params.loSampleSize = 20;
		m_ViewHomography = cv::findHomography(screen_points, ideal_corners, cv::noArray(), usac_params);

		// Apply the homography to the lens distortion map to combine them
		cv::warpPerspective(lens_correction_map, m_CorrectionMap, m_ViewHomography, m_OutputResolution, cv::INTER_LANCZOS4);

		// Return original screen corners
		return screen_corners;
	}

//---------------------------------------------------------------------------------------------------------------------
	
	// TODO: put this somewhere better
	constexpr auto CMAP_SIZE = 8;
	constexpr auto CMAP_STEP = 1.0f / (CMAP_SIZE - 1.0f);

//---------------------------------------------------------------------------------------------------------------------

	void ViewCalibrator::find_photometric_model(
		Webcam& webcam,
		const int settle_time_ms,
		const std::string& window_name,
		const cv::UMat& white_sample
	)
	{
		cv::UMat capture_buffer(cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY);
		cv::UMat sample_buffer(cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY);
		cv::Mat cpu_buffer;

		// Process the white sample. 
		cv::Mat white_response;
		const auto white_point = cv::mean(white_sample);
		white_sample.convertTo(white_response, CV_32FC3);

		// Estimate spatial reflectance of all pixels using white sample. 
		m_ReflectanceMap.create(m_OutputResolution, CV_32FC3);
		m_ReflectanceMap.forEach<cv::Vec3f>([&](cv::Vec3f& ref, const int coord[2]) {
			const auto& response = white_response.at<cv::Vec3f>(coord[0], coord[1]);

			ref[0] = response[0] / white_point[0];
			ref[1] = response[1] / white_point[1];
			ref[2] = response[2] / white_point[2];
		});

		// Capture the photometric sample colours. 
		for(int k = 0; k < 2; k++)
		{
			// Fill in the 16x16 colour pattern
			cv::Mat pattern(16, 16, CV_8UC3);
			for(int i = 0; i < 256; i++)
			{
				// Convert the map index to a colour
				const int map_index = (k * 256) + i;

				const int x =  map_index % CMAP_SIZE;
				const int y = (map_index / CMAP_SIZE) % CMAP_SIZE;
				const int z =  map_index / (CMAP_SIZE * CMAP_SIZE);

				pattern.at<cv::Vec3b>(i) = cv::Vec3b(
					cv::saturate_cast<uint8_t>(x * CMAP_STEP * 255.0f),
					cv::saturate_cast<uint8_t>(y * CMAP_STEP * 255.0f),
					cv::saturate_cast<uint8_t>(z * CMAP_STEP * 255.0f)
				);
			}

			// Capture and correct the colour pattern. 
			capture_image(webcam, capture_buffer, pattern, settle_time_ms, CAPTURE_SAMPLES, window_name);
			correct(capture_buffer, sample_buffer);
			sample_buffer.convertTo(cpu_buffer, CV_32FC3);

			// Show the colour patterns 
			if constexpr (show_photometric_samples)
			{
				thread_local cv::UMat tmp;
				cv::resize(pattern, tmp, sample_buffer.size(), 0, 0, cv::INTER_NEAREST);
				imshow_2x1("Photometric Pattern " + std::to_string(k), tmp, sample_buffer);
				cv::pollKey();
			}

			// Size of each colour sample in the sample buffer. 
			const auto sample_size = cv::Size(
				m_OutputResolution.width / pattern.cols, 
				m_OutputResolution.height / pattern.rows
			);

			// Fill in the colour map using the captured pattern colours. 
			for(int r = 0; r < pattern.rows; r++)
			{
				for(int c = 0; c < pattern.cols; c++)
				{
					const cv::Rect roi({c * sample_size.width, r * sample_size.height}, sample_size);
					
					// Grab the average measured colour, taking into account the reflectance. 
					cv::Vec3f measured(0, 0, 0);
					for(int rr = 0; rr < roi.height; rr++)
					{
						for (int rc = 0; rc < roi.width; rc++)
						{
							const auto& raw = cpu_buffer.at<cv::Vec3f>(rr + roi.y, rc + roi.x);
							const auto& ref = m_ReflectanceMap.at<cv::Vec3f>(rr + roi.y, rc + roi.x);

							measured += cv::Vec3f(
								raw[0] / ref[0],
								raw[1] / ref[1],
								raw[2] / ref[2]
							);
						}
					}
					measured /= roi.area();
					
					// Insert sample colour into the colour map. 
					const int map_index = (k * 256) + (r * pattern.rows) + c;
					m_ColourMap[map_index] = cv::Vec3f(measured[0], measured[1], measured[2]);
				}
			}
		}
	}

//---------------------------------------------------------------------------------------------------------------------

	void ViewCalibrator::predict(
		const cv::Mat& src,
		cv::Mat& dst
	) const
	{
		CV_Assert(src.type() == CV_8UC3);
		dst.create(src.size(), CV_32FC3);

		src.forEach<cv::Vec3b>([&](const cv::Vec3b& colour, const int coord[2]) {
			// Normalize the colour. 
			const auto norm_col = cv::Vec3f(colour) / 255.0f;

			// Locate the sub-cube within the map. 
			const int x = static_cast<int>(norm_col[0] / CMAP_STEP);
			const int y = static_cast<int>(norm_col[1] / CMAP_STEP);
			const int z = static_cast<int>(norm_col[2] / CMAP_STEP);
			const auto sub_coord = cv::Vec3f(x, y, z) * CMAP_STEP;

			// Perform trillinear interpolation of map colours. 
			const auto tlerp_factors = (norm_col - sub_coord) / CMAP_STEP;
			const auto prediction = tlerp<cv::Vec3f>(
				m_ColourMap[xyz_to_3d_index(x, y, z, CMAP_SIZE)],
				m_ColourMap[xyz_to_3d_index(x, y + 1, z, CMAP_SIZE)],
				m_ColourMap[xyz_to_3d_index(x + 1, y + 1, z, CMAP_SIZE)],
				m_ColourMap[xyz_to_3d_index(x + 1, y, z, CMAP_SIZE)],
				m_ColourMap[xyz_to_3d_index(x, y, z + 1, CMAP_SIZE)],
				m_ColourMap[xyz_to_3d_index(x, y + 1, z + 1, CMAP_SIZE)],
				m_ColourMap[xyz_to_3d_index(x + 1, y + 1, z + 1, CMAP_SIZE)],
				m_ColourMap[xyz_to_3d_index(x + 1, y, z + 1, CMAP_SIZE)],
				tlerp_factors[0], tlerp_factors[1], tlerp_factors[2]
			);

			const auto& reflectance = m_ReflectanceMap.at<cv::Vec3f>(coord[0], coord[1]);
			cv::Vec3f& final_colour = dst.at<cv::Vec3f>(coord[0], coord[1]);
			final_colour[0] = prediction[0] * reflectance[0];
			final_colour[1] = prediction[1] * reflectance[1];
			final_colour[2] = prediction[2] * reflectance[2];
		});
	}

//---------------------------------------------------------------------------------------------------------------------

	void ViewCalibrator::correct(
		const cv::UMat& src,
		cv::UMat& dst
	) const
	{
		cv::remap(src, dst, m_CorrectionMap, cv::noArray(), cv::INTER_CUBIC);
	}

//---------------------------------------------------------------------------------------------------------------------

	ViewProperties ViewCalibrator::context() const
	{
		ViewProperties context;
		m_ReflectanceMap.copyTo(context.reflectance_map);
		context.output_resolution = m_OutputResolution;
		context.view_homography = m_ViewHomography;
		context.screen_contour = m_ScreenContour;
		context.correction_map = m_CorrectionMap;
		context.colour_map = m_ColourMap;
		return context;
	}

//---------------------------------------------------------------------------------------------------------------------

	std::optional<std::vector<cv::Point2f>> ViewCalibrator::detect_screen(
		const std::vector<cv::Scalar>& colours,
		const std::vector<cv::UMat>& samples
	) const
	{
		CV_Assert(samples.size() == colours.size());
		CV_Assert(samples.size() > 0);

		// Fill in all colour masks
		cv::UMat difference, mask;
		std::vector<cv::UMat> colour_masks;
		for (int i = 0; i < colours.size(); i++)
		{
			// Create colour mask by detecting closest calibration colours. 
			cv::absdiff(samples[i], colours[i], difference);
			cv::cvtColor(difference, mask, cv::COLOR_BGR2GRAY);
			cv::threshold(mask, mask, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);

			colour_masks.push_back(mask.clone());

			if constexpr (show_screen_detect_masks)
			{
				cv::imshow(
					"Mask Colour: (" + std::to_string((int)colours[i][0]) + ","
					+ std::to_string((int)colours[i][1]) + ","
					+ std::to_string((int)colours[i][2]) + ")",
					mask
				);
				cv::pollKey();
			}
		}

		// Find the union of all colour masks, only the screen area should survive.
		mask.setTo(cv::Scalar(255));
		for (auto& colour_mask : colour_masks)
		{
			cv::bitwise_and(mask, colour_mask, mask);
		}

		// Uncomment to show resulting screen mask
		if constexpr (show_screen_detect_masks)
		{
			cv::imshow("Screen Mask", mask);
			cv::pollKey();
		}

		// Assume the screen region is represented by the largest external contour in the mask.
		std::vector<std::vector<cv::Point>> external_contours;
		cv::findContours(mask, external_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		if (external_contours.empty())
		{
			std::cerr << "Screen detection failed - no contour was detected." << std::endl;
			return std::nullopt;
		}

		auto screen_contour = *std::max_element(external_contours.begin(), external_contours.end(),
			[](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); }
		);


		// If the screen contour is properly detected as a quadrilateral, 
		// then we should be able to simplify it to only four points here.
		cv::approxPolyDP(screen_contour, screen_contour, 4, true);
		if (const auto& vertices = screen_contour.size(); vertices != 4)
		{
			std::cerr << "Screen detection failed - contour was " << vertices << " vertices." << std::endl;
			return std::nullopt;
		}


		// Convert the screen contour to subpixel corner coordinates. 
		// To test that the whole screen is visible, we must also check 
		// that the screen contour is not touching the edges of the mask. 
		std::vector<cv::Point2f> corners, ordered_corners(4);
		for (const auto& vertex : screen_contour)
		{
			corners.push_back(vertex);

			if (vertex.x <= 0 || vertex.y <= 0 || vertex.x >= mask.cols - 1 || vertex.y >= mask.rows - 1)
			{
				std::cerr << "Screen detection failed - screen touches border of image" << std::endl;
				return std::nullopt;
			}
		}
		cv::cornerSubPix(mask, corners, cv::Size(30, 30), { -1,-1 }, cv::TermCriteria(1, 500, 0));


		// Re-order the contour vertices to be counter-clockwise from the top left. 
		// We do this by sorting the corners into their correct index based on which 
		// quadrant they end up with in relation to the centroid. Note that we have
		// a guaranteed four points in the corners vector by this point in the function. 
		const auto centroid = 0.25f * (corners[0] + corners[1] + corners[2] + corners[3]);
		for (const auto& corner : corners)
		{
			size_t index = 0;
			if (corner.x < centroid.x)
				index = (corner.y < centroid.y) ? 0 : 1;
			else
				index = (corner.y < centroid.y) ? 3 : 2;

			ordered_corners[index] = corner;
		}

		return ordered_corners;
	}

//---------------------------------------------------------------------------------------------------------------------

	std::optional<std::vector<cv::Point2f>> ViewCalibrator::detect_chessboard(
		const std::vector<cv::Point2f>& screen_bounds,
		const cv::UMat& chessboard_sample,
		const cv::Size& chessboard_size
	) const
	{
		const cv::Size inner_pattern_size(chessboard_size.width - 1, chessboard_size.height - 1);
		
		// The corner finder doesn't like it when the chessboard pattern 
		// doesn't have a border, so use the screen bounds to add our own. 
		std::vector<std::vector<cv::Point>> screen_contour(1);
		for(const auto& pt : screen_bounds) screen_contour[0].push_back(pt);

		cv::UMat bordered_chessboard_sample(chessboard_sample.size(), CV_8UC3);
		bordered_chessboard_sample.setTo(cv::Scalar::zeros());
		cv::drawContours(bordered_chessboard_sample, screen_contour, -1, cv::Scalar(255, 255, 255), cv::FILLED);
		cv::bitwise_not(bordered_chessboard_sample, bordered_chessboard_sample);
		cv::add(bordered_chessboard_sample, chessboard_sample, bordered_chessboard_sample);

		if constexpr (show_chessboard_detection)
		{
			cv::imshow("Detection Pattern", bordered_chessboard_sample);
			cv::pollKey();
		}

		// Detect the corners of the chessboard pattern. 
		std::vector<cv::Point2f> corners;
		const bool corners_found = cv::findChessboardCorners(
			bordered_chessboard_sample,
			inner_pattern_size,
			corners
		);

		if(corners_found)
		{
			if constexpr (show_chessboard_detection)
			{
				cv::drawChessboardCorners(bordered_chessboard_sample, inner_pattern_size, corners, corners_found);
				cv::imshow("Chessboard Corners", bordered_chessboard_sample);
				cv::pollKey();
			}

			return corners;
		}
		else 
		{
			std::cout << "Failed to detect chessboard corners" << std::endl;
			return std::nullopt;
		}
	}

//---------------------------------------------------------------------------------------------------------------------

}
