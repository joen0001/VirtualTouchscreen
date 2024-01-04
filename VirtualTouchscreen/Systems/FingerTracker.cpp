#include "FingerTracker.hpp"

#include <algorithm>
#include <numbers>

#include "../Utility/Common.hpp"
#include "../Configuration.hpp"

namespace vt
{
	
//---------------------------------------------------------------------------------------------------------------------

	// Contour Settings
	constexpr auto MIN_CONTOUR_AREA = 500;         // Minimum area of contour. 

	// Arc Test Settings
	constexpr auto ARC_MIN_SCORE = 50;
	constexpr auto ARC_TEST_LENGTH = 450;
	constexpr auto NONMAX_PROXIMIITY = 500;

	// Tracking Settings
	constexpr auto MAX_TRACKING_RANGE = 75;
	constexpr auto MAX_TRACKING_LIFE = 10;
	constexpr auto FOCUS_RESET_TIME = 10;

//---------------------------------------------------------------------------------------------------------------------

	FingerTracker::FingerTracker(){}

//---------------------------------------------------------------------------------------------------------------------
	
	std::vector<FingerTracker::Fingertip> FingerTracker::detect(const cv::UMat& mask, const cv::UMat& shadow_mask)
	{
		std::vector<Fingertip> fingertips;

		// Initialize debug render. 
		if constexpr (show_tracking_output)
		{
			m_DebugRender.create(mask.size(), CV_8UC3);
			m_DebugRender.setTo(cv::Scalar::zeros());
		}
		shadow_mask.copyTo(m_ShadowMask);

		// Update tracking region
		if(--m_TrackingResetTimer <= 0)
		{
			m_TrackingRegion.x = 0;
			m_TrackingRegion.y = 0;
			m_TrackingRegion.width = mask.cols;
			m_TrackingRegion.height = mask.rows;
		}

        // Find all external contours within the focus area.
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(
			mask(m_TrackingRegion), 
			contours, 
			cv::RETR_EXTERNAL, 
			cv::CHAIN_APPROX_NONE, 
			m_TrackingRegion.tl()
		);

		// Find candidate fingertip arcs in the contours.  
		for(auto& contour : contours)
		{
			// Ignore small contours which are likely noise. 
			const auto area = cv::contourArea(contour);
			if(area < MIN_CONTOUR_AREA)
				continue;

			// Draw the contour in the debug render. 
			if constexpr (show_tracking_output)
			{
				for(const auto& point : contour)
				{
					m_DebugRender.at<cv::Vec3b>(point) = cv::Vec3b::all(64);
				}
			}

			// Get the convex hull of the contour. We assume that the hull
			// points represent extremities in the mask, and that fingers
			// will always be at an extremity when pointing outwards. 
			cv::convexHull(contour, m_Extremities, false, false);
			if(m_Extremities.empty())
				continue;
			
			// Draw the convex hull in the debug render. 
			if constexpr (show_tracking_output)
			{
				int last_index = m_Extremities.back();
				for (const auto& index : m_Extremities)
				{
					cv::line(m_DebugRender, contour[last_index], contour[index], {0,0,192}, 1);
					last_index = index;
				}
			}

			// Find an edge point on the convex hull. This is where
			// all our tests should begin to avoid cutting an arc. 
			size_t offset = 0;
			for(; offset < m_Extremities.size(); offset++)
			{
				if(edge_test(contour[m_Extremities[offset]]))
					break;
			}

			// Run arc tests for curved extremities, while also 
			// performing non-max suppression on nearby points. 
			int last = offset, best = -1, best_score = ARC_MIN_SCORE;
			for(size_t i = 0; i < m_Extremities.size(); i++)
			{
				const auto index = m_Extremities[(offset + i) % m_Extremities.size()];
				const auto score = arc_score(contour, index);

				// Test if the extremity is part of the latest cluster.
				const auto v = contour[index] - contour[last];
				if(v.dot(v) > NONMAX_PROXIMIITY)
				{
					// Save max point if there is one 
					if(best != -1)
					{
						m_Candidates.emplace_back(
							contour[best],
							(contour[(best + 15) % contour.size()] + contour[(best - 15 + contour.size())  % contour.size()]) / 2
						);
					}
					
					// Start a new cluster. 
					best_score = ARC_MIN_SCORE;
					last = index;
					best = -1;
				}
				else last = index;

				// Compare score of extremity to the current best in the cluster. 
				if(score > best_score)
				{
					best_score = score;
					best = index;
				}

				// Draw the candidate in the debug render. 
				if constexpr (show_tracking_output)
				{
					if(score > ARC_MIN_SCORE)
					{
						cv::circle(m_DebugRender, contour[index], 1, {255,0,255}, 1);
					}
				}
			}
		}

		
		// Attempt to match the candidates with tracked fingertips. 
		for(int m = 0; m < m_TrackingMemory.size(); m++)
		{
			const auto& [finger, life] = m_TrackingMemory[m];
			int match_index = -1;

			// Find closest candidate within tracking distance.
			double closest_distance_sqr = std::pow(MAX_TRACKING_RANGE, 2);
			for(int c = 0; c < m_Candidates.size(); c++)
			{
				const auto& candidate = m_Candidates[c];

				const auto offset = finger.point - candidate.first;
				const auto distance_sqr = offset.dot(offset);
				if(distance_sqr < closest_distance_sqr)
				{
					closest_distance_sqr = distance_sqr;
					match_index = c;
				}
			}

			// If we found a match, update the tracked fingertip. 
			if(match_index >= 0)
			{
				const auto& candidate = m_Candidates[match_index];

				// Add the updated tracked finger
				fingertips.emplace_back(
					candidate.first,
					candidate.second,
					finger.age + 1,
					finger.id
				);

				// Draw the successfuly tracked candidate
				if constexpr (show_tracking_output)
				{
					cv::circle(m_DebugRender, candidate.first, 2, {000,255,000}, 2);
				}
			
				// Remove tracker & candidate so they can't be re-used. 
				std::swap(m_TrackingMemory[m--], m_TrackingMemory.back());
				m_TrackingMemory.pop_back();

				std::swap(m_Candidates[match_index], m_Candidates.back());
				m_Candidates.pop_back();
			}
		}


		// Add remaining candidates as fingers
		for(const auto& candidate : m_Candidates)
		{
			fingertips.emplace_back(candidate.first, candidate.second, 1, m_NextID++);

			// Draw the newly added candidate
			if constexpr (show_tracking_output)
			{
				cv::circle(m_DebugRender, candidate.first, 2, {000,255,255}, 2);
			}
		}

		// Update state for the next run. 
		m_Candidates.clear();
		update_tracking_memory(fingertips);

		// Output the tracking debug render
		if constexpr (show_tracking_output)
		{
			static bool window_initialized = false;
			if (!window_initialized)
			{
				cv::namedWindow("Fingertip Debug Map", cv::WINDOW_GUI_NORMAL);
				cv::resizeWindow("Fingertip Debug Map", cv::Size(640, 480));
				window_initialized = true;
			}

			cv::imshow("Fingertip Debug Map", m_DebugRender);
			cv::pollKey();
		}

		return fingertips;
	}

//---------------------------------------------------------------------------------------------------------------------

	float FingerTracker::arc_char_max(int x) const
	{
		if(x < 40)
			return -0.05f * (x * x) + 175;
		else 
			return -0.001f * (x * x) + 75.0f; 
	}

//---------------------------------------------------------------------------------------------------------------------

	float FingerTracker::arc_char_min(int x) const
	{
		return std::max<float>(-0.1f * (x * x) + 50.0f, 10);
	}

//---------------------------------------------------------------------------------------------------------------------

	int FingerTracker::arc_score(const std::vector<cv::Point>& contour, const size_t index) const
	{
		const auto& ref = contour[index];

		// We cannot be an arc if we are on the edge. 
		if(edge_test(ref))
			return 0;

		int score = 0;
		for (int i = 4; i < ARC_TEST_LENGTH + 4; i++)
		{
			const auto& prev = contour[(index - i + contour.size()) % contour.size()];
			const auto& next = contour[(index + i) % contour.size()];

			// Finish the test if we hit an edge. 
			if (edge_test(prev) || edge_test(next))
				break;

			// Test that the angle is within angle bounds. 
			const auto angle = fmod(360.0f + signed_angle_between(next - ref, prev - ref), 360.0f);
			if(angle < arc_char_min(i) || angle > arc_char_max(i))
				break;

			score++;
		}

		return score;
	}

//---------------------------------------------------------------------------------------------------------------------

	void FingerTracker::update_tracking_memory(const std::vector<Fingertip>& fingertips)
	{
		// Decrease life of tracking objects and remove dead ones. 
		for(int m = 0; m < m_TrackingMemory.size(); m++)
		{
			auto& [finger, life] = m_TrackingMemory[m];
			if(--life <= 0)
			{
				std::swap(m_TrackingMemory[m--], m_TrackingMemory.back());
				m_TrackingMemory.pop_back();
			}
		}

		// Add new fingertips to memory
		for(const auto& finger : fingertips)
		{
			m_TrackingMemory.emplace_back(finger, MAX_TRACKING_LIFE);
		}
	}

//---------------------------------------------------------------------------------------------------------------------

	bool FingerTracker::edge_test(const cv::Point& pt) const
	{
		return pt.x == m_TrackingRegion.x
			|| pt.y == m_TrackingRegion.y
			|| pt.x == m_TrackingRegion.br().x - 1
			|| pt.y == m_TrackingRegion.br().y - 1;
	}

//---------------------------------------------------------------------------------------------------------------------

	void FingerTracker::focus(const cv::Point& point, const cv::Size& size)
	{
		const auto half_size = size / 2;

		const cv::Point top_left(
			std::max(point.x - half_size.width, 0),
			std::max(point.y - half_size.height, 0)
		);

		// TODO: get size properly
		const cv::Point bot_right(
			std::min(point.x + half_size.width, CALIB_OUTPUT_WIDTH - 1),
			std::min(point.y + half_size.height, CALIB_OUTPUT_HEIGHT - 1)
		);

		m_TrackingRegion = cv::Rect(top_left, bot_right);
		m_TrackingResetTimer = FOCUS_RESET_TIME;
	}

//---------------------------------------------------------------------------------------------------------------------

}