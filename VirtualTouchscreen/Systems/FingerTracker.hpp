#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

namespace vt
{
	class FingerTracker
	{
	public:

		struct Fingertip
		{
			cv::Point point;
			cv::Point com;

			// Statistics 
			size_t age = 0;
			size_t id = 0;
		};


	public:

		FingerTracker();

		std::vector<Fingertip> detect(const cv::UMat& foreground_mask, const cv::UMat& shadow_mask);

		void focus(const cv::Point& point, const cv::Size& size);

	private:

		float arc_char_min(int x) const;
		
		float arc_char_max(int x) const;

		int arc_score(const std::vector<cv::Point>& contour, const size_t index) const;

		void update_tracking_memory(const std::vector<Fingertip>& fingertips);

		bool edge_test(const cv::Point& point) const;

	private:
		cv::Rect m_TrackingRegion;
		int m_TrackingResetTimer = 0;
		inline static size_t m_NextID = 0;
		
		cv::Mat m_ShadowMask;
		std::vector<int> m_Extremities;
		std::vector<std::pair<cv::Point, cv::Point>> m_Candidates;
		std::vector<std::tuple<Fingertip, int>> m_TrackingMemory;
		
		cv::Mat m_DebugRender;
	};
}

