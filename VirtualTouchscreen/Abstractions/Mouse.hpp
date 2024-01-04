#pragma once

#include <opencv2/opencv.hpp>

namespace vt
{

	class Mouse
	{
	public:

		Mouse(const cv::Size& input_region);

		void move(const cv::Point2f& coord, const bool smoothing);

		void hold_left();

		void hold_right();
		
		void release_hold();

	private:
		cv::Point2f m_MouseCoord;
		bool m_LeftClickDown = false;
		bool m_RightClickDown = false;

		cv::Point m_InputOffset;
		cv::Size2f m_InputScaling;
	};

}