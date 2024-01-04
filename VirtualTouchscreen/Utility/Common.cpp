#include "Common.hpp"

#include <numbers>

namespace vt
{

//---------------------------------------------------------------------------------------------------------------------
	
	cv::Mat make_chessboard(const cv::Size& size, const cv::Vec3b& colour_1, const cv::Vec3b& colour_2)
	{
		CV_Assert(size.width % 2 == 0 && size.height % 2 == 0);
		CV_Assert(size.width > 1 && size.height > 1);

		cv::Mat sub_pattern(2, 2, CV_8UC3);
		sub_pattern.at<cv::Vec3b>(0, 0) = colour_1;
		sub_pattern.at<cv::Vec3b>(0, 1) = colour_2;
		sub_pattern.at<cv::Vec3b>(1, 1) = colour_1;
		sub_pattern.at<cv::Vec3b>(1, 0) = colour_2;

		cv::Mat full_pattern;
		cv::repeat(sub_pattern, size.height / 2, size.width / 2, full_pattern);
		 
		return full_pattern;
	}

//---------------------------------------------------------------------------------------------------------------------
	
	bool error_within(const float actual, const float sample, const float percentage_error)
	{
		return (std::abs(actual - sample) / actual) <= percentage_error;
	}
	
//---------------------------------------------------------------------------------------------------------------------

	float signed_angle_between(const cv::Point2f& v, const cv::Point2f& u)
	{
		return std::atan2(
			u.x * v.y - u.y * v.x,
			u.x * v.x + u.y * v.y
		) * (180.0f / std::numbers::pi);
	}

//---------------------------------------------------------------------------------------------------------------------

	float angle_between(const cv::Point2f& v, const cv::Point2f& u)
	{
		return std::abs(signed_angle_between(v, u));
	}

//---------------------------------------------------------------------------------------------------------------------

	bool between(const float v, const float lower, const float upper)
	{
		return v > lower && v < upper;
	}

//---------------------------------------------------------------------------------------------------------------------

	const size_t xyz_to_3d_index(const int x, const int y, const int z, const int size)
	{
		return (z * size + y) * size + x;
	}

//---------------------------------------------------------------------------------------------------------------------

	int sign(int value)
	{
		return (0 < value) - (value < 0);
	}
	
//---------------------------------------------------------------------------------------------------------------------

	void imshow_2x1(const std::string& title, const cv::UMat& left, const cv::UMat& right)
	{
		CV_Assert(left.type() == right.type());
		thread_local cv::UMat container;
		
		container.create(
			cv::Size(left.cols + right.cols, std::max(left.rows, right.rows)),
			left.type()
		);
		
		left.copyTo(container(cv::Rect({0,0}, left.size())));
		right.copyTo(container(cv::Rect({left.cols,0}, right.size())));

		cv::imshow(title, container);
		cv::pollKey();
	}

	//---------------------------------------------------------------------------------------------------------------------

	void imshow_3x1(const std::string& title, const cv::UMat& left, const cv::UMat& middle, const cv::UMat& right)
	{
		CV_Assert(left.type() == middle.type());
		CV_Assert(left.type() == right.type());
		thread_local cv::UMat container;

		container.create(
			cv::Size(left.cols + middle.cols + right.cols, std::max(left.rows, std::max(right.rows, middle.rows))),
			left.type()
		);

		left.copyTo(container(cv::Rect({0,0}, left.size())));
		middle.copyTo(container(cv::Rect({left.cols,0}, middle.size())));
		right.copyTo(container(cv::Rect({left.cols + middle.cols,0}, right.size())));

		cv::imshow(title, container);
		cv::pollKey();
	}

//---------------------------------------------------------------------------------------------------------------------

}

