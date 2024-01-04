#include <opencv2/opencv.hpp>

namespace vt
{

	cv::Mat make_chessboard(const cv::Size& size, const cv::Vec3b& colour_1, const cv::Vec3b& colour_2);

	bool error_within(const float actual, const float sample, const float percentage_error);

	float signed_angle_between(const cv::Point2f& v, const cv::Point2f& u);
	
	float angle_between(const cv::Point2f& v, const cv::Point2f& u);

	bool between(const float v, const float lower, const float upper);

	const size_t xyz_to_3d_index(const int x, const int y, const int z, const int size);

	int sign(int value);



	void imshow_2x1(const std::string& title, const cv::UMat& left, const cv::UMat& right);

	void imshow_3x1(const std::string& title, const cv::UMat& left, const cv::UMat& middle, const cv::UMat& right);



	template<typename T>
	T lerp(const T& v0, const T& v1, const float x)
	{ 
		return v0 * (1.0f - x) + v1 * x;
	}

	template<typename T>
	T blerp(const T& v00, const T& v01, const T& v11, const T& v10, float x, float y)
	{
		return lerp(lerp(v00, v10, x), lerp(v01, v11, x), y);
	}

	template<typename T>
	T tlerp(
		const T& v000, const T& v010, const T& v110, const T& v100,
		const T& v001, const T& v011, const T& v111, const T& v101,
		float x, float y, float z
	)
	{
		return lerp(blerp(v000, v010, v110, v100, x, y), blerp(v001, v011, v111, v101, x, y), z);
	}
}