#include "Mouse.hpp"

#include "../Configuration.hpp"

#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

namespace vt
{

//---------------------------------------------------------------------------------------------------------------------

	constexpr float DRAG_THRESHOLD = 20;
	constexpr float JUMP_THRESHOLD = 150;
	constexpr float STOP_RATE = 0.05f;
	constexpr float DRAG_RATE = 0.8f;

//---------------------------------------------------------------------------------------------------------------------
	
	Mouse::Mouse(const cv::Size& input_region)
	{
		// Assume that the mouse will only be applied to the primary monitor.
		HMONITOR monitor = MonitorFromPoint(POINT{MONITOR_OFFSET}, MONITOR_DEFAULTTOPRIMARY);
		
		MONITORINFO monitor_info{0};
		monitor_info.cbSize = sizeof(MONITORINFO);

		GetMonitorInfo(monitor, &monitor_info);
		
		// Get monitor origin in virtual coordinates. 
		m_InputOffset = cv::Point{
			monitor_info.rcMonitor.left,
			monitor_info.rcMonitor.top
		};

		const auto monitor_width = monitor_info.rcMonitor.right - m_InputOffset.x;
		const auto monitor_height = monitor_info.rcMonitor.bottom - m_InputOffset.y;

		// This is the scaling that needs to be applied to input points. 
		m_InputScaling = cv::Size2f(
			static_cast<float>(monitor_width) / static_cast<float>(input_region.width),
			static_cast<float>(monitor_height ) / static_cast<float>(input_region.height)
		);
	}

//---------------------------------------------------------------------------------------------------------------------

	void Mouse::move(const cv::Point2f& coord, const bool smoothing)
	{
		// Find virtual location of coord within the monitor. 
		const cv::Point2f new_mouse_coord(
			coord.x * m_InputScaling.width + m_InputOffset.x,
			coord.y * m_InputScaling.height + m_InputOffset.y
		);

		// Apply smoothing to the virtual coord. 
		if(smoothing)
		{
			const auto delta = new_mouse_coord - m_MouseCoord;
			const auto dist = cv::norm(delta);
			if(dist > JUMP_THRESHOLD)
				m_MouseCoord = new_mouse_coord;
			else if(dist > DRAG_THRESHOLD)
				m_MouseCoord += DRAG_RATE * delta;
			else
				m_MouseCoord += STOP_RATE * delta;
		}
		else m_MouseCoord = new_mouse_coord;

		SetCursorPos(m_MouseCoord.x, m_MouseCoord.y);
	}
	
//---------------------------------------------------------------------------------------------------------------------

	void Mouse::hold_left()
	{
		INPUT input = {0};
		input.type = INPUT_MOUSE;
		input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
		SendInput(1, &input, sizeof(INPUT));

		m_LeftClickDown = true;
	}

//---------------------------------------------------------------------------------------------------------------------

	void Mouse::hold_right()
	{
		INPUT input = {0};
		input.type = INPUT_MOUSE;
		input.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
		SendInput(1, &input, sizeof(INPUT));

		m_RightClickDown = true;
	}

//---------------------------------------------------------------------------------------------------------------------

	void Mouse::release_hold()
	{
		if(m_LeftClickDown)
		{
			INPUT input = {0};
			input.type = INPUT_MOUSE;
			input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
			SendInput(1, &input, sizeof(INPUT));

			m_LeftClickDown = false;
		}
	
		if(m_RightClickDown)
		{
			INPUT input = {0};
			input.type = INPUT_MOUSE;
			input.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
			SendInput(1, &input, sizeof(INPUT));

			m_RightClickDown = false;
		}
	}

//---------------------------------------------------------------------------------------------------------------------

}