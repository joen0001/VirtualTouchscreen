
// TODO: remove defines

// Test Environment
#define MONITOR_OFFSET 3441,0
#define WEBCAM_ID 1

// Program Configuration
#define WEBCAM_WIDTH 640		
#define WEBCAM_HEIGHT 480
#define CALIB_SETTLE_TIME_MS 1000
#define CALIB_OUTPUT_WIDTH 640
#define CALIB_OUTPUT_HEIGHT 480
#define CALIB_MIN_COVERAGE 0.1
#define CHESSBOARD_SIZE 22,18
#define CAPTURE_SAMPLES 6


// Debug Configuration
constexpr bool show_raw_webcam_view = true;
constexpr bool show_auto_exposure_samples = false;
constexpr bool show_chessboard_detection = false;
constexpr bool show_screen_detect_masks = false;
constexpr bool show_photometric_samples = false;
constexpr bool show_raw_projector_input = false;
constexpr bool show_output_prediction = true;
constexpr bool show_backsub_outputs = false;
constexpr bool show_tracking_output = false;
constexpr bool show_ratio_patch = false;

// Runtime Controls
constexpr bool auto_start_calibration = false;
constexpr bool skip_auto_exposure = false;
constexpr bool show_latencies = false;
constexpr int prediction_delay = 3;