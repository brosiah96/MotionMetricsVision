# config.py

# Constants for unit conversion
PIXELS_TO_INCHES = 0.026458333
PIXELS_TO_FEET = PIXELS_TO_INCHES / 12  # 12 inches in a foot

# Tracker configuration
MAX_FRAMES_TO_TRACK = 30
MIN_AREA = 500
MAX_FAILURES = 10

# HSV color range for detecting objects of a particular color.
LOWER_COLOR = [0, 0, 150]  # LOWER_WHITE
UPPER_COLOR = [180, 50, 255]  # UPPER_WHITE

# Morphological operations kernel size
KERNEL_SIZE = (5, 5)

# Gaussian blur kernel size
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)

# Contour area filter range
CONTOUR_MIN_AREA = 100  # Minimum area to be considered a valid contour
CONTOUR_MAX_AREA = 1000  # Maximum area to be considered a valid contour

# Plot configuration
PLOT_UPDATE_INTERVAL = 50  # Interval in milliseconds for updating plots

# Calibration configuration
FRAME_WIDTH_INCHES = 48  # Frame width in inches for calibration
FRAME_HEIGHT_INCHES = 45  # Frame height in inches for calibration

# Motion detection thresholds
MOTION_THRESHOLD_PIXELS = 1.0  # Threshold for detecting motion in pixels
