Great! Below is the updated `README.md` file, including the correct repository link:

```markdown
# MotionMetricsVision

MotionMetricsVision is an advanced object tracking and motion analysis system built using OpenCV and Matplotlib. It tracks multiple objects in real-time, calculates their motion parameters such as speed and acceleration, and visualizes the results dynamically. This project is ideal for applications in surveillance, sports analytics, and automated monitoring systems.

## Features

- **Multi-Object Tracking**: Efficiently tracks multiple objects simultaneously using various OpenCV trackers.
- **Motion Analysis**: Calculates detailed motion parameters including speed and acceleration.
- **Real-Time Visualization**: Provides dynamic visual feedback of tracked objects and their trajectories.
- **Configurable Parameters**: Easily adjustable settings for different tracking and analysis needs.
- **Extensive Logging**: Detailed logging for easy debugging and performance monitoring.

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed. You will also need the following Python packages:

- `opencv-python`
- `opencv-contrib-python`
- `numpy`
- `matplotlib`

You can install these dependencies using pip:

```sh
pip install opencv-python opencv-contrib-python numpy matplotlib
```

### Clone the Repository

```sh
git clone https://github.com/brosiah96/MotionMetricsVision.git
cd MotionMetricsVision
```

## Usage

### Running the Application

To run the application, simply execute the `main.py` script:

```sh
python main.py
```

This will start the video capture, object tracking, and real-time visualization.

### Configuration

You can adjust the tracking and analysis parameters in the `config.py` file.

## Project Structure

```
MotionMetricsVision/
├── config.py
├── main.py
├── object_tracker.py
├── video_capture.py
├── test/
│   ├── test_main.py
│   ├── test_object_tracker.py
│   └── test_video_capture.py
└── README.md
```

- **config.py**: Contains configuration parameters for tracking and analysis.
- **main.py**: The main entry point of the application.
- **object_tracker.py**: Implements the object tracking and motion analysis logic.
- **video_capture.py**: Handles video capture and processing.
- **test/**: Contains unit tests for the various components.

## Contributing

Contributions are welcome! Please fork the repository and use a feature branch. Pull requests are warmly welcome.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The OpenCV team for their powerful computer vision library.
- The Matplotlib community for their versatile plotting library.
```

This `README.md` provides a clear and comprehensive overview of the MotionMetricsVision project, including installation instructions, usage guidelines, project structure, and contribution information.
