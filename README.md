## Thesis Summary

### Identity

* **Project Name:** Stesis (Stereo-based Sonification Interface System)
* **Thesis Title:** World Navigation Hat – Development of a Wearable Navigation Aid using AIoT for the Visually Impaired

### Problem

Despite advancements in assistive technology, the visually impaired continue to face significant barriers in physical navigation and digital accessibility (SDG 10). Existing Visual Sensory Substitution Devices (VSSDs) often fail due to **sensory overload**. By continuously flooding the user with audio data, these devices prevent "subconscious integration" to the brain's ability to process signals intuitively, leading to cognitive fatigue and device abandonment.

### Solution

**Stesis** is a wearable "smart hat" powered by a **Raspberry Pi 4** (tethered to a pocketed power bank). It utilizes a dual-mode interface to balance information flow:

* **Physical Mode (Navigation):** Uses stereo-rectified depth mapping (OpenCV) to convert the 3D environment into binaural "radar-like" audio. This processing is local (non-AI), ensuring low latency (<100ms) for real-time safety.
* **Digital Mode (Interaction):** Uses the right camera to detect hand gestures (MediaPipe), enabling control of external applications like for AI and IoT features.
* **Selective Filtering:** Integrated MPU6050 head-tracking logic allows users to "toggle" data streams or request specific depth scans via head gestures. This mimics natural vision (looking where one wants to see), significantly reducing the constant noise inherent in traditional VSSDs.

### Hypothesis

Prioritizing **selective filtering** via intentional head gestures will minimize sensory overload and fatigue compared to continuous audio streaming, thereby increasing navigation efficiency and accelerating the user's ability to subconsciously integrate the device's feedback.

### Related Studies

* [**Louis Commere, and Jean Rouat (2023):**](https://arxiv.org/pdf/2304.05462) The most newest visual sensory subtitution device that both uses depth and sonification, being the most closest to our study and will be used as the baseline to test the null hypotesis. *Limitation:* It provides only localized, single-target directional information, severely restricting global scene comprehension. Users cannot perceive a general room outline simultaneously; instead, they are forced into tedious, sequential "Roomba-like" scanning and must rely heavily on working memory to construct a complete spatial map.
* [**Eye-Tracking Sonification (2025):**](https://doi.org/10.3389/frvir.2025.1598776) Filters VR environments using eye-tracking. *Limitation:* Not tested on real-world visual data; relies on eye movement, which may not be functional for all visually impaired users.
* [**Sound of Vision (2018):**](https://doi.org/10.1097/OPX.0000000000001284) Generates continuous 3D audio-haptic maps. *Limitation:* Suffers from sensory overload affecting long-term adoption. Stesis addresses this via gesture-based filtering.
* [**Eye Cane (2015):**](https://doi.org/10.3233/RNN-130351) Uses infrared for single-point distance sound. *Limitation:* Lacks the spatial context and "scene outline" Stesis provides.

### Theoretical Basis

Stesis mimics human visual hierarchy: the brain perceives spatial geometry (outlines) before details. To bridge the gap between high-bandwidth vision and low-bandwidth hearing, Stesis utilizes a **Time-Frequency-Intensity** mapping schema (Turns 2D visual data to 1D audio data):

1. **Horizontal Position  Temporal Sweep (Time):** The audio pans across the stereo field (Left  Right), allowing the user to visualize obstacle location based on *when* and *where* the sound occurs during the sweep.
2. **Vertical Position  Frequency (Pitch):** Relative to the floor (Reference Plane):
* *High Pitch:* Head level (Duck).
* *Mid Pitch:* Torso level (Collision risk).
* *Low Pitch:* Drop-offs/Stairs (Step down).

3. **Distance  Volume (Intensity):** Louder audio indicates proximity. Fast-approaching objects trigger a distinct "stereo entity" overlay for immediate collision warnings.

### System Design

The device first captures the readings of the IMU to determine what does the user want to hear, if the user is facing downwards indicating they want to see their obstacles then the device generates the depth map from the stereo images and places it into a radar map for it to be sonified to the earphones, if the user is facing towards their right hand (down and turned) indicating they want to navigate more specific actions then the device captures the right camera and runs MediaPipe AI to detect the hand gesture and trigger the specific action accordingly. These actions could request AI or IoT features like Object Detection, Language Models, Map Navigation Information, or Monitoring details, here the device makes the request to the server to access these third party applications through an Android phone that provides the internet access and other details like GPS. The server side hosts to the internet through cloudflare and accepts an API or cargiver/user side access with authentication details of the Device ID and password, the interface shows the third party applications the device installed and other important details like last used or device information for caregiver monitoring.

The device itself contains the following components (TODO Compute power consumption and expected duration):
- **Raspberry PI 4** - The central microcomputer running the client side program *(5.5W active, 3W standby)*
- **MPU6050** - The IMU that measures the head orientation *(0.1W active and standby)*
- **OV5640 Cameras** - Placed 13cm apart from each other and provides the stereo images *(2W active, 0.5W standby)*
- **Stereo Earphone** - Provides the audio output generated from the client side program *(0.1W active, 0W standby)*
- **Raspberry PI Fan** - Cools the device and prevents overheating *(0.5W active, 0.2W standby)*
- **Powerbank** - A 10,000mAh powerbank that should be in the user's pockets *(10Ah$\times$3.7V$\times$0.85 efficency=31.45Wh)*
- **Cables/Wires** - Connects each components to each other
- **Case** - A 3d printed case that holds every component except for the powerbank
- **Cap Hat** - The case is glued to the hat to hold everything in place

The expected duration of the device is from 8.27 hours at standby $(\frac{31.45}{3+0.5+0.2+0.1+0}=8.27h)$ to 3.83 hours at continuous active navigation $(\frac{31.45}{5.5+2+0.5+0.1+0.1}=3.83h)$.

The software dependencies of the client side is as follows:
- **Raspberry PI OS** *(OS)* - The main operating system the client device is in
- **Python** *(Programming Language)* - The main framework the client side program is running in
- **Conda** *(Environment Manager)* - Responsible for managing the python environment
- **OpenCV2** *(Library)* - Does the stereo vision and image processing
- **SMBus** *(Library)* - Allows communication through I2C especially for the IMU
- **Numpy** *(Library)* - Helps manage process data/matrices
- **MediaPipe** *(Library/AI)** - Detects the hand gesture from the image
- **PyAudio** *(Library)* - Connects to the earphones allowing to play generated audio
- **SocketServer** *(Library)* - To host the client side device info website
- **Requests** *(Library)* - Connects to the server side AIoT server

The codebase of this client side uses a modular design where each module represents a class with their own responsibility:
- `config.py` - This is where all of the configurations of the device is stored in
- `camera.py` (from `config`) - This module connects to the stereo cameras to read the images
- `imu.py` (from `config`) - This module reads from the IMU through I2C and processes the device orientation details
- `calibrate.py` (from `camera`) - This module is ran directly to calibrate the depth map generation by generating the stereo rectification JSON `stereo-{baseline}.json`
- `depth.py` (from `camera`) - This module generates the depth map by using the `stereo-{baseline}.json`
- `radar.py` (from `depth` and `imu`) - This module maps the depth map to the radar map
- `audio.py` (from `radar`) - This module sonifies the radar details as audio output to the earphones
- `hand.py` (from `camera`) - This module detects the hand guesture of the right camera through MediaPipe AI, and triggers their associated action
- `main.py` (from `hand` and `audio`) - This module hosts the device info website and switches between the depth map sonification processes `audio` and the modular application calling `hand` based on the reading of the IMU
- `rrl_kchr2023.py` (from `camera`) - This is the algorithm based on (Louis Commere, and Jean Rouat 2023) to be ran on State A as a comparision to the research's algorithm.

The device connects to the AIoT server through an Android phone that hosts the hotspot, providing the internet, GPS, etc. The software dependencies of the android phone are as follows:
- **Kotlin** - Programming language to program the app logic/server.
- **Google Mobile Services (GMS)** - Provides APIs for the app to forward to the device like `play-services-location` for GPS location.

The server side uses the following software dependencies:
- **NodeJS** - The main framework of the server side process
- **TypeScript** - Allows stricter type checking to prevent server side issues
- **ElectronJS** - Creates an application compiled format, and allows advance API/Automation
- **Ollama** - Connects with Langauge Models for advance intellegence requests
- **Cloudflare** - Allows server to be hosted through a cloudflare tunnel through tunneling

For the softwares/tools used for development includes:
- **VSCode** - IDE for developing client-side and server-side codebases
- **Android Studio** - The software used to develop the android application
- **SolidWorks** - CAD software to develop the 3d case model
- **Github** - Manages the entire codebase, versions, and distribution

## Stereo Rectification Calibration

To calibrate the stereo vision, 3 baseline was tested 8.75cm (minimum), 10cm, and 13cm (maximum). For image training/calibration, 22 images where provided where 9 of them are the sides top/middle/bottom and left/middle/right, for both close and far (so total 18). Four other images angled top, bottom, left, and right were added totaling 22 images. The statistics like Root Mean Square (RMS) pixel error and minimum/maximum detectable range are:

| Baseline (cm) | Calculated (cm) | RMS Both (px) | Minimum (cm) | Maximum (cm) | FOV Left | FOV Right | RMS Left (px) | MS Right (px) |
| - | - | - | - | - | - | - | - | - |
| 8.75 | 8.7861 | 1.7393 | 7 | 90 | 112 | 112.88 | 1.3914 | 1.3963 |
| 10 | 10.66 | 1.5494 | 15 | 100 | 113.19 | 113.32 | 1.2314 | 1.2314 |
| 13 | 13.0646 | 1.4055 | 24 | 130 | 112.52 | 111 | 1.0852 | 1.181 |

Since the equation for depth $Z=\frac{f\cdot b}{d}$ shows that depth is linear if the focal length $(f)$ and baseline $(b)$ are constant, the minimum and maximum depth for a specific baseline can be represented as $Z=Ab+C$. Through assuming linear trend the minimum depth is $Z_{min}=4b-26.5$ while the maximum depth is $Z_{max}=9.3b+8.5$. Since our target range is for close quarter navigation (rooms, corridors, buildings, etc.) 130cm max depth detection is perfect for range detection, even for close outdoor navigation.

### Benchmarking & Testing

The study employs a **Within-Subject** design with 20 blindfolded participants comparing **State A** (Continuous Stream) vs. **State B** (Stesis Selective Filtering) on a controlled obstacle course (Our school classrooms and corridors). As for the modular system, the user test if they could successfully call a mockup call through hand gesture.

**Comparitive/Baseline Methods (State A)**
Continuous streaming will implement existing visual sensory substitution to audio algorithms to provide a better comparision to the Stesis Selective Filtering algorithms. The algorithm is based on [Louis Commere, and Jean Rouat (2023)](http://arxiv.org/pdf/2304.05462) which uses the beep-repetition rate has it remains the gold standard baseline for non-verbal spatial warning systems.

**System Performance Metrics:**

* **Latency:** Time from capture to audio output (Target: <100ms).
* **Accuracy:** Reliability of Head (MPU6050) and Hand (MediaPipe) gesture recognition.

**User Performance Metrics:**

* **Navigation:** Time to Completion (TTC), Obstacle Hits (OH), and Veering (lateral deviation).
* **Cognitive Load:** NASA-TLX (Task Load Index) survey to quantify mental fatigue.
* **Intuitiveness:** Wilcoxon Signed-Rank Test on user feedback.

**Statistical Analysis:**
A Paired T-Test will be used for TTC and OH. If the null hypothesis is rejected, indicating that gesture-based filtering significantly improves performance.

### Implications and Future Work
Beyond physical navigation, this too might have significant implications for digital environments, including social media interfaces, text recognition, facial recognition, and object identification. Successful API integration via the call gesture also underscores extensibility to a range of applications tailored to specific user needs.
While optimization of the hardware has not been a priority, the idea can be applied on hardware platforms already in place, like smart glasses with stereo cameras and LiDAR for improved accuracy and speed of depth perception. Future versions may add further refinement with eye-tracking that would provide hands-free control or extend sensory output to tactile to minimize the sensory fatigue indicated in test 6, expanding accessibility for those with restricted motor capability.
This research focused on validating the controllability hypothesis rather than hardware optimization. The prototype assumes basic motor abilities for head node and gesture input, addresses only visual-to-audio substitution, and represents one component of comprehensive inequality reduction requiring complementary policy and medical interventions.
Future directions include incorporating LiDAR for improved depth estimation in the stereo vision algorithm, implementing alternative control modalities such as eye-tracking or voice commands, exploring hybrid audio-tactile feedback to reduce fatigue, and conducting longitudinal studies with visually impaired participants to assess real-world usability and impact.

### Design Backstory & Rationale

* **The Hardware Pivot:** The prototype utilizes a **Raspberry Pi 4** rather than the smaller Orange Pi Zero 2W. While the Orange Pi offered better portability, it presented critical stability issues (WLAN failures on Debian, I2C kernel conflicts) that hindered development. The RPi 4 provides the architectural stability necessary to validate the *software hypothesis* (selective filtering), which takes precedence over miniaturization in this phase.
* **Why a Hat?** Distributing hardware weight across the cranium avoids the nose/ear fatigue common with smart glasses. Additionally, the stereo camera spacing on the brim mimics the natural separation of human eyes, aiding the brain in depth perception and sonification.
* **Nomenclature:** "Stesis" is a triple-entendre: **Ste**reo-based **S**onification **I**nterface **S**ystem, while the suffix "-sis" nods to its dual nature as a Thesis and a Hypothesis.