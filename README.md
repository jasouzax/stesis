# Thesis Summary

- [Project Identity](#identity)
- [Project Scopes](#scopes)
- [Problem Statement](#problem)
- [Solution](#solution)
- [Hypothesis](#hypothesis)
- [Objectives](#objectives)
- [Significance of the Study](#significance-of-the-study-rps) *(RPS)*
- [Conceptual Framework](#conceptual-framework)


## Identity

- **Project Name:** Stesis (Stereo-based Sonification Interface System)
- **Thesis Title:** World Navigation Hat – Development of a Wearable Navigation Aid using AIoT for the Visually Impaired

## Scopes

- **IEEE** - Minimizing sensory limitations through **radar-like selective filtering** algorithm, testing includes the efficency of the algorithm to achieve the task.
- **RPS** - Prioritizes in AIoT aspects by testing the AI and IoT capabilities and the user's ability to access them how it integrates to their digital/physical life.

## Problem

Despite advancements in assistive technology, the visually impaired continue to face significant barriers in physical navigation and digital accessibility (SDG 10).
> Sensory Substitution Devices research has two primary pillars of "Valley of death" like Sensory limitation which includes the neuroplastic adaptation and bandwidth bottleneck like sensory fatigue and overload, and Generalization Problem where SSD fails in different environments because it specializes in specific environmental conditions which real life often is not. These two problems are split between the two papers for IEEE and RPS
- **IEEE** - Existing Visual Sensory Substitution Devices (VSSDs) often fail due to **sensory overload** as continuously flooding the user with audio data, these devices prevent "subconscious integration" to the brain's ability to process signals intuitively, leading to cognitive fatigue and device abandonment.
- **RPS** - Existing Visual Sensory Substitution Devices (VSSDs) often fail due to **integration** issues as varied visual imparities have varied needs and wants, so often they are only useful for lab or specific conditions and fail at being adaptable to real-life scenarios. 

## Solution

**Stesis** is a wearable "smart hat" powered by a **Raspberry Pi 4** (tethered to a pocketed power bank). It utilizes a dual-mode interface to balance information flow:

- **Physical Mode (Navigation):** *(IEEE)* Uses stereo-rectified depth mapping (OpenCV) to convert the 3D environment into binaural "radar-like" audio. This processing is local (non-AI), ensuring low latency (<100ms) for real-time safety.
- **Digital Mode (Interaction):** *(RPS)* Uses the right camera to detect hand gestures (MediaPipe), enabling control of external applications like for AI and IoT features.
- **Selective Filtering:** Integrated MPU6050 head-tracking logic allows users to "toggle" data streams or request specific depth scans via head gestures. This mimics natural vision (looking where one wants to see), significantly reducing the constant noise inherent in traditional VSSDs.

## Hypothesis

- **IEEE** - Prioritizing **selective filtering** via intentional head gestures will minimize sensory overload and fatigue compared to continuous audio streaming, thereby increasing navigation efficiency and accelerating the user's ability to subconsciously integrate the device's feedback.
- **RPS** - Prioritizing **modular system** design controllable through hand gesture and IoT server would help make substitution devies more personalized to the needs of the visually impaired thereby supporting widespread adaptability and implementation of the device. 

## Objectives

- **IEEE** - To develop an AIoT visual sensory substitution hat that minimizes sensory overload by allowing users to selectively filter environmental sonification.
    - Develop a dual-mode prototype powered by a Raspberry Pi 4 that generates local depth maps via OpenCV stereo vision (Physical Mode) and integrates MediaPipe AI hand gesture recognition (Digital Mode).
    - Design a 3D-printed hat mount optimizing weight distribution, powered by a 10,000mAh bank, with modular NodeJS/ElectronJS server for AIoT integration.
    - Evaluate system and user performance using a within-subject design to test the impact of selective filtering on cognitive load and navigation efficiency against continuous streaming baselines.

- **RPS** - To develop an AIoT visual sensory substitution hat that allows applications to be managed in the device with AI and IoT features to allow personalized device to accomidate with Visual Impared People's varied conditions.
    - To build a prototype system that includes:
        - A modular system that allows the development of third party applications for custom modules like AI services, and Server/IoT actions using NodeJS.
        - A toggleable cue mode by hand gesture commands using MediaPipe and using the Inertial Measurement Unit (IMU) to adjust perception focus, to prioritize specific spatial information into audio cues for navigation or to configure external services with the users’ preferences.
        - A stereo rectified sonification system using dual OV5640 stereo cameras and Inertial Measurement Unit (IMU) to convert environmental information through depth maps enabling real-time 3D reconstruction of the environment into audio cues for navigation.
    - To design a portable device that includes:
        - 3D-printed addition on a cap hat to distribute weight evenly and prevent strain on sensitive facial areas like eyes, nose, and ears.
        - Connection to a powerbank placed on the user's pockets to minimize the weight on the head but still power the system
    - To test the functionality of the device system in terms of:
        - Ability to manage applications in the IoT server and its effect in the device
        - Evalulating the Precision, Recall, and Accuracy of the two pre-installed applications of the Toggleable cue interface and the stereo sonification system
        - Measuring the performance of the device running the system

## Significance of the Study *(RPS)*
- **Visually Impaired and Blind Individuals.** They are the primary beneficiaries, gaining increased mobility, safety, and access to digital and social spaces through sensory substitution and IoT assistance.
- **Families and Caregivers.** With the device promoting user independence, caregivers will experience reduced physical and emotional strain while maintaining peace of mind about the user's safety.
- **Medical and Rehabilitation Specialists.** Eye health professionals and rehabilitation centers can use the system as a tool for sensory training and mobility rehabilitation programs, especially to fill in aspects the patient lacks or has trouble with.
- **Researchers and Engineers in Assistive Technology.** The project offers a novel framework that combines virtual world modeling, IoT, and sensory emulation, providing a foundation for future innovations in human-computer interaction and embedded systems. The project can inspire further studies on brain-inspired computing and wearable assistive devices, paving the way for more innovative tools.
- **Government and NGOs for Disability Support.** Organizations involved in disability welfare can adopt or fund similar low-cost solutions to support national accessibility programs and meet inclusive development goals.

## Conceptual Framework 
The IPO of the system is *(Figure 1)*
- **Input**
    - **Hardware**
        - **Raspberry PI 4** - The main microcontroller
        - **10,000mAh Powerbank** - Powers the whole system
        - **Camera (OV5640 5MP/120FOV)** - Captures stereo images in front of the user
        - **Binaural Earphone** - Outputs the binaural audio for user to see through sound
        - **IMU (MPU6050)** - Detects head orientation for interface through head movement
        - **Raspberry PI Cooling Fan** - Responsible for cooling the raspberry PI
        - **3D printed enclosure** - Holds most of the components and sticks to the hat
        - **Hat Cap** - The hat placed on top of the user and holds the enclosure
    - **Software**
        - **VSCode** - Main IDE used in the system's development
        - **Arduino Studio** - Development of the android app for APIs passthough
        - **SolidWorks** - Creates the 3D model of the 3D printed enclosure
        - **GitHub** - Manages the project codebase and files
    - **Connectivity Protocols**
        - **WiFi** - This allows the device to connect to the IoT server, as a hotspot by the phone to provide internet and phone sensor data through the android app
        - **I2C** - Connection to read gyroscoping and accelerometer data from the IMU
- **Process**
    1. Problem Statement
    2. Literature Review (RRL)
    3. Prototype
        1. Requirements
        2. Designing
        3. Development
        4. Refinement
        5. Implementation
        6. Testing
    4. Data Recording
    5. Data Analysis *(IEEE)*
        1. Latency from capture to output
        2. Accuracy of IMU and AI
        3. Stereo Calibration Scopes
        4. Navigation Metrics
        5. Cognitive Load (NASA-TLX)
        6. Intuitiveness (Wilcoxon Signed-Rank Test)
    6. Data Analysis *(RPS)*
        1. Device's performance (weight, latency, etc.)
        2. Server's performance (latency, security, etc.)
        3. Gesture accuracy (head and hand gestures)
        4. Navigation ease of use survey
- **Output** - Development of the wearable navigation aid using AIoT for the visually impared named Stesis
- **Feedback** - Feedback from the output to redesign back to the input as to improve the system



## Related Studies

* [**Louis Commere, and Jean Rouat (2023):**](https://arxiv.org/pdf/2304.05462) The most newest visual sensory subtitution device that both uses depth and sonification, being the most closest to our study and will be used as the baseline to test the null hypotesis. *Limitation:* It provides only localized, single-target directional information, severely restricting global scene comprehension. Users cannot perceive a general room outline simultaneously; instead, they are forced into tedious, sequential "Roomba-like" scanning and must rely heavily on working memory to construct a complete spatial map.
* [**Eye-Tracking Sonification (2025):**](https://doi.org/10.3389/frvir.2025.1598776) Filters VR environments using eye-tracking. *Limitation:* Not tested on real-world visual data; relies on eye movement, which may not be functional for all visually impaired users.
* [**Sound of Vision (2018):**](https://doi.org/10.1097/OPX.0000000000001284) Generates continuous 3D audio-haptic maps. *Limitation:* Suffers from sensory overload affecting long-term adoption. Stesis addresses this via gesture-based filtering.
* [**Eye Cane (2015):**](https://doi.org/10.3233/RNN-130351) Uses infrared for single-point distance sound. *Limitation:* Lacks the spatial context and "scene outline" Stesis provides.

## Theoretical Basis

- The hypothesis of the study is heavily based on a theoretical phenomenon called **Sensorimotor Contingencies** where allowing direct control on the substitution input should shift the cognitive load from conscious decoding to intuitive subconscious spatial awareness.
- The process of substituting visual to audio means representing 2D data in 1D, so the algorithm utilizes a **radar-like** mapping schema representing the Y-axis as audio and the X-axis as time.
- Human visual perception in **Scene Gist and Gestalt Organization** perceives spatial geometry (outlines) before details/textures so the algorithm generates depth map to represent the spatial geometry through stereo rectification to allow real time processing when compared to an AI model like MiDaS.
- To represent the outlines, only the closest proximity of that time slice is represented in audio where the frequency is inversely proportional to the proximity (closer the object the higher the frequency) and to ensure it doesnt effect the sense of volume then the **Equal Loudness Compensation** is applied
- The volume of the audio output is split between left and right channels through **Head-Related Transfer Function** to create a sense in the 3D space.
- The rate of the radar scan is proportional to the angle of depression of the device, this allows the rate of output to slowed down if needed for the user to digest the information and to skip or focus on specific areas mimicing **Focus and Spatial Resolution**
- While the radar creates this entity that goes back and forth based on the rate of radar scan, it overtime losses it total volume to reduce redundant information while prioritizing movement as represented as new audio entities as visual **Selective Attension and Saliency**

## System Design

The device first runs the main service which connects to a hotspot or wifi so that it can use server APIs and the internet, then activating the device application running the installed applications. There are two pre-installed applications:
- **Hand Gesture** - This application is activated if the IMU detects that the user is facing to their right hand *(down and rolled/turned facing the right side)*, it captures the right camera and detects the hand guesture through MediaPipe AI. If the hand guesture matches a specific hand guesture then it activates its corresponding action.
- **Stereo Sonification** - This application is activated if the IMU detects that the user is facing downwards indicating they want to see their obstacles to navigate. It captures the stereo images from the two cameras where stereo rectification is applied to generate the depth map, it is then processed to generate a radar map of the environment and then sonified to represent that data as audio to the binaural earphones.
The device can make a request to the server to access these third party applications through an Android phone that provides the internet access and other details like GPS or Compass. The server side hosts to the internet through cloudflare and accepts an API or cargiver/user side access with authentication details of the Device ID and password, the interface shows the third party applications the device installed and other important details like last used or device information for caregiver monitoring.

The device itself contains the following components:
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
- **SocketServer** *(Library)* - To host the client side device info website
- **Requests** *(Library)* - Connects to the server side AIoT server

The pre-installed applications uses these dependencies:
- **OpenCV2** *(Library)* - Does the stereo vision and image processing
- **SMBus** *(Library)* - Allows communication through I2C especially for the IMU
- **Numpy** *(Library)* - Helps manage process data/matrices
- **MediaPipe** *(Library/AI)** - Detects the hand gesture from the image
- **PyAudio** *(Library)* - Connects to the earphones allowing to play generated audio

The codebase of this client side uses a modular design where each module represents a class with their own responsibility, this makes applications their own class with their dependencies being parent classes allowing easy tracking of application dependencies. The pre-installed modules are:
- `main.py` (loads `hand` and `audio` because their applications) - This module hosts the device info website and switches between the depth map sonification processes `audio` and the modular application calling `hand` based on the reading of the IMU
- `config.py` - This is where all of the configurations of the device is stored in
- `camera.py` (from `config`) - This module connects to the stereo cameras to read the images
- `imu.py` (from `config`) - This module reads from the IMU through I2C and processes the device orientation details
- `calibrate.py` (from `camera`) - This module is ran directly to calibrate the depth map generation by generating the stereo rectification JSON `stereo-{baseline}.json`
- `depth.py` (from `camera`) - This module generates the depth map by using the `stereo-{baseline}.json`
- `radar.py` (from `depth` and `imu`) - This module maps the depth map to the radar map
- `audio.py` (from `radar`) *(Main Application)* - This module sonifies the radar details as audio output to the earphones
- `hand.py` (from `camera`) *(Main Application)* - This module detects the hand guesture of the right camera through MediaPipe AI, and triggers their associated action
- `rrl_kchr2023.py` (from `camera`) *(IEEE only)* - This is the algorithm based on (Louis Commere, and Jean Rouat 2023) to be ran on State A as a comparision to the research's algorithm.

The device connects to the AIoT server through an Android phone that hosts the hotspot, providing the internet, GPS, Compass, etc. The software dependencies of the android phone are as follows:
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

## Benchmarking & Testing *(IEEE)*

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

## Implications and Future Work
Beyond physical navigation, this too might have significant implications for digital environments, including social media interfaces, text recognition, facial recognition, and object identification. Successful API integration via the call gesture also underscores extensibility to a range of applications tailored to specific user needs.
While optimization of the hardware has not been a priority, the idea can be applied on hardware platforms already in place, like smart glasses with stereo cameras and LiDAR for improved accuracy and speed of depth perception. Future versions may add further refinement with eye-tracking that would provide hands-free control or extend sensory output to tactile to minimize the sensory fatigue indicated in test 6, expanding accessibility for those with restricted motor capability.
This research focused on validating the controllability hypothesis rather than hardware optimization. The prototype assumes basic motor abilities for head node and gesture input, addresses only visual-to-audio substitution, and represents one component of comprehensive inequality reduction requiring complementary policy and medical interventions.
Future directions include incorporating LiDAR for improved depth estimation in the stereo vision algorithm, implementing alternative control modalities such as eye-tracking or voice commands, exploring hybrid audio-tactile feedback to reduce fatigue, and conducting longitudinal studies with visually impaired participants to assess real-world usability and impact.

## Design Backstory & Rationale

* **The Hardware Pivot:** The prototype utilizes a **Raspberry Pi 4** rather than the smaller Orange Pi Zero 2W. While the Orange Pi offered better portability, it presented critical stability issues (WLAN failures on Debian, I2C kernel conflicts) that hindered development. The RPi 4 provides the architectural stability necessary to validate the *software hypothesis* (selective filtering), which takes precedence over miniaturization in this phase.
* **Why a Hat?** Distributing hardware weight across the cranium avoids the nose/ear fatigue common with smart glasses. Additionally, the stereo camera spacing on the brim mimics the natural separation of human eyes, aiding the brain in depth perception and sonification.
* **Nomenclature:** "Stesis" is a triple-entendre: **Ste**reo-based **S**onification **I**nterface **S**ystem, while the suffix "-sis" nods to its dual nature as a Thesis and a Hypothesis.