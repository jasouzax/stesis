## Thesis Summary

### Identity

* **Project Name:** Stesis (Stereo-based Sonification Interface System)
* **Thesis Title:** World Navigation Hat – Development of a Wearable Navigation Aid using AIoT for the Visually Impaired

### Problem

Despite advancements in assistive technology, the visually impaired continue to face significant barriers in physical navigation and digital accessibility (SDG 10). Existing Visual Sensory Substitution Devices (VSSDs) often fail due to **sensory overload**. By continuously flooding the user with audio data, these devices prevent "subconscious integration"—the brain's ability to process signals intuitively—leading to cognitive fatigue and device abandonment.

### Solution

**Stesis** is a wearable "smart hat" powered by a **Raspberry Pi 4** (tethered to a pocketed power bank). It utilizes a dual-mode interface to balance information flow:

* **Physical Mode (Navigation):** Uses stereo-rectified depth mapping (OpenCV) to convert the 3D environment into binaural "radar-like" audio. This processing is local (non-AI), ensuring low latency (<100ms) for real-time safety.
* **Digital Mode (Interaction):** Uses the right camera to detect hand gestures (MediaPipe), enabling control of external applications and LLM interactions.
* **Selective Filtering:** Integrated MPU6050 head-tracking logic allows users to "toggle" data streams or request specific depth scans via head gestures. This mimics natural vision (looking where one wants to see), significantly reducing the constant noise inherent in traditional VSSDs.

### Hypothesis

Prioritizing **selective filtering** via intentional head gestures will minimize sensory overload and fatigue compared to continuous audio streaming, thereby increasing navigation efficiency and accelerating the user's ability to subconsciously integrate the device's feedback.

### Related Studies

* **Eye-Tracking Sonification (2025):** Filters VR environments using eye-tracking. *Limitation:* Not tested on real-world visual data; relies on eye movement, which may not be functional for all visually impaired users.
* **Sound of Vision (2018):** Generates continuous 3D audio-haptic maps. *Limitation:* Suffers from sensory overload affecting long-term adoption. Stesis addresses this via gesture-based filtering.
* **Eye Cane (2015):** Uses infrared for single-point distance sound. *Limitation:* Lacks the spatial context and "scene outline" Stesis provides.

### Theoretical Basis

Stesis mimics human visual hierarchy: the brain perceives spatial geometry (outlines) before details. To bridge the gap between high-bandwidth vision and low-bandwidth hearing, Stesis utilizes a **Time-Frequency-Intensity** mapping schema:

1. **Horizontal Position  Temporal Sweep (Time):** The audio pans across the stereo field (Left  Right), allowing the user to visualize obstacle location based on *when* and *where* the sound occurs during the sweep.
2. **Vertical Position  Frequency (Pitch):** Relative to the floor (Reference Plane):
* *High Pitch:* Head level (Duck).
* *Mid Pitch:* Torso level (Collision risk).
* *Low Pitch:* Drop-offs/Stairs (Step down).


3. **Distance  Volume (Intensity):** Louder audio indicates proximity. Fast-approaching objects trigger a distinct "stereo entity" overlay for immediate collision warnings.

### Benchmarking & Testing

The study employs a **Within-Subject** design with 20 blindfolded participants comparing **State A** (Continuous Stream) vs. **State B** (Stesis Selective Filtering) on a controlled obstacle course.

**System Performance Metrics:**

* **Latency:** Time from capture to audio output (Target: <100ms).
* **Accuracy:** Reliability of Head (MPU6050) and Hand (MediaPipe) gesture recognition.

**User Performance Metrics:**

* **Navigation:** Time to Completion (TTC), Obstacle Hits (OH), and Veering (lateral deviation).
* **Cognitive Load:** NASA-TLX (Task Load Index) survey to quantify mental fatigue.
* **Intuitiveness:** Wilcoxon Signed-Rank Test on user feedback.

**Statistical Analysis:**
A Paired T-Test will be used for TTC and OH. If , the null hypothesis is rejected, indicating that gesture-based filtering significantly improves performance.

### Design Backstory & Rationale

* **The Hardware Pivot:** The prototype utilizes a **Raspberry Pi 4** rather than the smaller Orange Pi Zero 2W. While the Orange Pi offered better portability, it presented critical stability issues (WLAN failures on Debian, I2C kernel conflicts) that hindered development. The RPi 4 provides the architectural stability necessary to validate the *software hypothesis* (selective filtering), which takes precedence over miniaturization in this phase.
* **Why a Hat?** Distributing hardware weight across the cranium avoids the nose/ear fatigue common with smart glasses. Additionally, the stereo camera spacing on the brim mimics the natural separation of human eyes, aiding the brain in depth perception and sonification.
* **Nomenclature:** "Stesis" is a triple-entendre: **Ste**reo-based **S**onification **I**nterface **S**ystem, while the suffix "-sis" nods to its dual nature as a Thesis and a Hypothesis.