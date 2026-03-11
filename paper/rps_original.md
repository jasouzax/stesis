  
**World Navigation Hat \- Development of a Wearable Navigation Aid using AIoT for the Visually Impaired**

An Undergraduate Design Project Proposal Presented to Faculty of the Computer  
Engineering Department of College of Technology University of San Agustine

In Partial Fulfillment of the Requirement of the Course  
**CPE 413 \- CpE Practice and Design I**

**By:**  
Daywan, Vince Ginno B.  
D’Souza, Jason C.  
Pabito, Ethel Herna C.  
Wang, ChenLin

Engr. Glenda S. Guanzon  
November 2025  
**Introduction**

# **Background of the Study**

Visual impairment is a global health problem that significantly affects individual's daily lives by impeding independent navigation, social interaction, and overall quality of life (Theodorou et al., 2023). There is an estimation of 2.2 billion people globally who are identified as visually impaired and this number could still increase to 2.5 billion by 2050 as stated by the World Health Organization (World Health Organization, 2023). In the Philippines, 2.17 million Filipinos are identified as visually impaired as quantified by reports from the year 20218 from the Philippines Eye Research Institute (PERI) and Department of Health (DOH) (Shinagawa Lasik & Aesthetics, 2025). Visual impairment does not pertain to total blindness. According to World Health Organization (2023), visual impairment can be identified and categorized based on the presenting visual acuity; 

a) Mild Vision Impairment, as visual acuity is better than 6/18, 

b) Moderate Vision Impairment, as visual acuity is worse than 6/18 but better than 6/60, 

c) Severe Vision impairment, as visual acuity is worse than 6/60 but better than 3/60, 

d) Blindness, as visual acuity is worse than 3/60, types such as blindness with light perception, individuals can only perceive light, and total blindness, individuals who have no light perception. 

Other common types include astigmatism, near-sightedness (myopia) and far-sightedness (hyperopia), which account for a large share of impairments, alongside conditions like cataracts, glaucoma, and age-related macular degeneration (AMD), with AMD alone affecting 8.06 million people globally in 2021 (Zou et al., 2024).

Navigation is one of the problems visually impaired people (VIP) encounter, they often have difficulty crossing the road and tracking their location, as well as experiencing accidents, falls, and collisions (Ikram et al., 2024; Gao et al., 2025; Muhsin et al., 2023).   
In addition, VIPs experience social isolation and reduced access to information, limiting their educational and employment opportunities (Arvind, 2023). There are already existing solutions or aids for VIPs, for daily assistance, which include guide dogs, white canes, and electronic travel aids (Muhsin et al., 2023). Additionally, one of the technologies VIPs depend on for navigation is assistive devices, which use sensory substitution to convert visual information into auditory or tactile cues to provide spatial awareness (Skulimowski, 2025). These devices often incorporate IoT sensors to capture real-time environmental data, providing navigational assistance in complex environments (Real & Araújo, 2023).

Sensory substitution devices (SSDs) are assistive wearable devices that can be eyewear, a vest, or a hat, allowing VIP users to perceive their surroundings by converting sensory information from one modality to another (Mishra et al., 2025). These SSDs are embedded with sensors that allow obstacle detection, navigation, and object recognition, enhancing VIPs' daily independence and reducing their reliance on traditional aids (Tokmurziyev et al., 2025). However, global adoption of these devices is limited due to cognitive overload performance, extensive training needs, ergonomic discomfort design, and the lower processing bandwidth of non-visual senses compared to vision (Hou et al., 2025).

# **Rationale**

Visual Impairment encompasses a spectrum of conditions, from moderate vision loss to complete blindness, which significantly impacts an individual's ability to perceive their environment and perform daily activities (Kumar et al., 2025). This condition also affects the individual’s social interaction and overall quality of life (Que et al., 2025). More than 2.17 million people in the Philippines live with visual impairment, which restricts their perception, mobility, navigation, and accessibility to information—depriving them of work opportunities and independence (Chavarria et al., 2025). To mitigate the challenges encountered by the VIP, assistive technologies and rehabilitation strategies have been developed (Skulimowski, 2025). Traditional assistive aids like white canes offer limited feedback, initiating the development of electronic travel aids that deliver richer environmental information and improve autonomous navigation (Chandra et al., 2025). Another solution is a wearable SSD, an assistive technology that an individual can wear on their body to help users navigate both their physical surroundings and digital applications, and emerging as a potential avenue to restore or gain a sense of spatial perception or awareness and navigational ability (Skulimowski, 2025). Additionally, SSDs promote social interaction that aligns with one of the United Nations' Sustainable Development Goals for health and equality (Xue et al., 2025).

In previous studies, SSDs have been limited in their applicability, as existing devices often perform poorly in real-world situations, encountering cognitive overload problems that overwhelm users with excessive information, making navigation difficult (Arora et al., 2024). Another identified concern is that the SSD design is uncomfortable, as it is bulky and requires extensive training, which limits its practicality (Gao et al., 2025; Muhsin et al., 2023, p. 148). In addition, many of these devices lack digital integration as capabilities, focusing mainly on obstacle detection and not connecting well with digital tools (Makati et al., 2024). These deficiencies underscore the pressing need for higher SSDs that can outperform and enhance existing devices. Therefore, a need for development of optimized smart navigation devices improves the functional efficiency and performance of SSDs without overlooking the previous concerns like overloading data processing.

In this study the researchers aim to address these matters through the development of a wearable navigation hat integrated with AIoT, which introduces a smart navigation hat that uses 3D data to give audio feedback and connects to IoT services for navigation in both physical and digital environments. Furthermore, it enhances sensory processing, focusing on reducing cognitive overload in the system and making navigation simpler for VIP users. Additionally, the hat features customizable options, including a modular operating system that allows users to choose between hand gesture or voice commands.

# **Objectives of the Study**

## **General Objective**

To develop an AIoT visual sensory substitution hat device that converts real-time environment information through emulating human sensory processes into audio cues for the VIP.

## **Specific Objectives**

1) To build a prototype system that includes:  
1) A photogrammetry (2D to 3D) system using dual OV5640 stereo cameras and Inertial Measurement Unit (IMU) to convert environmental information through depth maps enabling real-time 3D reconstruction of the environment into audio cues for navigation.  
2) A toggleable cue mode by hand gesture commands using MediaPipe and using the Inertial Measurement Unit (IMU) to adjust perception focus, to prioritize specific spatial information into audio cues for navigation or to configure external services with the users’ preferences.  
3) A modular system that allows the development of third party applications for custom modules like AI services, Server actions using NodeJS, and hand gesture recognition.

2\)	To design a portable device that includes:

1) A rechargeable 5V/3A UPS module with 18650 lithium-ion batteries for cord-free operation via Type-C charging.  
2) 3D-printed hat enclosure to distribute weight evenly and prevent strain on sensitive facial areas like eyes, nose, and ears.

3\)	To test the functionality of the device system in terms of:

1) Calculating the Error and Success Percentage performance of photogrammetry by comparing the generated depth maps to ground truth measurements.  
2) Evaluating the Precision, Recall, and Accuracy of the toggleable cue mode hand gesture commands responsiveness to switching modes.  
3) Measuring the frame per second (FPS) of AI processing, error rates in recognition, and route accuracy of the modular system.  
4) Simulating full system functionality by combining photogrammetry, IMU, and modules to track overall FPS, audio cue fidelity, and failure cases, with qualitative feedback from the users using surveys.

# **Significance of the Study**

This project device will specifically benefit to the following:  
**Visually Impaired and Blind Individuals.** They are the primary beneficiaries, gaining increased mobility, safety, and access to digital and social spaces through sensory substitution and IoT assistance.  
**Families and Caregivers.** With the device promoting user independence, caregivers will experience reduced physical and emotional strain while maintaining peace of mind about the user's safety.  
**Medical and Rehabilitation Specialists.** Eye health professionals and rehabilitation centers can use the system as a tool for sensory training and mobility rehabilitation programs, especially to fill in aspects the patient lacks or has trouble with.  
**Researchers and Engineers in Assistive Technology.** The project offers a novel framework that combines virtual world modeling, IoT, and sensory emulation, providing a foundation for future innovations in human-computer interaction and embedded systems. The project can inspire further studies on brain-inspired computing and wearable assistive devices, paving the way for more innovative tools.  
**Government and NGOs for Disability Support.** Organizations involved in disability welfare can adopt or fund similar low-cost solutions to support national accessibility programs and meet inclusive development goals.

# **Conceptual Framework**

[**Figure 1**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_ipo)  
*IPO Model of the Conceptual Frame of the Study*

Figure 1 illustrates the study's conceptual framework as an Input-Process-Output (IPO) model. Starting with three kinds of required input, such as the hardware components Microcontroller, Sensors, Actuators, SIM Module, and Wireless Connectivity, used on the device, which can be either 2/3G used for deployment doors, or Wi-Fi as used in deployment indoors or development. In addition to the input, the software components include Project management, Client-side device, and Server-side device. After the input, it will proceed to the process, which includes the following: 1\) Problem Statement, 2\) Literature Review, 3\) Prototyping, which includes Materials Requirements, Device Design, Development, Refinement, and Implementation, 4\) Data Recording, 5\) Data Analysis of the findings using P-test on the Device Performance in terms of accuracy in recognizing user controls (hand gestures) and generating relevant audio from environmental inputs; Sensory Overload on user accuracy in interpreting generated audio cues (e.g., via task completion rates) and qualitative feedback on perceived overload on user; Sensory Fatigue on usage duration before symptoms like headaches emerge (e.g., via self-reported scales, physiological monitoring, or user surveys); Sensory Integration in tracking reaction times to environmental data, learning curves over sessions, and survey responses on ease of subconscious adaptation. The research process mimics typical development cycles, including literature review, prototyping, designing, testing/evaluation, and implementation. By the end of each iteration, the researchers expect to be closer to the end goal, with user and professional feedback to start the cycle again until the end goal of a wearable navigation aid is achieved.

# **Theoretical Framework**

This research attempts to solve an issue facing most Visual Sensory Substitution Devices which is primarily about the issue on translating a high bandwidth visual information to low bandwidth audio information without causing overstimulation, our novel solution is the emulate the human sensory mechanism to carry the processing load from the user and ensure that substituted information is minimal and necessary to the user. The theoretical principles include:

**Focus and Spatial Resolution (Foveation)**   
The human vision has a tiny high-resolution fovea covering around 1 to 2 degrees of the visual field but accounts for around half of visual cortex (Himmelberg et al., 2023), outside this region is a much coarser peripheral vision where acuity drops rapidly (Aydin et al., 2024). In this study, the device attempts to mimic this by using a focus point system where the user can contain the focus radius indicating that area of the environment the user wants to be translated into audio and at what detail.

**Selective Attention and Saliency**   
As the visual system cannot process all details at once, it selectively attends to salient or task-relevant features, this is done by implementing a bottom-up saliency and top-down goals to filter out redundant/irrelevant information (Arora et al., 2025). This demonstrates that people focus on key objects ignoring uniform backgrounds as the nervous system "tunes out" repeated stimuli and amplifies novel/focused ones (Gershman, 2024). The device translates environmental information where any stagnating/constant information must be tuned out over time leaving behind changes indicating motion.

**Scene Gist and Gestalt Organization**

Human vision rapidly extracts the gist (background) and figures (foreground) from a scene within the first fixation (\~36ms) with around 80% accuracy (Loschky, 2025). This process is done through Gestalt principles that groups elements to simplify complex images such as by similarity, proximity, common region, continuity, etc. (User Testing, 2024). This indicates that the device should be able to generally/primitively separate the environmental data into background and foreground categories where foreground can then be separated into groups, this is information to indicate the translated audio.

**Parallel Motion vs. Detail Pathways**   
Human visual systems process motion and details in two separate channels, the magnocellular pathway for fast motion and size but in less detail, and the parvocellular pathway for static objects in more detail (Al-Najjar et al., 2022). Suggests that when the device switches to motion detection it should prioritize the speed, size, and direction of that motion.

**Temporal Dynamics and Scanning**  
Vision is not a singular static snapshot but rather a continuous sample of the world via eye movements has humans typically shift gaze 3-4 times a second (Nghiem et al., 2025). Thus, in this study, the device should rapidly provide updating audio to allow temporal integration rather than one large static soundscape all at once, this also solves the issue of cognitive overload and sensory fatigue.

**Multisensory Integration**   
The brain integrates additional senses like auditory, vestibular, and proprioceptive senses to form a coherent representation of the environment (Multisensory Integration, 2025). Thus, the device should align with those senses to prevent conflicting senses that could often cause nausea. This could be in the form of aligning the virtual environment with IMU to align with vestibular senses, or lower the output audio when the user is focusing on something else like talking to others.

**Scope and Delimitation**  
This research study focuses on Sensory Substitution Devices, more specifically on Vision Substitution. The device aims to fill in what different variety of visually impaired individuals lacks, like if fully blind the device should be able to act as an “artificial eye” providing information the person wants, or if the person is near sighted it could read text from afar from the user and warn of objects rapidly approaching the user, and so on. This is done by having a modular system running the main pipeline, the main pipeline inputs images from the stereo camera, depth map it to a virtual-physical map, follows an algorithm to emulate human vision processes, and finally generates the necessary audio cues for the individual. This pipeline allows applications to take into account more varied needs such as hand gesture recognition to control the parameters of the main pipeline so the person can navigate the device without seeing anything, image recognition to read text near or far, face recognition to recognize friends and family, and even more personalized interest like keeping track of location. In the testing, the device will only prioritize the functionality and feasibility of the system to determine if the designed and developed project can perform efficiently. Mock API will be utilized to demonstrate the potential functionality of external services of the system. 

# **Review of Related Literature**

Zvorișteanu et al. (2021) presented the Sound of Vision (SoV) system. It is a wearable SSD designed to aid VIPs in spatial cognition and navigation, using stereo-vision-based 3D reconstruction to interpret the environment and translate spatial data into audio and haptic cues. The device integrates a stereo camera, an infrared depth sensor, and an inertial measurement unit (IMU) to track the motion of the user’s head and the installed camera mounted on the headgear, using these as its inputs. Furthermore, the output audio and haptic feedback are delivered via headphones and a haptic belt. The device processes visual data in real time using GPU-accelerated computation to reconstruct the 3D Environment and identify obstacles via segmentation algorithms based on disparity and histogram analysis. The depth of each pixel is determined by the stereo disparity formula Z=f⋅Bd, where Z is the depth, B is the baseline, f is the focal length, and d is the disparity. The SoV converts environmental data into audio-haptic cues, allowing users to detect obstacles, open spaces, and open areas. In testing the device's performance with the visually impaired participants, it showed a 10% improvement in depth accuracy and an 88.5% task success rate compared to standard stereo vision methods. The results highlight the device's effectiveness in real-world navigation and its potential to enhance mobility and spatial awareness compared to traditional aids such as white canes.

Sami et al. (2025) developed a smart wearable (hat) assistive device that integrates object detection with voice-assisted navigation to support VIP in real-time spatial awareness. The device uses an ESP32-CAM module to capture live video and transmit it via Wi-FI to a mobile application that employs a TensorFlow-based deep learning model for efficient object recognition. The camera-detected objects are then processed into audio prompts that send alerts to the user when there are nearby obstacles and provide navigation directions. The programmed system produced a seamless object-detection-to-voice pipeline. The Smart Hat's system architecture demonstrates the integration of an IoT processing method to enable wireless communication between the smart hat and the user interface for remote processing and real-time updates for the user. Testing results show an object detection accuracy of over 91%, highlighting the reliability in dynamic environments. In addition, the Smart Hat device offers compact, low-cost, and user-friendly features. It demonstrates how IoT-enabled computer vision can transform traditional assistive technologies into intelligent, voice-guided navigation tools for the visually impaired.

RealSense (2025), in collaboration with Eyesynth, developed a wearable sensory-substitution device designed to enhance spatial awareness and independent navigation for visually impaired users by converting real-time 3D spatial data into bone-conduction audio cues. The device employs the RealSense D415 depth camera to capture real-time 3D depth data calculated using the stereo disparity formula Z=f⋅Bd. Then it is processed to generate a point cloud representation of the surroundings, where each depth pixel being mapped to 3D coordinates using the equation X=u-cxzfx, Y=v-CyZfy, Z=Z to provide a complete spatial surrounding rather than a 2D view. The point-cloud data conjoint with the inertial and head motion tracking, is then processed through an onboard ASIC and Simultaneous Localization and Mapping (SLAM) modules to identify object orientation, translating them into audio signals such as pitch (height), volume (distance), and stereo panning (lateral position), allowing users to perceive objects and obstacles up to 5 meters away without external infrastructure. In testing NIIRA, it demonstrated improved independence and navigation in both indoor and outdoor environments, adapting to users’ preferences, and displayed high obstacle-detection accuracy, minimal processing latency, and enhanced independent navigation. The participants reported that the bone-conduction audio feedback provided a clear environmental awareness without blocking external sounds.

Udayakumar et al. (2025) designed a Smart Vision Glasses (SVG) that processes environmental input data using AI, LiDAR depth mapping, and computer vision algorithms. The device’s front-facing camera captures visual data, while the LiDAR sensor measures real-time depth information using the principle of Time-of-Flight (ToF) – where the distance (D) is calculated using the formula D=c⋅t2, where c represents the speed of light and t the time taken for emitted light to return after reflection. These depth and image data are then processed through an AI-based recognition model that uses Convolutional Neural Networks (CNNs) to classify objects, text, and faces within the user's field of view. The recognized elements are subsequently analyzed and prioritized based on proximity and relevance using the LiDAR-assisted spatial mapping, which produces a semantic understanding of the scene; implementing Volume of Interest (VOI), a controllable spatial region or focus radius that limits feedback, to prevent sensory overload, enhancing user focus in dynamic environment. This processed information is then transmitted to a smartphone-based application converting the recognized data into voice audio cues through text-to-speech (TTS) engine. The device applies natural language processing (NLP) principles to ensure that the audio output is contextually meaningful and user-friendly. The input-output process follows a sequential pipeline:

1) capture of the image depth data  
2) AI-driven detection and classification  
3) distance estimation through the ToF equation  
4) conversion of results into voice audio output describing nearby objects, text, or faces. 

The SVG through hearing supports the four primary modes – “Thighs Around You”, “Reading”, (Walking Assistance), and “Face Recognition”. The SVG is also equipped with gesture and voice control interfaces, allowing users to switch modes or issue commands hands-free, enhancing accessibility and user interaction. A multicenter usability study was conducted across five rehabilitation centers in India, involving 90 participants with a mean age of 23.5 years, to test the device functionality. The participants tested the four primary modes by completing real-world navigation and recognition tasks, and their feedback on the device's usability and helpfulness was also recorded. Results showed that 72.9% indicated that "Reading” mode is helpful, 44.7% for “The Things Around You”, 36.5% for “Face Recognition”, and 22.4% for “Walking Assistance”, showing that the device effectively enhances environmental awareness, reading ability, and object identification.

Ruan et al. (2025) developed a multifaceted sensory substitution wearable device that uses an audio-based curb detection to improve real-time awareness and navigation safety for individuals who are blind or have low vision. The device integrates a stereo camera, ultrasonic range sensors, and inertial motion units (IMUs) to identify curbs and ground-level transitions Adapting the stereo disparity formula Z=f⋅Bd, the system calculates the curb height and distance, while the ultrasonic sensors confirm the surface proximity of the environment for redundancy and accuracy. The sensory data are processed and converted into audio and vibrational cues, where the pitch and repetition rate of the sound correspond to the curb’s distance and height, providing intuitive, time-sensitive feedback. In the pilot testing, 12 participants were involved, 8 with blindness and 4 with low vision. The participants tested the navigation feature of the device in both indoor and outdoor test environments, including simulated sidewalks, ramps, and descending curbs. After testing, the performance statistics of the device showed an average curb detection accuracy of 94.3%, with a false-negative rate of 3.7%, and an average alert response time of 1.2 seconds, indicating the system’s reliability in detecting and signaling curbs in real time. Participants reported that it also improved confidence in mobility and reduced the risk of missteps or trips. This demonstrates that the audio-based curb detection approach driven by LiDAR-like depth estimation and multimodal feedback effectively enhances real-time alerting and safe navigation for the VIP.

Viancy V et al. (2024) introduced AuralVision, a wearable assistive device that is designed as eyewear to improve navigation for visually impaired individuals by incorporating object detection, scene classification, and reinforcement learning. The device's system architecture includes a camera sensor to capture the user’s environment, image-processing and object-detection software to classify obstacles and identify significant objects, and an audio module to translate detection results into auditory cues for navigation. Utilizing a convolution neural network (CNN) approach or deep learning-based vision algorithms trained on the Object Net 3D dataset to interpret the surrounding environment and convert the visual information to auditory cues that convey spatial awareness. AuralVision uses 3D visualization and sound-based feedback to identify objects, classify road scenes, and assist users with obstacle avoidance and pathfinding. The device adapts to dynamic environments through continuous learning, thereby improving navigation decision-making.

Hamilton-Fletcher et al. (2021) developed a mobile sensory-substitution device called SoundSight. This device translates input color, depth, and temperature data into output audio cues, allowing the visually impaired users to sense their environment. The system captures environmental input using a mobile with an RGB-D camera and a thermal sensor. Then it processes the data stream through a multi-feature mapping pipeline before integrating it into a composite audio stream. Depth is calculated using the stereo disparity formula to determine the distance of the object from the user's perspective. At the same time, the RGB camera's color hue values are mapped to sound pitch using the equation fc=kc×H, where H represents the hue intensity and scaling constant. Temperature measurements are converted to amplitude modulation through At=kt×T, where T is the sensed temperature and kt is a sensitivity coefficient. These parallel mappings are integrated using the data fusion algorithms, synchronizing depth, color, and thermal inputs before generating the composite audio streams that encode environmental depth through rhythm, color through pitch, and temperature through loudness. In testing the device's functionality, 15 blind and 10 low vision participants are involved. The device demonstrated rapid user learning and over 85% recognition accuracy, demonstrating that the multi-feature sonification can reliably convey complex environmental information.

Han et al. (2025) designed a Multi-Path Sensory Substitution Device (MSSD) that enhances low-vision mobility and virtual navigation of the visually impaired people by integrating real-time depth mapping, IMU-based motion tracking, and depth optimization algorithms. The system processes input from a depth camera and inertial sensors to create a 3D spatial model of the environment, using the stereo disparity equation to calculate object distance and the projection model X=u-cxzfx, Y=v-cyzfy, to map each pixel to world spatial coordinates. After acquiring the data from the processed input, it generates a 3D point cloud, which then is converted into an occupancy grid for spatial awareness, while the IMU readings refine user orientation using the equation t=t-1+ωtΔt. A modified A\* (A-star) algorithm is used to compute the optimal navigation routes using the cost function fn=gn+hn, balancing actual distance and heuristic estimates to avoid obstacles. The selected virtual path is then translated into audio-haptic feedback, where the vibration intensity and sound frequency indicate direction and proximity, using speakers or headphones to create output data directional audio cues and vibration motors/haptic actuators for tactile feedback. To check the device's functionality, the researchers tested it with 20 low-vision participants. The system attained a result of 91% path following accuracy and a 38% reduction in navigation errors. As the device demonstrates a satisfactory functional performance, this indicates that combining 3D mapping, motion sensing, and heuristic path planning effectively supports real-time, safe, and virtual navigation for the visually impaired individuals.

Commère & Rouat (2023) evaluated five depth-to-sound sonification methods to compare their performance and effectiveness. One of the methods evaluated is the LiDAR-to-repetition-rate sonification model. The identified sonification model converts visual information into rhythmic auditory cues, helping visually impaired individuals to develop a sense of spatial awareness through sound. Using the Time-of-Flight (ToF) equation D=c⋅t2, the model measures the distance of an object and maps it inversely to the beep repetition rate R=kD, where R is the repetition rate (Hz), D is the distance (meters), and k is a scaling constant which is determined experimentally to maintain the perceptible tempo differences between near and far objects which results if an object is closer it generates faster repetition rates or beeps and far detected objects have slower rhythm beeps which the model’s spectrogram design reflects this relationship. Whereas the temporal density (number of pulses per second) indicates proximity, while frequency and amplitude encode additional object features such as height and reflection intensity, creating a dynamic “rhythmic depth map” that represents distance through time spacing between sound bursts—in experimental testing with 28 sighted but blindfolded participants, a three-phase protocol consisting of Depth Estimation, Azimuth Estimation, and Retention Assessment confirmed that repetition-rate sonification produces the lowest mean absolute error (MAE) in distance perception and was the most intuitive and accurate among the other mapping methods like frequency or amplitude-based cues. The results finding revealed a strong inverse correlation between the perceived distance and repetition rate (R∝1D), indicating that LiDAR-driven rhythmic approach effectively translates the spatial depth into comprehensive sound patterns, offering an enhanced basis for real-time auditory sensory substitution systems for the visually impaired.

Zhao et al. (2025) conducted a clinical evaluation of an assistive device, a wearable electronic navigation aid (ENA), for visually impaired individuals through a prospective, non-randomized, single-arm, open-label trial involving 30 participants (each participant was given five trials for each functional task), selected through purposive sampling to ensure that individuals with blindness or severe visual impairment could assess the performance of the assistive device. The study primarily focused on the functional efficacy of the device, assessing how effectively it supports navigation and daily tasks of the visually impaired users. The assistive device used in testing integrates ultrasonic sensors, vibration motors, and audio output components. The study implemented a structured, protocol-guided or functional evaluation framework that combined quantitative measures including accuracy rate (Successful TrailsTotal Trails×100), error rate (Navigation ErrorsTotal Trails×100), reaction time (RT=Tresponse-Tstimulus), task completion time (TCT=Tend-Tstart), and performance improvement rate (PIR=Post Test \- Pre TestPre Test×100) to assess navigation accuracy, responsiveness, and efficiency objectively, supported by qualitative user feedback on comfort, usability, and confidence. The study reveals that the ENA device effectively improved safe navigation and mobility performance for visually impaired users.

**Summary**  
Recent studies (2019–2025) on assistive technologies for the VIP highlight wearable and mobile devices utilizing a sensory substitution approach to convert visual/spatial data into audio, haptic, or multimodal cues, applying stereo vision (e.g., disparity formula Z=f⋅Bd), depth sensors like LiDAR/ToF (e.g., D=c⋅t2), AI-driven computer vision (e.g., CNNs, TensorFlow), and IMUs. This approach expresses the outcome produced by the device such as real-time obstacle detection, object recognition, and navigation, similar to the Caraiman et al.'s Sound of Vision (SoV), which addresses sensory overloading via a controllable focus radius (VOI) and modular components for partial generalizability; Sami et al.'s Smart Hat for cost-effective voice output; RealSense's NIIRA point-cloud processing to semantic audio; Commère and Rouat's LiDAR sonification for rhythmic depth cues; Hamilton-Fletcher et al.'s SoundSight for mobile timbre/volume mappings; Ruan et al.'s curb detector for 94.3% accuracy; Han and Li's Multi-Path Sensory Substitution Device (MSSD) for 91% path accuracy; Zhao et al.'s electronic navigation aid (ENA); and Udayakumar and Gopalakrishnan's Smart Vision Glasses (SVG) for AI-LiDAR voice modes. However, systems like AuralVision that utilizes ToF lasers for bone-conduction stereo sound, lab-effective but range-limited, Eyesynth’s NIIRA object recognition process prone to overload without human emulation, Depth Sonification repetition rates for intuitiveness, SoundSight LiDAR/thermal to audio which is app-constrained, Smart Hat that is non-modular, and multi-faceted SSDs with 85% curb accuracy often face issues like sensory overload, training demands, limited real-world adaptability, and lack of digital-physical navigation. These studies proposes an idea that work closely aligns with SoV by tackling sensory overloading through VOI and modular components but advances it by explicitly mimicking human perception via a modes system and a modular IoT pipeline that applications can embed for navigation in both physical and digital worlds, addressing SoV's training needs and real-world performance gaps while incorporating brain emulation for superior overload reduction and versatility, building on these predecessors to offer a more intuitive, scalable solution for VIP independence. In addition, evaluations across studies ranging from 12 to 90 participants showcase 85% to 95% accuracies and qualitative benefits like reduced errors (e.g., 38% in MSSD). These studies introduce an innovative and improved SSD by adding brain emulation, AI, and IoT for advanced overload reduction and versatility.   
**Table 1**  
*Differentiation Table 1*

| Components | Zvorișteanu et al. (2021) | Sami et al. (2025) | RealSense (2025)  | Udayakumar et al. (2025)  | Ruan et al. (2025) | World Navigation Hat  |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- |
| Zero Form Single Board Computer |  |  |  |  |  | **✓** |
| LiDAR Sensor |  |  | ✓ | ✓ |  |  |
| Depth / RGB Camera | ✓ |  | ✓ | ✓ | ✓ | **✓** |
| IMU | ✓ |  | ✓ |  | ✓ | **✓** |
| Computer Vision / AI | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| Audio Output (Earphones / Bone-Conduction) | ✓ |  | ✓ | ✓ | ✓ | **✓** |
| Object Detection / Distance Estimation | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| Mobile / Wearable Form | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| Point-Cloud Projection/3D Mapping |  |  | ✓ |  |  | **✓** |

**Table 2**  
*Differentiation Table 2*

| Components | Viancy et al. (2024) | Hamilton-Fletcher & Arias (2021) | Han et al. (2025) | Commère & Rouat (2023) | Zhao et al. (2025) | World Navigation Hat  |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- |
| Zero Form Single Board Computer | ✓ | ✓ | ✓ |  |  | **✓** |
| LiDAR Sensor |  | ✓ | ✓ | ✓ |  |  |
| Depth / RGB Camera | ✓ |  | ✓ |  |  | **✓** |
| IMU | ✓ | ✓ | ✓ |  |  | **✓** |
| Computer Vision / AI | ✓ | ✓ | ✓ | ✓ |  | **✓** |
| Audio Output (Earphones / Bone-Conduction) | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| Object Detection / Distance Estimation | ✓ | ✓ | ✓ | ✓ |  | **✓** |
| Mobile / Wearable Form | ✓ |  | ✓ | ✓ |  | **✓** |
| Point-Cloud Projection/3D Mapping |  | ✓ |  |  |  | **✓** |

**Table 3**  
*Prior Arts*

| Patent Title | Authors | Parameters | Publication No. | Publication Date |
| ----- | ----- | ----- | ----- | ----- |
| Stereo Vision-Based Sensory Substitution for the Visually Impaired | Simona Caraiman, Ovidiu Zagan | Stereo vision cameras, IMU integration, depth computation (Z \= 𝑓B/d), obstacle-to-audio/haptic mapping | PMC6630569 | June 20, 2019 |
| AI-Powered Smart Vision Glasses for the Visually Impaired | Devi Udayakumar, S. Gopalakrishnan | LiDAR, RGB camera, AI fusion, gesture/voice control, mobile app integration | PMC12178407 | May 30, 2025 |
| NIIRA: Non-Invasive Image Resynthesis into Audio (Sonic Vision for the Blind) | RealSense & Eyesynth Team | RealSense D415 camera, point-cloud depth mapping, SLAM, bone-conduction sound, obstacle detection up to 5 m | Intel RealSense Case Study | May 2025 |

# **Materials and Methods**

# **Research Design/Methods**

This research will follow a mixed-method design to develop, test, and analyze the results of the AIoT wearable navigation hat prototype for vision substitution, acting as an "artificial eye" for visually impaired users (e.g., fully blind, near-sighted, or partially sighted). The device uses a modular pipeline (stereo camera input to point cloud generation to virtual physical map to human vision emulation to audio cues) with toggleable hand gesture-controlled navigation, text/face recognition, and expense tracking. The study will employ a mixed-methods design, which will integrate a quantitative data collection and analysis of system performance metrics in terms of accuracy, speed, recall, and error rate derived from the device's functionality, with qualitative data collection and analysis of user experiences in terms of the device's performance, portability, and adaptability during device usage. As well as that, the quantitative findings for this research will be derived from the statistical analysis of data collected, including FPS, error percentages, success percentages, and route accuracies, and the qualitative insights will focus on the subjective user feedback on adaptation, ease of use, and overall comfort and performance in real-world scenarios of the device's functionality. In addition, in testing, there will be a within-subject experimental design and paired statistical testing (Paired T-Test), ensuring adequate power to detect significant differences between State A (continuous streaming of existing SSDs) and State B (research’s prototype, named Stesis, Selective Filtering). The functionality is prioritized in controlled/naturalistic settings, with p-values (e.g., t-tests, ANOVA at p \< 0.05) used to assess statistical significance, ensuring results are not due to chance. Utilizing the statistical tests, such as paired T-Test, P-Test, and ANOVA, assess reliability, and triangulation between methods validates findings, informing design refinements and potential clinical applications.

In addition, the researchers will use the Engineering Design Process (Figure 2\) to create a wearable navigation hat for visually impaired users. The goal is to ensure it meets VIPs needs while being comfortable and functional. The researchers will select key components and assemble them into a modular design, supported with clear diagrams and user-friendly models. Tailoring EDP, the navigation hat's prototyping method involves seven phases: 1\) identifying the problem and constraints, 2\) brainstorming possible solutions, 3\) selecting a solution and developing a design, 4\) building a prototype, 5\) evaluating the prototype, 6\) redesigning and optimizing prototype, and 7\) communicate results. This process focuses on creating practical and effective assistive devices tailored to users' needs and technical requirements.

[**Figure 2**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_research)  
*Engineering Design Process (EDP) diagram of the Study*

1) ### **Identify the Problem and Constraints**

Identify the key problems and constraints VIP faces, including difficulties navigating their environment, safety risks, and cognitive overload from existing aids such as white canes and basic sound-sensing devices. Identify the needs for advanced technology that can effectively navigate and replicate human vision without requiring extensive user training. Establish the requirements for the hat and outline the project's necessary functions, including converting 3D environmental data into audio cues and ensuring portability through features such as battery power and ergonomic design. Constraints such as budget considerations, available components (e.g., the Raspberry Pi Zero 2W or other zero-form-factor single-board computers), and ethical guidelines (e.g., obtaining Institutional Review Board (IRB) approval). The focus remains on specific VIP challenges such as social isolation, navigation, mobility, and employment barriers, to align with evaluating AI frame rate (FPS) and user adaptation.

2) **Brainstorm Possible Solutions**

Illustrate various possible ideas for wearable designs, drawing inspiration from existing technologies such as Sound of Vision and Smart Hat. Consider emphasizing the use of IoT and AI for photogrammetry and inertial measurement units (IMUs), with a creative approach to enhance human senses and thereby reduce cognitive load. The goal is to connect physical and digital experiences to benefit a broader range of users.

3) ### **Select a Solution and Develop a Design**

Select the best design for implementation. This entails creating a detailed development plan that includes component selection, circuit diagrams, prototyping timelines, and considerations for ethical data collection and risk management. The prototype will focus on creating a wearable hat that integrates AIoT, providing audio cues while ensuring comfort and customizability for VIP users.

4) **Build a Prototype**

In the prototyping stage, the researchers will assemble the components according to specifications and conduct initial tests to verify functionality. Document the entire process, including any challenges faced, to streamline improvements. The development will focus on a user-friendly, efficient, ergonomically designed prototype for VIPs, integrating AIoT features for real-time feedback.

5) **Evaluate the Prototype**

A pilot test with 3 to 5 participants will be conducted before full implementation to evaluate device stability, safety, clarity of instructions, and environmental control. Adjustments identified during this phase will be made prior to the main experiment. Evaluation of the prototype involves conducting both quantitative and qualitative tests. This includes measuring AI FPS, depth accuracy, and other collected data, along with user trials to assess comfort and understanding. Feedback from focus groups and careful analysis of results will inform further refinements, helping identify and document any issues encountered during prototype testing.

6) ### **Redesign and Optimize**

Based on the feedback received, the prototype design will undergo iterative refinement. Changes may include enhancements to audio algorithms or modifications to extend battery life, ultimately striving to optimize the prototype for practical use by visually impaired individuals.

7) **Communicate Results** 

Shares the outcomes of prototyping and testing with key stakeholders. The outcome report comes with quantitative data on the accuracy of depth mapping, responsiveness of hand-gesture controls, AI processing speeds (FPS), and route-tracking precision, alongside qualitative feedback from user trials on comfort, ease of use, and system effectiveness. The researchers will communicate through clear reports, visualizations (e.g., graphs, tables, and 3D models), and demonstrations. 

# **Material and System Requirements**

The World Navigation Hat operates through a multi-stage processing pipeline that begins with data acquisition and culminates in audio feedback delivery. The Orange Pi Zero 2W microcontroller, powered by a 5V/3A UPS module, serves as the central processing unit that orchestrates all system operations. The process initiates with stereo image capture from dual OV5640 USB cameras positioned 12 cm apart, providing 5MP resolution at 160° field of view for photogrammetry operations. Concurrently, the MPU6050 Inertial Measurement Unit tracks head motion and orientation to align the virtual environment with the user's vestibular senses, preventing sensory conflicts. Audio input is captured through an earphone microphone for voice command recognition. Five primary applications run simultaneously on the device:

1. Photogrammetry processing that converts 2D stereo images into depth maps and 3D point clouds using stereo disparity calculations.  
2. Human visual sense emulation pipeline that implements foveation, selective attention, and scene gist extraction to generate necessary audio cues while minimizing cognitive overload.  
3. MediaPipe AI for real-time hand gesture recognition enabling touchless device control.  
4. Modular framework allowing custom applications to integrate into the main pipeline with connection to the main server through standard APIs.  
5. Processed data transmitted via the WiFi to a NodeJS server infrastructure that handles additional computational tasks including external API integration, SQL database management, and HTTP server operations for client-server communication.

**Hardware Materials**  
The hardware architecture integrates several critical components optimized for portable wearable computing. 

The Orange Pi Zero 2W single-board computer features a quad-core ARM Cortex-A53 processor with 512MB/1GB RAM options, providing sufficient computational power for real-time computer vision tasks while maintaining low power consumption suitable for battery operation. 

Two OV5640 USB cameras serve as the stereo vision system, each delivering 5-megapixel resolution with a 160° field of view and positioned with a 10 cm baseline to enable accurate depth estimation through stereo disparity calculations. 

The MPU6050 6-axis Inertial Measurement Unit combines a 3-axis gyroscope and 3-axis accelerometer, providing motion tracking data at high refresh rates to synchronize the virtual environment with user head movements. 

The Orange Pi Extension Board’s audio module facilitates bidirectional audio communication, connecting both speaker output for audio cues and microphone input for voice commands through standard 3.5mm earphone jacks.

Power management is handled by a custom UPS module accepting 5V/3A input, featuring lithium battery charging circuits (supporting 18650 or similar rechargeable cells) and voltage regulation to ensure stable operation during mobile use. 

All components are mounted on a 3D custom-designed hat frame measuring 333.38 mm in length with integrated cable management and component housings to maintain ergonomic comfort during extended wear.

**Software Specifications**

The software ecosystem operates across two primary environments: client-side processing on the Orange Pi Zero 2W and server-side operations on a NodeJS platform. 

The client-side implementation utilizes Python 3.x as the primary programming language, leveraging OpenCV (cv2) for image acquisition, preprocessing, and stereo vision algorithms including stereo calibration, rectification, and disparity map generation.

NumPy provides optimized numerical computing for point cloud transformations, matrix operations in photogrammetry calculations (applying the stereo projection matrix Q to depth maps), and efficient array manipulations for real-time processing.

Matplotlib enables visualization during development and debugging phases, generating spectrograms for audio cue analysis and 3D scatter plots for point cloud verification. 

MediaPipe framework implements hand gesture recognition through pre-trained machine learning models optimized for mobile deployment, detecting key hand landmarks and classifying gestures with minimal latency. 

PyTorch serves as the deep learning backend for custom neural network implementations, including depth estimation refinement models and audio generation networks that emulate human auditory processing. 

Additional Python libraries include SciPy for signal processing (audio synthesis, filtering), librosa for audio feature extraction, and pyserial for communication with peripheral modules (MPU6050). 

The server-side infrastructure runs NodeJS with Express framework for HTTP server operations, handling REST API endpoints for client requests, managing WebSocket connections for real-time bidirectional communication, and coordinating database operations through SQL query execution. 

The development workflow integrates multiple productivity and collaboration tools to streamline project organization, documentation, and version control. 

Visual Studio Code (VSCode) serves as the primary integrated development environment (IDE), providing syntax highlighting for Python and JavaScript, integrated terminal access for remote SSH connections to the Orange Pi Zero 2W, Git integration for version control operations, and extensive extension support including Python linters (pylint, flake8), formatters (black, autopep8), and remote development extensions for direct editing on embedded systems. 

GitHub hosts the project repositories, implementing version control through Git for both client-side Python applications and server-side NodeJS code, facilitating collaborative development among team members through branch management, pull requests, code reviews, and issue tracking for bug reports and feature requests. 

LaTeX document preparation system generates the formal thesis documentation with precise mathematical typesetting for equations including stereo disparity formulas, point cloud projection matrices, and depth-to-audio mapping functions, ensuring professional formatting standards for academic submission while maintaining version-controlled .tex source files. 

Google Docs provides real-time collaborative editing for preliminary research documentation, literature review organization, meeting notes, and draft content development, enabling simultaneous multi-user editing with comment threads for peer feedback and revision tracking. 

Microsoft Office suite (Word, Excel, PowerPoint) supports supplementary documentation tasks including formatted research proposals, data analysis spreadsheets for experimental results (accuracy measurements, user feedback surveys, performance benchmarks), and presentation materials for design reviews and project demonstrations. 

Claude AI assists throughout the development lifecycle as an AI-powered coding assistant and research tool, providing algorithm optimization suggestions, debugging support for complex stereo vision calculations, documentation generation for API endpoints and function specifications, literature synthesis for theoretical framework development, and technical writing refinement for thesis chapters. 

SolidWorks is CAD software used to create 3D models and conduct simulations in engineering and manufacturing. The software is used to create the 3D design of the navigation hat's prototype device system.

KiCad is a free, open-source tool for designing electronic circuits and PCBs. The software application is used to design and simulate the schematics that connect various components for the navigation hat project. 

This integrated software toolchain ensures systematic project management from initial concept through final implementation, maintaining code quality through version control, facilitating team collaboration across distributed work environments, and producing comprehensive documentation that meets both academic standards and technical reproducibility requirements.

**Library and Board Managers**  
Development and deployment require specific software dependencies managed through multiple package systems across different platforms. For the Orange Pi Zero 2W running Armbian or Ubuntu-based distributions, the APT package manager handles system-level dependencies including Python 3.x development packages (python3-dev, python3-pip), OpenCV system libraries (libopencv-dev, python3-opencv), audio system requirements (libasound2-dev, portaudio19-dev), and USB camera support utilities (v4l-utils). Python package management through pip3 installs critical libraries with specific version constraints: opencv-python (≥ 4.5.0) for computer vision operations, numpy (≥ 1.21.0) for numerical computing, mediapipe (≥ 0.8.0) for hand tracking AI models, torch (≥ 1.10.0) with ARM-compatible builds for deep learning inference, sounddevice and soundfile for audio I/O operations, pyserial (≥ 3.5) for hardware communication, and requests for HTTP client functionality. Arduino IDE or Platform IO manages firmware for any auxiliary microcontrollers if implemented, including board definitions for custom boards and library dependencies for sensor communication protocols (I2C for MPU6050, SPI for peripheral modules). The NodeJS ecosystem uses npm (Node Package Manager) to install server-side dependencies: express (≥ 4.18.0) for web server framework, ws for WebSocket implementation, sequelize or knex for SQL database ORM, axios for HTTP client operations to external APIs, and child\\\_process management for spawning Ollama and YOLO processes. Board-specific configurations require Orange Pi GPIO libraries (OrangePi.GPIO or wiringOP) for direct hardware pin control if needed, and kernel module loading for camera interfaces (ensuring proper v4l2 driver support for OV5640 USB cameras through modprobe configurations). Git version control manages source code across development iterations, with repositories configured for both client and server codebases, while systemd service configurations enable automatic startup of critical applications on boot for production deployment.

### **Other Materials/Equipment/Devices**

Beyond the core electronic components, the project requires supplementary materials and equipment for fabrication, testing, and evaluation. The physical enclosure utilizes 3D-printed components designed in CAD software, with wireframe dimensions of 333.38 mm length, 197.50 mm width, and component-specific mounting features including camera housings with 12 cm spacing alignment, ventilated sections for heat dissipation from processing components, and cable routing channels. The 3D printing process uses PLA filament for structural components requiring strength and thermal stability, with print settings optimized for layer adhesion and surface finish to ensure user comfort during extended wear. A standard baseball cap or adjustable hat frame serves as the foundation for component integration, providing familiar ergonomics and adjustable sizing through standard strap mechanisms. Micro SD card (minimum 32GB Class 10 or UHS-I) stores the operating system, application code, and temporary data cache for the Orange Pi Zero 2W. Lithium-ion battery cells (18650 format, 3.7V nominal, minimum 2500mAh capacity) power the UPS module, with appropriate battery holders and spot-welded connections ensuring reliable electrical contact and mechanical stability. USB-C cable provides a charging interface for the UPS module, allowing users to recharge using standard mobile phone chargers or power banks. Testing and evaluation equipment includes a reference measuring device (laser rangefinder or calibrated ruler) for depth accuracy validation, comparing predicted distances from stereo disparity calculations against ground truth measurements across various ranges (0.5m to 5m typical operating distance). A standardized navigation course with marked obstacles, varying textures, and elevation changes facilitates user testing protocols for evaluating device effectiveness. Audio analysis equipment (reference microphone, audio interface, spectrum analyzer software) characterizes the generated audio cues for frequency response, dynamic range, and spatial positioning accuracy. Additionally, development workstations (laptop/desktop computers with Linux or Windows OS) run cross-compilation toolchains, debugging interfaces (SSH, serial console), and simulation environments for algorithm validation before hardware deployment. Safety equipment for user testing includes protective padding for obstacle courses, first aid supplies, and emergency communication devices in compliance with IRB-approved testing protocols outlined in the ethical considerations section.

# **Prototype Building**

[**Figure 3**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_block)  
*Block diagram of the System*

[Figure 3](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#fig_block) illustrates the simplified hardware-software interaction flow of the World Navigation Hat system, showing the essential component connections and data pathways. The architecture begins with power distribution from a 5V Lithium Battery that feeds into the UPS Module (5V/3A), which provides regulated power (indicated by a lightning bolt symbol) to the central Microcontroller (Orange Pi Zero 2W or similar embedded system). Multiple input sensors connect directly to the microcontroller: the Digital Camera (OV5640) captures stereo visual data for photogrammetry and depth mapping, the IMU (MPU6050) provides 6-axis motion tracking data for head orientation and movement detection. The microcontroller outputs audio feedback through the Speaker, delivering spatial audio cues that convey environmental information to the visually impaired user. Bidirectional communication flows between the microcontroller and the SIM and GPS Module, which provides both GPS positioning data for outdoor navigation and wireless connectivity options (WiFi or Mobile Data, indicated by wireless symbols) to enable internet access. This connectivity links the device to a remote Server (NodeJS), which hosts computationally intensive processes like Local AI models and facilitates integration with External APIs, all of these are mock APIs has actually implementing the applications is the role of the third party developers and not necessary to prove our hypothesis. The server acts as the computational backbone for tasks beyond the microcontroller's capacity, processing complex AI inference requests and coordinating with cloud services, while the embedded system handles real-time sensor fusion, basic computer vision operations, and immediate audio feedback generation—creating a distributed computing architecture that balances on-device responsiveness with cloud-enhanced intelligence for comprehensive assistive navigation capabilities.  
[**Figure 4**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_architecture)  
*System Architecture* 

Figure 4 illustrates the system architecture diagram of the complete data flow and component interaction of the World Navigation Hat, showing how sensory input is processed and transformed into actionable navigation assistance for visually impaired users. The process begins with three primary sensor categories: the dual OV5640 cameras (5MP at 160° FOV, spaced 10 cm apart) capture stereo images for photogrammetry, the Inertial Measurement Unit (IMU) tracks head motion and orientation. These inputs feed into the Orange Pi Zero 2W microcontroller, which runs four core applications concurrently: (1) Emulated Vision to Audio—converting stereo images into 3D point clouds and depth maps, then applying human visual processing emulation (foveation, selective attention, scene gist extraction) to generate spatially-aware audio cues delivered through the earphone speaker; (2) Photogrammetry—reconstructing 3D environmental geometry from stereo disparity calculations; (3) Hand Gesture Recognition via MediaPipe AI—detecting user hand gestures for touchless device control and parameter adjustment; and (4) Server API integration—connecting to external computational resources. Data flows bidirectionally between the Orange Pi Zero 2W and a central server platform, where NodeJS processes coordinate with external APIs (e.g. Google Maps, Meta platforms) to enhance navigation capabilities. The server infrastructure manages three interconnected virtual map representations: the Virtual Physical Map derived from real-time photogrammetry and GPS positioning showing the user's immediate surroundings with obstacle locations, the Virtual Digital Map providing access to online navigation services and digital content, and a unified spatial model that bridges physical and digital navigation contexts. Throughout this pipeline, the IMU continuously synchronizes the virtual environment with the user's head movements to maintain vestibular alignment and prevent disorientation, while the modular architecture allows applications to access any stage of the processing pipeline—enabling customization for specific user needs such as text reading, face recognition, or expense tracking—all working cohesively to transform complex visual-spatial information into intuitive, non-overwhelming audio feedback that empowers independent navigation for visually impaired individuals.  
[**Figure 5**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_3d)  
*3D Model Design of the Prototype*  
Figure 5 is the 3D CAD rendered model of the prototype that shows the physical appearance of the World Navigation Hat's core electronics enclosure, designed as a compact, wearable sensory substitution device for visually impaired individuals. The cylindrical housing features a removable top lid embossed with a stylized logo, revealing the internal Orange Pi Zero 2W single-board computer (shown in green PCB with the prominent blue Allwinner H618 processor) mounted on a platform with ventilation holes for heat dissipation during intensive AI processing tasks. The device integrates into a modified hat brim structure (visible as the curved frame at the base) with attachment points and cable routing channels that secure cameras, sensors, and the UPS battery module while maintaining ergonomic comfort for extended wear. Small external ports (visible on the lower left) provide access for USB charging, while the sealed design protects sensitive electronics from environmental elements during outdoor navigation, embodying the project's goal of transforming bulky assistive technology into an unobtrusive, socially acceptable wearable that converts real-time visual information into intuitive audio cues through advanced photogrammetry, AI-driven scene understanding, and human sensory process emulation.

[**Figure **](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_wireframe)**6**  
*Wireframe of the Device*

[**Figure**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_measurement) **7**  
*Measurement of the Prototype*

| A: Hat top diameter  B: Total height  C: Camera distance D: Hat Trim diameter  E: Hat Brim diameter  F: Hat Sweatband diameter  G: Wire clamp outer length  H: Strap handle outer length  I: Earphone wire channel radius  | J: Strap handle inner length K: Wire clamp width L: Wire clamp inner length M: Camera height \- from bottom of hat N: Total height \- without strap handle O: Cover diameter P: Antenna hole position \- relative to hat Q: Antenna hole diameter |
| :---- | :---- |

[**Figure**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_circuit) **8**  
*Complete Circuit Diagram of the System*  
![][image1]  
[Figure ](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#fig_circuit)8 illustrates the complete hardware interconnection architecture of the World Navigation Hat system, showing how all components integrate to form a functional wearable assistive device for visually impaired individuals.  
**Power Distribution System**  
The USB-C 5V/3A UPS Module serves as the primary power source, positioned at the bottom of the diagram with a yellow arrow indicating the external power input connection. This rechargeable power management unit supplies regulated 5V power to all system components, with Chinese text indicating battery charging specifications and protection circuitry. The green PCB features multiple power regulation stages with visible inductors (marked as 74Z), capacitors (both electrolytic and ceramic), and battery management ICs to ensure stable operation during mobile use and support lithium battery charging cycles. Power consumption and battery life calculations of the system device: 

1) Battery Energy Calculation

		Formula: Ebatt \= Qbatt × 2 ×Vbatt1000  
Qbatt \= 3600 mAh (capacity of one battery, doubled for two batteries)  
Vbatt \= 3.7 V  
		Solving:   
Ebatt \= 3600mAh × 2 × 3.7V1000 \= 26\. 64 Wh

2) Component Power Calculations  
   	2.a) Orange Pi  
   		Ppi-idle ≈ 1.09W 

   Ppi-active ≈ 2.4W 

   Ipi-peak \= 2A   
   Vpi \= 5V  
   	Ppi-idle \= 1.09W   
   Ppi-active \= 2.4W

   2.b) OV5640  
   		I5640-sleep \= 20𝜇A

   I5640-active \= 140mA   
   V5640 \= 3.3V

   Sleep:

   P5640-sleep \= I5640-sleep  V5640 \= 20𝜇A × 3.3V \= 66𝜇W \= 0.000066 W

   Active: 

   P5640-active \= I5640-active  V5640 \= 140mA × 3.3V \= 462mW   
   	   \= 0.462 W

   2\. c) Earphones

   Iearph ≈ 19.02mA   
   Vearph ≈ 0.21V  
   Pearph \= Iearph  Vearph \= 19.02mA × 0.21V ≈ 4mW \= 0.004 W  
     
3) Total System Power (Accounting for UPS Efficiency, (η\_ups) is 85% (0.85))

   Formula: Ptotal-mode \= Pcomponentsη\_ups

   Sleep Mode: 

   	Ptotal-sleep \= Ppi-idle \+ P5640-sleep \+ Pearphη\_ups

   		 \= 1.09 \+ 0.000066 \+0.00485%

   		 \= 1.094066 85%

   		 \= 1.2871364W ≈ 1.3W

   Active Mode:

   		Ptotal-sleep \= Ppi-active \+ P5640-active \+ Pearphη\_ups

   			 \= 2.4 \+ 0.462 \+0.00485%

   			 \= 2.866 85%

   			 \= 3.37 W 

4) Battery Life Calculation

		Formula: Tmode \= EbattPtotal-mode  
		Sleep Mode:   
				Tmode \= EbattPtotal-sleep \= 26.641.3 \= 20.4923077 h ≈ 20.49 h  
		Active Mode:  
				Tmode \= EbattPtotal-active \= 26.643.37 \= 7.9 h   
Therefore, the device can run approximately 20.49 hours in sleep mode and 7.9 hours in active mode, based on the battery capacity and calculated power draws. 

**Central Processing Unit**  
The Orange Pi Zero 2W single-board computer occupies the central position in the architecture, acting as the main computational hub. It features the Allwinner H618 quad-core processor (visible as the large silver chip), USB ports for camera connectivity, a micro-SD card slot for operating system and data storage, GPIO header pins for peripheral connections, and HDMI output. The board receives 5V power from the UPS module (indicated by red power lines) and ground connections (black lines), while managing bidirectional data communication with all sensors and actuators.

**Stereo Vision System**  
Two OV5640 USB Cameras are positioned at the top right of the Figure 8 diagram, each providing 5MP resolution at 160° field of view. These cameras are mounted with a precise 12 cm baseline separation (as noted in the physical prototype specifications) and connect to the Orange Pi Zero 2W via USB interfaces (shown by thick black connection lines). The stereo camera configuration enables depth perception through stereo disparity calculations, capturing simultaneous images from slightly different viewpoints to generate data essential for environmental mapping and obstacle detection.

**Motion Sensing**

The MPU6050 Inertial Measurement Unit (6-axis gyroscope and accelerometer) appears as a small blue module at the center-top of the Figure 8 diagram. It connects to the Orange Pi Zero 2W through I2C communication protocol (indicated by green connection lines labeled with I2C pins: SCL, SDA, VCC, GND). This sensor tracks head motion and orientation in real-time, providing critical data for aligning the virtual environment representation with the user's physical movements, preventing vestibular-visual conflicts that could cause disorientation or nausea.

**Audio Interface**  
The Orange Pi Extension Board V1.0 includes a 3.5mm jack supporting both audio output and microphone input through a single connector. The earphone illustration at the Figure 8 shows standard earbud-style output for delivering spatial audio cues to the user.

**Connection Architecture**  
Color-coded wiring illustrates the data flow and power distribution:

* Red lines: 5V power rails from UPS to all components  
* Black lines: Ground connections ensuring common reference voltage.  
* Green lines: I2C communication (MPU6050), or other digital communication protocols.  
* Thick black lines: USB data connections for high-bandwidth camera feeds.

**Physical Integration**  
All components are designed to mount within the 3D-printed hat enclosure shown in previous wireframe diagrams, with the UPS module likely positioned at the rear for weight distribution, the Orange Pi Zero 2W in the central crown area for protection, cameras mounted at the front brim for unobstructed field of view, and the IMU positioned near the cameras to accurately track head orientation relative to the visual input direction. The compact arrangement maintains the wearable form factor while ensuring proper thermal management through ventilation and component spacing.

[**Figure 9**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_schematic)  
*Complete Schematic Diagram of the System*

Figure 9 illustrates the schematic diagram of the entire system, showing the layout of the power, sensor, processing, communication, and audio modules, along with their interconnections, that form the wearable navigation hat system. The power system of the device is the battery (BT1) that powers the system via an uninterruptible power supply (UPS1), distributing stable power to the Orange Pi Zero 2W and other components. The main processing unit is the Orange Pi Zero 2W that serves as the central controller, interfacing with two USB cameras for stereo image capture and 3D reconstruction. The sensor modules that are connected to the Orange Pi are the MPU-6050 IMU via I2C or motion tracking. The audio output for the navigation cues is delivered through the TRRS bus to an earphone (LS1), with resistors for signal integrity and volume control. The system includes integrated earth and ground points to ensure stable reference voltages and minimize noise.

[**Figure 10**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_cam1)  
*OV5640 Camera Schematic Diagram*

Figure 10 shows the schematic diagram of the USB camera interface (DVP-USB1) connected to the camera module (OV5640-DVP1) for your wearable navigation hat system. The DVP-USB1 captures data from the USB camera. It provides power (USB-VCC), ground (USB-GND), data lines (USB-Data+ and USB-Data–), pixel data signals (Y0-Y9), and clock/control signals (PCLK, VSYNC, HREF, XCLK, SIO\_CLK, SIO\_DAT, PWDN, RESET). The OV5640-DVP1 receives these signals with matching pins for pixel data, grounds (AGND and DGND), and power supplies (AVDD, DVDD, DOVDD). Each pin from DVP-USB1 connects directly to the corresponding pin in OV5640-DVP1. It supports real-time stereo image acquisition and helps create depth maps for the navigation hat system.

[**Figure 11**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_cam2)  
*Component Diagram Orange Pi Expansion Board to Orange Pi Zero 2W*

**Table 4**  
*Pin Configuration of Orange Pi Expansion Board to Orange Pi Zero 2W*

| Wire/Cable | From (Expansion Board USB) | To (DVP-to-USB Bridge) | Function |
| :---- | :---- | :---- | :---- |
| Blue Trace 1 | USB 5V | 5V | Power Supply |
| Blue Trace 2 | USB D- | D- | USB Data \- |
| Blue Trace 3 | USB D+ | D+ | USB Data \+ |
| Blue Trace 4 | USB GND | GND | Ground |

Figure 11 and Table 1 illustrate the Orange Pi Zero 2W connects to the Orange Pi Expansion Board through a single flexible 24-pin FPC ribbon cable that links the matching 24-pin interfaces on both devices. This connection carries all essential signals including power lines, GPIO controls, and communication buses such as I2C, SPI, and UART from the Zero 2W’s processor to the Expansion Board. By relaying these signals, the Expansion Board expands the capabilities of the compact Zero 2W, providing access to additional peripherals such as full-sized USB ports, an RJ45 Ethernet jack, and an audio jack, thereby enhancing the system’s overall functionality.  
[**Figure 1**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_cam2)**2**  
*Component Diagram UPS Module to Orange Pi Orange Pi Zero 2W*

**Table 5**  
*Pin Configuration of UPS Module to Orange Pi Orange Pi Zero 2W*

| Wire Color | From (UPS Module) | To (Orange Pi Zero 2W) | Function |
| :---- | :---- | :---- | :---- |
| Red | UPS+ (Top Right Pad) | Pin 2 (Physical Pin 2\) | \+5V Power Input |
| Black | \-UPS (Bottom Right Pad) | Pin 6 (Physical Pin 6\) | Ground (GND) |

Figure 12 and Table 2 illustrate the power system is designed as a battery-centric Uninterruptible Power Supply (UPS) using an 18650 5V/3A UPS Module and two 18650 batteries to power the device. The Orange Pi Zero 2W receives its 5V operating voltage directly from the UPS module’s output terminals, with the 5V+ terminal (red wire) connected to the 5V pin on the Orange Pi’s GPIO header and the 5V– terminal (black wire) connected to a GND pin. The UPS module draws energy from the batteries, regulates it to a stable 5V, and supplies it to the Orange Pi Zero 2W, ensuring continuous operation even if the external power source is disrupted, thereby providing true uninterruptible power capability.

[**Figure 1**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_cam2)**3**  
*Component Diagram Earphones to Orange Pi Orange Pi Zero 2W*

**Table 6**  
*Pin Configuration of Earphones to Orange Pi Orange Pi Zero 2W*

| Component | Connected To | Description |
| :---- | :---- | :---- |
| 3.5mm Earphones | Expansion Board Audio Jack | Standard analog audio output for music or system sounds. |

Figure 13 and Table 3 illustrate the audio output capability of the system is realized by connecting standard earphones directly to the Orange Pi Expansion Board. The earphones connect to the Orange Pi Expansion Board through a standard 3.5mm audio jack, receiving an audio signal that originates from the Orange Pi Zero 2W, travels through the 24-pin FPC interface to the Expansion Board, and is routed to the audio jack circuitry, allowing users to listen to audio playback directly from the system.  
[**Figure 1**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_cam2)**4**  
*Component Diagram Orange Pi Expansion Board to Camera OV5640*

**Table 7**  
*Pin Configuration Orange Pi Expansion Board to Camera OV5640*

| Wire/Cable | From (DVP-to-USB Bridge) | To (OV5640 Camera) | Function |
| :---- | :---- | :---- | :---- |
| Ribbon Cable | FPC Connector | FPC Connector | DVP Interface (Video Data) |

\`	Figure 15 and Table 5 illustrate the system connects two OV5640 Camera Modules to the Orange Pi Expansion Board using a chain of components. The dual camera setup connects two OV5640 camera modules to a 2-channel DVP-to-USB bridge board, which converts their video signals into a single USB output. This USB output is then linked to one of the USB ports on the Orange Pi Expansion Board using a ribbon cable. From there, the video data travels through the Expansion Board, which is connected to the Orange Pi Zero 2W via a 24-pin FPC cable, allowing the Expansion Board to act as a hub that passes the combined USB video stream into the Zero 2W for processing. This arrangement enables the Orange Pi Zero 2W to handle dual OV5640 camera feeds through its extended USB interface, making it suitable for applications like stereo vision or multi-angle imaging.  
[**Figure 1**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_cam2)**5**  
*Component Diagram MPU9250 to Orange Pi Zero 2W*

**Table 8**  
*Pin Configuration MPU9250 to Orange Pi Zero 2W*

| Wire Color | From (MPU9250) | To (Orange Pi Zero 2W) | Function |
| :---- | :---- | :---- | :---- |
| Red | VCC | Pin 1 | 3.3V Power |
| Blue | SDA | Pin 3 | I2C SDA (Data) |
| Blue | SCL | Pin 5 | I2C SCL (Clock) |
| Black | GND | Pin 9 | Ground (GND) |

Figure 16 and Table 6 illustrate that the MPU9250 Inertial Measurement Unit (IMU) connects to the Orange Pi Zero 2W using four wires through the GPIO header and the I2C interface. Two wires are for communication: a black wire connects the SCL pin of the MPU9250 to the I2C Clock pin on the Orange Pi, while a blue wire connects the SDA pin to the I2C Data pin. For power, a red wire connects the MPU9250's VCC pin to the Orange Pi's 3.3V pin, and a black wire connects the GND pin to a ground pin on the Orange Pi.

[**Figure 1**](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#figur_flow)**6**  
*Flowchart of the Three Main Pipelines*

The flowchart at [Figure 12](https://docs.google.com/document/d/1N4Ya048OJufQJaMhjFfynmU-Aq_rgIfC3wOmSFFCeBE/edit#fig_flow) illustrates the modular software architecture of the World Navigation Hat, organized into three concurrent processing pipelines that work together to provide comprehensive assistive navigation through sensory substitution. The Main Pipeline forms the core sensory substitution system, beginning with "Two images from stereo cameras" captured simultaneously from the dual OV5640 cameras positioned 10 cm apart, which undergo photogrammetry processing by applying stereo disparity calculations (Z \= f·B/d) that create a three-dimensional representation of the environment where each pixel's depth is computed and transformed into 3D spatial coordinates; this point cloud data is then used to "Record to virtual environment," creating a persistent spatial map integrated with IMU-tracked head movements to form a coherent representation of the user's surroundings. The Gesture Pipeline runs in parallel, providing touchless user control through "Scan for hand gesture" using MediaPipe AI models that continuously analyze camera frames to detect hand landmarks—when a gesture is detected, the system evaluates "Gesture matches an instruction?" and if true, proceeds to "Update mode of expression," modifying parameters that affect how the Main Pipeline generates audio output (allowing users to switch between obstacle detection, text reading, or face recognition modes), then checks "Instruction requires Services?" to determine whether the command needs server-side processing, branching to the Server Pipeline if yes or merging directly back to the Main Pipeline if no. The Server Pipeline handles computationally intensive tasks, where requests are transmitted via the WiFi to the NodeJS platform, which may invoke external APIs, then proceeds to "Read response" to collect and format results before "Add to output audio" packages the server-processed information and transmits it back to merge with the Gesture Pipeline. All three pipelines converge through diamond-shaped merge points before the final stage where the system "Generate audio based on mode of expression," transforming consolidated environmental data, user control inputs, and cloud-enhanced intelligence into spatial audio cues through algorithms that emulate human visual processing—implementing foveation (focus-based detail levels), selective attention (filtering salient features), scene gist extraction (background/foreground separation), and temporal dynamics (progressive audio updates)—before finally proceeding to "Output audio to earphones," delivering intuitive sonic representations that respect cognitive load limitations while maximizing navigational utility, embodying the research's core innovation of solving the bandwidth mismatch between high-detail visual information and lower-bandwidth auditory perception.

# **Data Presentation & Analysis**

In this study, the researchers will use specific formulas and statistical tools to quantify the system's depth conversion, precision, and gesture detection. Furthermore, the metrics will be used by the researchers to assess the navigation hat's performance in converting stereo camera inputs into audio cues for navigation. The results will be analyzed and displayed by two categories: 1\) Quantitative Data Presentation and Analysis, and 2\) Qualitative Data Presentation and Analysis. 

1\) Quantitative Data Presentation and Analysis 

The results will be displayed through descriptive statistics such as means and standard deviations, inferential tests (specifically t-tests with p-values), and visual aids such as charts. The analysis will emphasize the statistical significance (p \< 0.05) to discern meaningful trends, using software such as Python for reliability checks and cross-validation.

1.a) Photogrammetry Performance: The data presentation includes tables of error metrics and success rates by environment, alongside scatter plots for regression analysis. The statistical tests, paired t-tests will reveal the differences in accuracy.

Formula References:

1.a.1) Percentage Error Formula- Expresses the error between a measured value and the true value (e.g., predicted vs. actual direction or depth).

1.a.2) Success Percentage Formula- Measures the proportion of successful trials (e.g., correct predictions by user or device).

1.a.3) Stereo Camera Matrices \- Intrinsic and extrinsic representation of left *(C1)* and right *(C2)* cameras,where *f* is focal length in pixels, *cx, cy* are principal point coordinates, and *Tx* is the baseline. 

1.a.4) Stereo Reprojection Matrix *(Q)* \-Use to compute 3D points from disparity: convert (*x, y* disparity) into 3D coordinates.

1.a.5) Point Cloud Projection Formula \- Reprojects a pixel with disparity into 3D space using the *Q* matrix.

**Table 9**  
*Photogrammetry Performance Metrics*

| Environment | Mean Percentage Error (%) ± SD | Success Percentage (%) | t-test p-value  (vs. Ground Truth) |
| ----- | ----- | ----- | ----- |
| Indoor  |  |  |  |
| Outdoor  |  |  |  |
| Average |  |  |  |

Table 7 compares the mean percentage error, success percentage, and t-test p-values for depth mapping accuracy in indoor and outdoor environments, highlighting environmental impacts on the navigation hat's 3D reconstruction.

1.b) **Gesture Command Responsiveness:** Data will be presented in confusion matrices and metric tables, with binomial tests used to evaluate proportions. Whereas,

* TP (True Positive) stands as a "correct yes." For example, if the user does a thumbs-up gesture and the hat correctly switches to navigation mode. It means the hat got it right when it should have acted.   
* FP (False Positive): This is the "wrong yes." If the hat thinks the user did a thumb-up (and switches modes) but did not actually do the gesture, that is an FP. It is like a false alarm—the hat acted when it should not have.   
* TN (True Negative): The "correct no." If the user does not do any gestures and the hat stays put correctly (no mode switch), that is a TN. It means the hat did not react when it should not have.   
* FN (False Negative): A "wrong no". The user does a gesture, but the hat misses it and does not switch. These help calculate accuracy: more TP/TN means the hat is reliable.  
  1.b.1) Precision: Correctness of detections. Formula: TP / (TP \+ FP).   
  1.b.2) Recall: Completeness of detections. Formula: TP / (TP \+ FN).   
  1.2.c) Accuracy: Overall performance. Formula: (TP \+ TN) / Total.

**Table 10**  
*Gesture Command Responsiveness Metrics*

| Gesture | TP | FP | FN | TN | Precision | Recall | Accuracy |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Thumbs Up |  |  |  |  |  |  |  |
| Pointing Index |  |  |  |  |  |  |  |
| Open Back Hand |  |  |  |  |  |  |  |
| Close Back Hand |  |  |  |  |  |  |  |
| Average |  |  |  |  |  |  |  |

Table 8 shows the confusion matrix values of classification outcomes of hand gestures (TP, FP, FN, TN) for the navigation hat, illustrating how well it detects hand commands for mode toggling. The table also summarizes the precision, recall, and accuracy for each gesture type, with ANOVA p-values indicating variations in responsiveness across modes.

1.c) **Modular AI and GPS Performance**: Tests how well the navigation hat's built-in AI (for recognizing objects or faces) and GPS (for tracking location) work together. It is like checking whether the hat can quickly spot items and guide the user accurately as they move.  
1.c.1) Tables and graphs depict FPS trends and errors: The researchers will use charts to show how fast the hat processes data (FPS) over time and any mistakes (errors). Regression is a math trick for determining whether one thing (like speed) relates to another (like errors).  
1.c.2) Mean FPS ± SD: The average speed of the hat's processing (frames per second, like video frames). The formula sums all speeds and divides by the number of tests, with SD indicating how much it varies. Example: For speeds of 18, 19, and 18.5, the average is 18.5, and the SD of 0.5 means it is pretty consistent.  
Formula: 

* Processing speed: FPSn  
* SD for variability: (FPSi-Mean)2n-1  
  1.c.3) Error Rate: How often the AI gets incorrect (e.g., mistaking an object). Example: 2 mistakes out of 100 tests \= 2% error rate.

  Formula: FP+FNTotal× 100

  1.c.4) Route MAE: For GPS, the average distance the hat's location guess is off from the real spot (in meters). Example: Errors of 1.0, 1.5 and 1.2 meters average to 1.23 meters.  
  Formula: |errors|n

**Table 11**  
*Modular System AI Processing and Route Accuracy Metrics*

| Module | Mean FPS ± SD | Error Rate (%) | Route MAE (m) ± SD | t-test p-value (Pre/Post-Toggle) |
| :---- | ----- | ----- | ----- | ----- |
| Object Recognition |  |  |  |  |
| Face Recognition |  |  |  |  |
| Text Recognition |  |  |  |  |
| GPS Tracking |  |  |  |  |
| Average |  |  |  |  |

Table 9 presents the mean FPS, error rates, route MAE, and t-test p-values for AI modules and GPS tracking, demonstrating processing speed and accuracy on modular external APIs.  
1.d) **Full System Simulation** (Putting It All Together): Combines overall functionality for real-world tests, like navigating with the hat. ANOVA compares scenarios. Using integrated visuals that show holistic metrics, charts combine all data (like errors and successes) to give a big-picture view of the hat's performance.

* **Mean Percentage Error**: The average mistake rate across tests, as a percentage. Example: Errors of 0.05, 0.08, and 0.06 average to 6.3%.

  Formula: errorsn×100

* **Success Percentage**: How often the hat gets things right. Example: 90 successes out of 100 \= 90%.

  Formula: ST×100

**Table 12**  
*Full System Simulation Metrics*

| Scenario | Mean Percentage Error (%) ± SD | Success Percentage (%) | ANOVA p-value (Across Scenarios) |
| :---- | ----- | ----- | ----- |
| Navigation |  |  |  |
| Text Reading |  |  |  |

Table 10 integrates mean percentage errors, success rates, and ANOVA p-values across scenarios (e.g., navigation vs. text reading) to evaluate the holistic performance full simulation of the navigation hat in end-to-end use.  
2\) Qualitative Data Presentation and Analysis   
The results will be presented in thematic summaries, quotes, and narratives (in appendices). Furthermore, the analysis will use inductive thematic coding to identify patterns in user feedback, focusing on emergent themes such as acclimation barriers and comfort enablers. Triangulation with quantitative data will occur via joint displays (e.g., matrices linking metrics to themes). Reliability will be ensured through member checking (e.g., user validation of themes) and inter-coder agreement.

2.a) Photogrammetry Performance: Quotes/summaries on acclimation; coding explores comfort/adaptation, linked to error rates.  
2.b) Gesture Command Responsiveness: Narratives/themes on ease; coding identifies barriers, integrated with accuracy scores.  
2.c) Modular AI and GPS Performance: Diaries/flowcharts for trajectories; coding assesses portability, correlated with FPS.  
2.d) Full System Simulation: Highlights/maps of experiences; synthesis triangulates with metrics for design refinement.

The researchers will integrate the quantitative and qualitative findings in the discussion section. In addition, the researchers will use simple side-by-side charts to illustrate how the hat's accuracy aligns with users' usage to display comprehensive insights, demonstrating significant performance improvements (p-values \< 0.05) and aligning with users' qualitative reports on device acclimation and overall functionality. Overall, this validates how efficient the navigation hat is as an artificial eye for people with vision impairments. The researchers will also address limitations (e.g., sample bias) and implications for future improvement of assistive technologies.

**Respondents of the Study**  
	The study will involve human participants for testing navigation on a controlled obstacle course, designed by the researchers, using the study’s wearable prototype called Stereo-based Sonification Interface System (Stesis). There will be 40 participants aged 18-30 years old, selected through purposive sampling within the university community to be recruited with conditions; good vision (14 participants) and mild (14 participants), moderate (10 participants), and blind (2 participants) visual impairments with good hearing condition. Participants with good vision will be blindfolded, for experimental control and pilot and evaluation testing, as the system is intended for visually impaired users. This sample size is appropriate for within-subject experimental design and paired statistical testing (Paired T-Test), ensuring adequate power to detect significant differences between State A and State B.

# **Ethical Considerations**

1) **Informed Consent and Autonomy.** All participants in the study will receive information about their rights, including the right to participate voluntarily in prototype testing. Informed consent will be obtained through a clear explanation of the study’s purpose, duration, procedures, risks, and data use, presented in accessible formats like audio instructions. Participation is voluntary, and participants may decline to answer questions or withdraw at any time without penalties. No monetary compensation will be offered. The consent form will be read aloud, and an audio-recorded explanation will be provided. Participants will be encouraged to ask questions to ensure understanding. An independent witness will confirm comprehension before consent is given. For those unable to sign, a thumbprint will be accepted in the presence of a witness to document consent. Researchers will assess each participant's capacity for informed consent and may involve caregivers when appropriate, while preserving participants’ rights and independence. In addition, the researchers will assess participants’ capacity to consent, involving caregivers if needed, while preserving participants’ rights. Importantly, the researchers will also seek professional consultation with experts, confirm their willingness to participate, disclose any conflicts of interest, and clarify their voluntary role.

2) **Privacy and Confidentiality.** The researchers will practice data privacy and confidentiality when acquiring participants’ personal experiences or device feedback data. The researchers will obtain consent, ensure data is anonymized, and securely store it to prevent unauthorized access, excluding participants’ identifiable details from reports. In addition, during professional consultations, they often share information that must remain private, so the researchers will need confidentiality agreements to protect ideas and opinions. Participants will be informed that what they share in the group will stay within the group. The researchers will only collect the necessary data for AI models and get clear permission for any other uses. Finally, the data acquired will be protected by encryption, and access will be exclusive to authorized researchers.  
3) **Safety and Risk Assessment.** The research team will prioritize minimizing and evaluating both physical and psychological risks. Safety protocol checks and pre-tests will be implemented throughout the evaluation of the navigation hat. The device contains components, such as batteries and sensors, that may overheat, malfunction, or produce incorrect audio signals, potentially leading to falls. Emergency protocols will be in place during user testing to immediately halt testing and provide medical assistance if device failure occurs. If an injury occurs as a direct result of the device during testing, appropriate first aid will be provided immediately, and referral to medical services will be arranged as necessary. Participants may experience cognitive overload or sensory fatigue following extended device use. Monitoring for adverse effects such as disorientation or anxiety will be conducted, and psychological support will be available to ensure participant safety and well-being. The following outlines identified risks and corresponding mitigation strategies:  
* Battery Overheating: Limit each trial or test session to 20 minutes.  
* Electrical Short Circuit: Utilize insulated casing and conduct pre-test inspections.  
* Cognitive Overload: Limit exposure duration, schedule breaks during testing, and monitor using NASA-TLX.  
* Disorientation: Assign a safety guide to remain beside the participant during obstacle testing.  
* Data Leakage: Employ encrypted communication and exclude personal identifiers from data collection.  
4) **Beneficence and Non-Maleficence.** Minimize harm while maximizing benefits. The researchers will monitor the FGDs for emotional distress when discussing navigation challenges and provide support resources. Researchers will also ensure venues are accessible to prevent physical or psychological harm to participants. The survey improves the study’s quality through user feedback and enhances the device design for participants, but researchers will address the prototype’s expectations and limitations to avoid false anticipations. Improve the AI process without biased advice. Promote equity by making technology accessible and affordable for various participant groups to avoid exclusion. Evaluate the long-term impact on overall well-being and the risk of isolation due to decreased human interaction.  
5) **Justice and Inclusivity.** Fair representation will be ensured when selecting participants and consultants for survey and when evaluating the prototype, including people from different backgrounds, with varying levels of impairment, ages, genders, and local cultural views on disability in the Philippines. This helps prevent bias. Accessibility is also essential. The researchers will use tools like audio aids for visually impaired participants and ensure that logistical issues do not exclude anyone. Our goal is to benefit persons with disabilities (PWDs), especially VIPs, beyond researchers and institutions, and to promote equal participation in research outcomes.  
6) **Transparency and Integrity**. Clearly outline the study’s objectives during surveys and consultations to avoid misleading participants. The researchers will document how the input received influences the project, such as by refining the modular system. In addition, the researchers will seek neutral experts who can provide objective advice. Ethical practices will be followed, including appropriately attributing contributions while safeguarding participants’ privacy to uphold research integrity.  
7) **AI and Technology Ethics.** Potential biases in photogrammetry models or audio cues can happen when the training data is not representative. This can lead to inaccuracies and require the researchers to conduct regular fairness checks. The researchers will clearly explain AI decision-making to participants, ensuring it aligns with human sensory experiences without being misleading. The researchers must also address dual-use concerns, such as using real-time tracking for surveillance, which requires safeguards to protect user privacy and prevent unauthorized data sharing.  
8) **Inclusion and Exclusive Criteria.** The study will utilize purposive sampling with a target of 40 visually impaired participants. The participants aged 18-59 years old with good vision (14 participants) and mild (14 participants), moderate (10 participants), and blind (2 participants) with good hearing condition will participate in the study’s survey, testing, and evaluation of the prototype.  
9) **Regulatory and Institutional Compliance.** The researchers will submit plans for survey, consultations, and testing to the university’s Institutional Review Board (IRB) or an advisor for approval. This is particularly important given the vulnerability of vulnerable populations. The goal is to ensure compliance with ethical standards and legal frameworks, including the Philippines’ Data Privacy Act, especially regarding recordings and information sharing. Same with testing the device, the researchers will ensure that sensor and camera data are anonymized and encrypted to comply with privacy laws, and establish emergency protocols for handling device failures during trials. Post-activity follow-ups, such as summaries of the survey and consultation to build trust and allow feedback on the use of input. Researchers will disclose conflicts of interest, such as affiliations with tech companies, to ensure impartiality.  
10) **Declaration of Conflicts of Interest.** The researcher declares no financial, commercial, or institutional conflicts of interest. The Stesis prototype was developed exclusively for academic research and did not receive funding from any external technology company.  
    

# **References**

Al-Najjar, A. A., Winawer, J., & Hagoort, P. (2022). A parvocellular-magnocellular functional gradient in human visual cortex. bioRxiv. https://doi.org/10.1101/2022.02.08.479509  
Arora, B., Duecker, K., & Theeuwes, J. (2025). Dynamic competition between bottom-up saliency and top-down goals in early visual cortex. bioRxiv. https://doi.org/10.1101/2025.08.22.671530  
Arora, L., Choudhary, A., Bhatt, M., Kaliappan, J., & Srinivasan, K. (2024). A comprehensive review on NUI, multi-sensory interfaces and UX design for applications and devices for visually impaired users \[Review of A comprehensive review on NUI, multi-sensory interfaces and UX design for applications and devices for visually impaired users\]. Frontiers in Public Health, 12\. Frontiers Media. https://doi.org/10.3389/fpubh.2024.1357160  
Aydin, A., Smith, R., & Mital, V. (2024). Visual search in naturalistic scenes from foveal to peripheral vision: A comparison between dynamic and static displays. Journal of Vision, 24(10), 4\. https://doi.org/10.1167/jov.24.10.4  
Chavarria, M., Ortiz-Escobar, L. M., Bacca, E., Romero-Cano, V., Villota, I., Peña, J. K. M., Sanchez, J., Campo, O., Suter, S., Cabrera-López, J. J., Patiño, M. F. S., Caicedo, E., Stein, M. A., Hurst, S., Schönenberger, K., & Velarde, M. R. (2025). Challenges and Opportunities of the Human-Centered Design Approach: Case Study Development of an Assistive Device for the Navigation of Persons With Visual Impairment. JMIR Rehabilitation and Assistive Technologies, 12\. https://doi.org/10.2196/70694  
Commère, L., & Rouat, J. (2023). Evaluation of Short-range Depth Sonifications for Visual-to-Auditory Sensory Substitution.  
Gao, Y., Wu, D., Song, J., Zhang, X., Hou, B. W., Liu, H., Liao, J., & Zhou, L. (2025). A wearable obstacle avoidance device for visually impaired individuals with cross-modal learning. Nature Communications, 16(1). https://doi.org/10.1038/s41467-025-58085-x  
Gershman, S. J. (2024). Habituation as optimal filtering. iScience, 27(8), 110523\. https://doi.org/10.1016/j.isci.2024.110523  
Hamilton-Fletcher, G., Alvarez, J., Obrist, M., & Ward, J. (2021). Soundsight: A mobile Sensory Substitution Device that Sonifies Colour, Distance, and Temperature. Journal on Multimodal User Interfaces, 16(1), 107–123.  
Han, Z., Li, S., Wang, X., Hu, X., Higashita, R., & Liu, J. (2025). Multi-path Sensory Substitution Device Navigates the Blind and Visually Impaired Individuals. Displays, 91, 103200\.  
Himmelberg, M. T., Tünçok, E., Al-Najjar, A. A., & Winawer, J. (2023). Cortical magnification in human visual cortex parallels task performance around the visual field. eLife, 12, e84205. https://doi.org/10.7554/eLife.84205  
Hou, Y., Xie, Q., Zhang, N., & Lv, J. (2025). Cognitive load classification of mixed reality human computer interaction tasks based on multimodal sensor signals. Scientific Reports, 15(1).  
Loschky, L. C. (2025, October). Recognizing the gist of a scene \[Accessed: 2025-11-06\].  
Makati, T., Tigwell, G. W., & Shinohara, K. (2024). The Promise and Pitfalls of Web Accessibility Overlays for Blind and Low Vision Users. 1\. https://doi.org/10.1145/3663548.3675650  
Mishra, A., Bai, Y., Narayanasamy, P., Garg, N., & Roy, N. (2025). Spatial audio processing with large language model on wearable devices.  
Multisensory Integration: Space, Time and Superadditivity. (2025). Multisensory Integration, Space, Time and Superadditivity. https://www.researchgate.net/publication/7595223\_Multisensory\_Integration\_Space\_Time\_and\_Superadditivity  
Nghiem, T. K., Fiser, J., & Ruff, D. A. (2025). Fixational eye movements as active sensation for high visual acuity. eLife, 14, e91123. https://doi.org/10.7554/eLife.91123  
Que, L., Zhu, Q., Jiang, C., & Lu, Q. (2025). An analysis of the global, regional, and national burden of blindness and vision loss between 1990 and 2021: the findings of the Global Burden of Disease Study 2021\. Frontiers in Public Health, 13\. https://doi.org/10.3389/fpubh.2025.1560449  
RealSense AI. (2025). Eyesynth: Powering Sonic Vision for the Blind.  
Ruan, L., Hamilton-Fletcher, G., Beheshti, M., Hudson, T. E., Porfiri, M., & Rizzo, J.-R. (2025). Multi-faceted sensory substitution using wearable technology for curb alerting: A pilot investigation with persons with blindness and low vision. Disability and Rehabilitation: Assistive Technology, 20(6), 1884–1897.  
Sami, M.,Agha, D., Baloch, J., Dewani,A., & Maheshwari, K. D. (2025). Efficient object detection and voice-assisted navigation for the visually impaired: The smart hat approach. Sukkur IBA Journal of Computing and Mathematical Sciences, 8(2), 1–12.  
Shinagawa Lasik & Aesthetics. (2025, May). Perfect vision and eye health in the Philippines. \[Retrieved from: https://shinagawa.ph/perfect-vision-eye-health\]  
Theodorou, P., Tsiligkos, K., & Meliones, A. (2023). Multi-sensor data fusion solutions for blind and visually impaired: Research and commercial navigation applications for indoor and outdoor spaces. Sensors, 23(12), 5411\.  
Tokmurziyev, I., Cabrera, M.A., Khan, M. H., Mahmoud, Y., Moreno, L., & Tsetserukou, D. (2025). Llm-glasses: Genai-driven glasses with haptic feedback for navigation of visually impaired people.  
Udayakumar, D., Gopalakrishnan, S., Raghuram, A., Kartha, A., Krishnan, A. K., Ramamirtham, R., Muthangi, R., & Raju, R. (2025). Artificial intelligence-powered smart vision glasses for the visually impaired. Indian Journal of Ophthalmology, 73(Suppl 3), S492–S497.  
User Testing. (2024). 7 gestalt principles of visual perception: Cognitive psychology for ux.  
Viancy V, N. R. C., M Mouli, M., & S, S. (2024). Aural vision \- envisioning the independence of the visually impaired. 2024 International Conference on Inventive Computation Technologies (ICICT), 387–392.  
World Health Organization. (2023, August). Blindness and visual impairment. World Health Organization. \[Retrieved from: https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment\]  
Xue, W., Ren, Y., Tang, Y., Gong, Z., Zhang, T., Chen, Z., Dong, X., Ma, X., Wang, Z., Xu, H., Zhao, J., & Ma, Y. (2025). Interactive wearable digital devices for blind and partially sighted people. Nature Reviews Electrical Engineering. https://doi.org/10.1038/s44287-025-00170-w  
Zhao, Y., Yang, S., Tao, Y., & Kang, H. (2025). Evaluation of the efficacy of an assistive device for blind people: A prospective, non-randomized, single arm, and open label clinical trial. Current Eye Research, 50(8), 865–869.  
Zou, M., Chen, A., Liu, Z., Jin, L., Zheng, D., Congdon, N., & Jin, G. (2024). The burden, causes, and determinants of blindness and vision impairment in Asia: An analysis of the Global Burden of Disease Study. Journal of Global Health, 14\. https://doi.org/10.7189/jogh.14.04100  
Zvorișteanu, O., Caraiman, S., Lupu, R.-G., Botezatu, N. A., & Burlacu, A. (2021). Sensory substitution for the visually impaired: A study on the usability of the sound of vision system in outdoor environments. Electronics, 10(14), 1619\.

# **APPENDIX A**

**Total budget**

| Description | Quantity | Price | Supplier and Manufacture |
| ----- | ----- | ----- | ----- |
| Orange Pi Zero 2W | 1 | ₱1,359.00 | Orange Pi Official Store |
| Orange Pi Extension | 1 | ₱1,176.00 | Orange Pi Official Store |
| OV5640 USB Camera 5MP and 160° FOV | 2 | ₱1,200.00 | DigiKey Philippines |
| Type-C 5V/3A 18650 Lithium Battery UPS | 1 | ₱121.00 | SAMIORE Philippines |
| Stereo Earphone | 1 | ₱101.00 | Zeus Philippines |
| MPU6050 | 1 | ₱102.00 | BBModule Philippines |
| 18650 Lithium Battery | 2 | ₱70.00 | BBModule Philippines |
| Filament | 2 | ₱715.00 | Makerlab |
| **Total** |  | **₱6,113.00** |  |

# **APPENDIX B**

**Timeline (Gantt Chart)**  
*Gantt Chart of Project Timeline from August to December*

| Task No. | Description | August 2025 |  |  |  | September 2025 |  |  |  |  | October 2025 |  |  |  | November 2025 |  |  |  | December 2025 |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Week |  |  |  | Week |  |  |  |  | Week |  |  |  | Week |  |  |  | Week |  |  |  |  |
|  |  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
| 1 | Identify Problem |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | Brainstorming sessions to generate project ideas |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 | Evaluation of ideas based on feasibility and skills of team |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 | Review of Studies and Literature |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | Definition of objectives and scope of the project |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 | Identification of the hardware and software materials |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 | System Design / Block Diagram |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 | Outline Methodology & System |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 8 | Creation of Conceptual and Theoretical Framework |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 9 | Purchase Materials |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 10 | Program of Zero form factor SBC and other Hardware Components |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 11 | Point Cloud Projection & 3D Mapping Setup |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 12 | AI Model Setup and Fine-Tuning |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 13 | Building Prototype (Phase one – Circuit Prototype) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | Unit Testing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 14 | Writing of Thesis Proposal Manuscript |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 15 | Thesis Proposal Presentation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 16 | Refinement of Project based on panelist feedback |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 17 | Design UI of Web Server |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 18 | 3D Print Hat and Components Case Parts |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 19 | Refinement of Manuscript  based on panelist feedback |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 20 | Ethics Review |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 21 | Letters and validation forms |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 22 | Building Prototype (Phase Two – Device Prototype) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 23 | Calibration of the System |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 24 | Web Application Development |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 25 | Web Application Testing and Debugging |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 26 | Data Preparation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 27 | IoT Sensors Integration with Web Application |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 28 | Debugging & Troubleshooting |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| January \- May |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Task No.** | **Description** | **January 2026** |  |  |  | **February 2026** |  |  |  |  | **March 2026** |  |  |  | **April  2026** |  |  |  | **May 2026** |  |  |  |  |
|  |  | Week |  |  |  | Week |  |  |  |  | Week |  |  |  | Week |  |  |  | Week |  |  |  |  |
|  |  | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 23 | 24 | 25 | 26 | 27 |
| 29 | Refinement of Prototype |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 30 | Ethics Review |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 31 | Implementation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 32 | Interpretation and analysis of the gathered data |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 33 | Finalization of Manuscript for defense |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  34 | Final Defense |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 35 | Revisions based on panelist’s feedback |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36 | Research Approval |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37 | Hardbound |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

# **APPENDIX C**

**Research Proponents and Adviser’s Profile**

|   |  JASON C. D’SOUZA  Address: Monticello, Balabag, Pavia, Iloilo City Phone Number: 09282543420 Email Address: jdsouza@usa.edu.ph  |  |
| :---- | :---- | :---- |
| **EDUCATIONAL BACKGROUND** |  |  |
|  |  |  |
| **College**  | **University of San Agustin** Gen. Luna St., Iloilo City, 5000 2022 – Present  |  |
| **Secondary** | **Central Philippine University** Lopez-Jaena St., Jaro, Iloilo City 2015 – 2021  |  |
| **Elementary** | **St. Jerome Academy** Duenas, Iloilo 2011 \- 2015 |  |

|  |  ETHEL HERNA C. PABITO  Address: Purok 1, Brgy. Igbalangao, Bugasong, Antique, 5704 Phone Number: 09663447178 Email Address: ehpabito@gmail.com  |  |
| :---- | :---- | :---- |
| **EDUCATIONAL BACKGROUND** |  |  |
|  |  |  |
| **College**  | **University of San Agustin** Gen. Luna St., Iloilo City, 5000 2022 – Present  |  |
| **Secondary** | **University of San Agustin**  Senior High School \- STEM Strand Gen. Luna St., Iloilo City, 5000 2019 \- 2022 **Pax Catholic Academy Diocese of Bacolod Inc.** Junior High School CWGC+588, La Paz Street, La Carlota City, Negros Occidental, 6130 2015 \- 2019 |  |
| **Elementary** | **Nelia Boston Maghari Memorial Elementary School** Brgy. Igbalangao, Bugasong, Antique, 5704 2009 \- 2015 |  |

| ![][image2] |  CHENLIN WANG  Present Address: San Pedro, Molo, Iloilo City, Iloilo, 5000 Home Address: TaiYuan City, ShanXi Province, China Phone Number: 09274553340 Email Address: cwang@usa.edu.ph |  |
| :---- | :---- | :---- |
|  |  |  |
| **EDUCATIONAL BACKGROUND** |  |  |
|  |  |  |
| **College**  | **University of San Agustin** Gen. Luna St., Iloilo City, 5000 2022 – Present  |  |
| **Secondary** | **MianYang Church Home Schooling**  Senior High School MianYang City, Sichuan Province, China  2017 \- 2021  **LONGMEN Middle School**  Junior High School TaiYuan City, ShanXi Province, China  2014-2017 |  |
| **Elementary** | **LONGMEN Elementary School**  TaiYuan City, ShanXi Province, China  2008-2014 |  |

| ![][image3] |  VINCE GINNO B. DAYWAN   Address: Sta. Justa, Tibiao, Antique Phone Number: 09991755994 Email Address: Vgbalitao-saan@usa.edu.ph |
| :---- | :---- |
|  |  |
| **EDUCATIONAL BACKGROUND** |  |
|  |  |
| **College**  | **University of San Agustin** Gen. Luna St., Iloilo City, 5000 2020 – Present  |
| **Secondary** | **University of Antique Tario-Lim Memorial Campus Laboratory High School** Poblacion, Tibiao, Antique, Philippines 2014 \- 2020  |
| **Elementary** | **TIBIAO CENTRAL SCHOOL**  72PP+CFX, Poblacion, Tibiao, Antique 2008 \- 2014 |
