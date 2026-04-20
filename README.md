<h1 align="center">Hi there, I'm Roham 👋</h1>

<p align="center">
  <a href="https://github.com/rzninvo">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=4F8CC9&center=true&vCenter=true&width=620&lines=MSc+Artificial+Intelligence+%40+ETH+Z%C3%BCrich+%26+UZH;Computer+Vision+%C2%B7+Robotics+%C2%B7+Embodied+AI;Teaching+machines+to+perceive+and+act+in+the+real+world" alt="Typing SVG" />
  </a>
</p>

<p align="center">
  <a href="mailto:rzendehdel@ethz.ch">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
  <a href="https://www.linkedin.com/in/roham-zendehdel-nobari-6a686a241/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="https://ethz.ch/en.html">
    <img src="https://img.shields.io/badge/ETH%20Z%C3%BCrich-1F407A?style=for-the-badge&logoColor=white" alt="ETH Zürich" />
  </a>
  <a href="https://www.uzh.ch/en.html">
    <img src="https://img.shields.io/badge/UZH-0028A5?style=for-the-badge&logoColor=white" alt="UZH" />
  </a>
  <img src="https://komarev.com/ghpvc/?username=rzninvo&label=Profile%20views&color=4F8CC9&style=for-the-badge" alt="Profile views" />
</p>

---

## 🧑‍🔬 About Me

I'm an MSc student in **Artificial Intelligence** at **ETH Zürich** and the **University of Zürich**, working at the intersection of **computer vision**, **robotics**, and **language-grounded AI**. My research focuses on giving machines a richer understanding of the physical world — from indoor localization through natural language to whole-body imitation learning for humanoid robots.

- 🔭 &nbsp;**Currently** &nbsp;— &nbsp;Graduate Student Researcher at the [ETH Computer Vision and Geometry Group](https://cvg.ethz.ch/), building *HERMES SLAM* (CVPR 2027 target) with Dr. Dániel Béla Baráth; also RL Engineer in the Humanoid Group of the [ETH Robotics Club](https://www.eth-robotics.com/)
- 🌱 &nbsp;**Exploring** &nbsp;— &nbsp;Embodied AI · Vision-Language Models · Sim-to-Real Transfer · 3D Scene Understanding
- 🤝 &nbsp;**Open to collaborate on** &nbsp;— &nbsp;Spatial reasoning, RL for robotics, multimodal perception
- 💬 &nbsp;**Ask me about** &nbsp;— &nbsp;SLAM, scene graphs, indoor localization, humanoid control
- 📫 &nbsp;**Reach me at** &nbsp;— &nbsp;[rzendehdel@ethz.ch](mailto:rzendehdel@ethz.ch)
- 🌍 &nbsp;**Based in** &nbsp;— &nbsp;Zürich, Switzerland 🇨🇭
- 🗣️ &nbsp;**Speaks** &nbsp;— &nbsp;English · Persian · German · Japanese

---

## 🔬 Current Research & Projects

<table>
<tr>
<td colspan="2" valign="top">

### 🛰️ HERMES SLAM &nbsp;<sub>CVPR 2027 (target)</sub>
**ETH D-INFK · Computer Vision and Geometry Group** &nbsp;·&nbsp; *Graduate Student Researcher (Mar 2026 – Present)*

A state-of-the-art **Dynamic SLAM** system developed under the supervision of **Dr. Dániel Béla Baráth** at CVG. Pushing the frontier of real-time localization and mapping in scenes with moving objects — where classical SLAM assumptions break down.

`Visual SLAM` `3D Vision` `Dynamic Scenes` `C++ · Python`

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🧭 LangLoc &nbsp;<sub>ECCV 2026 (under review)</sub>
**ETH CVG Lab × UZH AI/ML Group**

The first pipeline for fine-grained indoor localization from **natural language alone** — no camera required. A dual-branch GATv2 + CLIP architecture for scene retrieval (+8 pp Top-1 over SOTA), a visibility-based floor-grid scoring module reaching ~1 m median error, and a Bayesian dialog module that drives median error down to **7 cm** via targeted yes/no questions.

`PyTorch` `CLIP` `GNNs` `Bayesian Inference`

</td>
<td width="50%" valign="top">

### 🌊 Dark Diversity for eDNA
**ETH Ecosystem & Landscape Evolution Lab**

A Transformer-based **mask modeling** framework (inspired by Pl@ntBERT) on **phylogenetic embeddings** to detect ecologically suitable but absent marine species — quantifying anthropogenic impact at scale. Built end-to-end FASTQ → DADA2 → GBIF pipeline in collaboration with marine ecologists.

`Transformers` `Bioinformatics` `eDNA` `R · Python`

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🤖 Humanoid Whole-Body Imitation
**ETH Robotics Club**

Implementing **TWIST** in NVIDIA **IsaacLab** — training a Unitree G1 locomotion policy on AMASS / MoCap data with robust sim-to-real transfer. Fine-tuning **NVIDIA GR00T N1.6** on teleoperation data and deploying **SONIC** for real-time whole-body control. Also building a low-cost VR-based teleoperation rig with PICO headsets.

`IsaacLab` `RL` `Sim2Real` `Unitree G1`

</td>
<td width="50%" valign="top">

### 🗺️ SpotMap &nbsp;<sub>Boston Dynamics Spot</sub>
**ETH CVG Lab**

Onboard 3D perception for the Spot robot: dense **RGB-D SLAM** with **TSDF** volumetric fusion from a single monocular camera, plus a dynamic **scene graph generator** powered by **OpenMask3D** for open-vocabulary instance segmentation — letting the robot semantically understand and update its world during interaction.

`SLAM` `TSDF` `OpenMask3D` `Scene Graphs`

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🥽 Conversational Indoor Navigation
**Microsoft Spatial AI Lab**

A voice-based navigation assistant for **smart glasses** that guides users with context-aware, **landmark-based** verbal cues instead of metric instructions. Image-based localization, semantic landmark extraction, and mesh-aligned grounding inside Habitat-Sim — evaluated on a 3D mesh of the ETH HG building.

`Habitat-Sim` `VLMs` `Smart Glasses` `Spatial AI`

</td>
<td width="50%" valign="top">

### 🚁 Vision-Based Drone Pursuit
**UZH Robotics & Perception Group**

A **PPO**-based vision policy for an autonomous "camera drone" trained in **Flightmare** / **Agilicious** to track dynamic targets while keeping safe flight dynamics. Deployed on real hardware with a **60 Hz** ROS control loop and extensive domain randomization for robust real-world transfer.

`PPO` `Flightmare` `ROS` `Sim2Real`

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

#### &nbsp;Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![C](https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=black)
![C#](https://img.shields.io/badge/C%23-239120?style=flat-square&logo=csharp&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)

#### &nbsp;Machine Learning & Deep Learning
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

#### &nbsp;Robotics & Simulation
![ROS 2](https://img.shields.io/badge/ROS%202-22314E?style=flat-square&logo=ros&logoColor=white)
![NVIDIA Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-76B900?style=flat-square&logo=nvidia&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-1A73E8?style=flat-square&logoColor=white)
![Unity](https://img.shields.io/badge/Unity-000000?style=flat-square&logo=unity&logoColor=white)
![Habitat-Sim](https://img.shields.io/badge/Habitat--Sim-3B5998?style=flat-square&logoColor=white)

#### &nbsp;Tools & Infrastructure
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white)

---

## 📊 GitHub Stats

<p align="center">
  <img height="180em" src="https://github-readme-stats.vercel.app/api?username=rzninvo&show_icons=true&theme=tokyonight&hide_border=true&include_all_commits=true&count_private=true&rank_icon=github" />
  <img height="180em" src="https://github-readme-stats.vercel.app/api/top-langs/?username=rzninvo&layout=compact&theme=tokyonight&hide_border=true&langs_count=8" />
</p>

<p align="center">
  <img src="https://github-readme-streak-stats.herokuapp.com/?user=rzninvo&theme=tokyonight&hide_border=true" />
</p>

<p align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=rzninvo&theme=tokyo-night&hide_border=true&area=true" />
</p>

<p align="center">
  <img src="https://github-profile-trophy.vercel.app/?username=rzninvo&theme=tokyonight&no-frame=true&no-bg=true&row=1&column=7" />
</p>

---

<p align="center">
  <i>"The best way to predict the future is to invent it." &nbsp;—&nbsp; Alan Kay</i>
</p>
