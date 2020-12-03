# DrumOnsetDetectionCNN


# Installation

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url] -->
[![Forks][forks-shield]][forks-url]
<!-- [![Stargazers][stars-shield]][stars-url] -->
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/jarovaisanen/DrumOnsetDetectionCNN">
    <!--img src="images/logo.png" alt="Logo" width="80" height="80"-->
  </a>

  <h3 align="center">Drum Onset Detection CNN</h3>

  <p align="center">
    Audio data preprocessing and Convolutional Neural Network training algorithm for detecting drum onsets from polyphonic music.
    <br />
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#furtherdevelopment">Further development</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project was implemented as part of my Master's Thesis in University of Jyväskylä,
in the department of Mathematical Information Technology.

https://www.jyu.fi/en/



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites

* Python 3.6+


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/jarovaisanen/DrumOnsetDetectionCNN.git
   ```
2. Install pip packages
   ```sh
   pip install -r requirements.txt
   ```
3. Install <a href="https://developer.nvidia.com/cuda-10.1-download-archive-base" target="_blank">CUDA 10.1</a> for GPU accelerated CNN training:
4. Get the <a href="https://perso.telecom-paristech.fr/grichard/ENST-drums/" target="_blank">ENST Drum database</a> dataset
5. Recommended to setup a <a href="https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/" target="_blank">virtual environment</a>



<!-- USAGE EXAMPLES -->
## Usage

<!--Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources. -->
1. Modify
   [main.py](https://github.com/jarovaisanen/DrumOnsetDetectionCNN/blob/main/src/main.py)
  to suit your needs (Hyperparameters, Paths)
2. Start data pre-processing and training the CNN by running [main.py](https://github.com/jarovaisanen/DrumOnsetDetectionCNN/blob/main/src/main.py)




<!-- furtherdevelopment -->
## Further development

The field is open for further development

1. Fork the project
2. Improve the method



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Jaro Väisänen - jaro.vaisanen@gmail.com

Project Link: [https://github.com/jarovaisanen/DrumOnsetDetectionCNN](https://github.com/jarovaisanen/DrumOnsetDetectionCNN)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

Master's Thesis (in Finnish)
* [Link under construction](https://www.jyu.fi/en/)

Original article
* [Automatic drum transcription with convolutional neural networks](https://hal.archives-ouvertes.fr/hal-02018777/file/DAFx2018_paper_59.pdf)
Céline Jacques, Axel Roebel. Automatic drum transcription with convolutional neural networks.
21th International Conference on Digital Audio Effects, Sep 2018, Aveiro, Portugal, Sep 2018, Aveiro,Portugal. hal-02018777

Code that helped me to get started
* [Bokoch](https://gitlab.at.ispras.ru/angara/uncaptcha/-/blob/master/dataset.py)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/jarovaisanen/DrumOnsetDetectionCNN.svg?style=for-the-badge
[contributors-url]: https://github.com/jarovaisanen/DrumOnsetDetectionCNN/graphs/contributors -->
[forks-shield]: https://img.shields.io/github/forks/jarovaisanen/DrumOnsetDetectionCNN.svg?style=for-the-badge
[forks-url]: https://github.com/jarovaisanen/DrumOnsetDetectionCNN/network/members
[stars-shield]: https://img.shields.io/github/stars/jarovaisanen/DrumOnsetDetectionCNN.svg?style=for-the-badge
[stars-url]: https://github.com/jarovaisanen/DrumOnsetDetectionCNN/stargazers
[issues-shield]: https://img.shields.io/github/issues/jarovaisanen/DrumOnsetDetectionCNN.svg?style=for-the-badge
[issues-url]: https://github.com/jarovaisanen/DrumOnsetDetectionCNN/issues
[license-shield]: https://img.shields.io/github/license/jarovaisanen/DrumOnsetDetectionCNN.svg?style=for-the-badge
[license-url]: https://github.com/jarovaisanen/DrumOnsetDetectionCNN/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/jarovaisanen
