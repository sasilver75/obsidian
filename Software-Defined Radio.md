---
aliases:
  - SDR
---
Radio where much of the signal processing is done in software, instead of using fixed analog hardware.
- In a traditional radio, we use dedicated circuits for tuning, filtering, [[Demodulation]], decoding, etc. 
- An SDR uses radio front-end hardware to receive or transmit RF signals, then converts them to digital samples so that software can process them.

In practice, ==SDR lets you inspect or work with many signal types using the same device==:
- Receive ADS-B aircraft signals
- Listen to FM radio
- Inspect drone control/video bands
- Analyze spectrum activity
- Decode weather satellite signals
- Prototype radios and waveforms

SDR is flexible, but needs enough compute, sampling bandwidth, and a good RF front end. Cheap SDRs are excellent learning tools, but limited in frequency range, sensitivity, transmit capability, and bandwidth.


