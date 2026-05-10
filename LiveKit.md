An open-source real-time communication platform built on WebRTC, providing infrastructure for adding live audio, video, and data streaming to applications (think video calls, livestreams, voice chat rooms, and increasingly, real-time AI voice agents). 

Core pieces:
- Selective Forwarding Unit (SFU) server: written in Go, self-hostable or used via LiveKit Cloud
- Client SDKs for web, iOS, ANdroid, React Native, Flutter, Unity, etc
- Server SDKs for issuing JWT access tokens and managing rooms/participants.
- Agents framework for building AI participants (e.g. an LLM that joins a room, listens, and speaks back), popular for AI with [[Automatic Speech Recognition|STT]] -> LLM -> [[Text to Speech|TTS]] pipelines.




