# 🤖 Jarv – Hand-based Keyboard + Mouse Control

<p align="center">
  <a href="https://github.com/hph45/Jarv/actions/workflows/ci.yml?branch=main"><img src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge" alt="CI status"></a>
  <a href="https://github.com/hph45/Jarv/releases"><img src="https://img.shields.io/badge/Release-V2026.2.14-orange?style=for-the-badge" alt="GitHub release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
</p>

**Jarv** is a python program for MacOS computer vision mapping hand gestures to mouse + keyboard control.

## How to run after cloning
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 air_mouse.py
```

Note: you will likely have popups when first run asking for permission to use the camera and whatnot. Accept and run again. I pinky promise that I'm not recording you.

## Mappings
```
Peace sign -------------------->  quit program            (working)
Index and thumb pinch --------->  single click            (working)
Index and middle together ----->  mouse down/drag         (working)
Middle and thumb pinch -------->  double click            (not working rn)
Four finger swipe right/left -->  switch screens          (not working rn)
Pinky finger extended --------->  open typing menu        (planned)
ASL alphabet ------------------>  individual characters   (planned)
Fist -------------------------->  ⌘ + space               (planned)
```
