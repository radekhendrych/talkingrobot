# talkingrobot
## What
Python app which listens for spoken input, transcribes, interacts with select AI model, receives the answer and reads it back aloud.

## Prerequisites
Version 1.2 and onward assumes having piper module installed.

## Context
Assumes usage of Raspberry Pi 4B with Seeed Respeaker HAT. 
The first version (prior to introduction of release management) assumed that recording of input would be triggered by pushing the button on the HAT, which physically maps to GPIO 11.
Later version use a separate button to trigger the recording (see below).

## Release Notes
### Version 1.0
Unstructured, single file Flask-based python app. Not compliant with SOLID or other well-established software design principles.
Release management not introduced (i.e., no "release-*" branch yet. Just "develop" and "main".
The slightly robotic "espeak-ng" module used for spoken output. Recording triggered based on pushing the button on the Respeaker HAT.

### Version 1.1
Release branching introduced with this branch.
Code refactored to reflect SOLID principles wherever applicable.

### Version 1.2
Switch to using locally installed "piper" module for spoken output.

### Version 1.3
Switch to triggering the recording of spoken input from a separate, Grove-connected button attached to the Respeaker HAT. 
