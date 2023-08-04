# slmOptim
#### Spatial Light Modulator Optimizer
This is python package that contains a collection of modules that serve to solve an optiization problem in experimental optics. 

The goal is to shape the wavefront of a coherent light source that is subsequently coupled to a vibrant nanoresonator and ultimately optimize the measurement of its motion. 

### Requirements
- This library uses the slmPy package (https://github.com/wavefrontshaping/slmPy) written by S. Popoff which handles an SLM as an external monitor. 
- For these experiments a CMOS USB camera purchased from Thorlabs (Zelux CS165MU) is used. This library uses the TLCameraSDK provided by Thorlabs (https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam)

### Modules
- zeluxPy
- patternSLM
- optim