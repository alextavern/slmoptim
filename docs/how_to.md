# How to use the project

In this document, one can find all the needed documentation about the different classes, functions and modules of this project, along with simple use case examples. The architecture of the project is detailed in the [README.md](../README.md) file. Here we give some documentation about the content of each module. 

## Initializer

## Utils

### Download.py
#### Class PropheseeCamera

This class is designed to use a Prophesee neuromorphic event based camera. One can find som docmentation about the Python API [here](https://docs.prophesee.ai/stable/get_started/get_started_python.html).

- **`init_cam`**
    
    This function initiate the Prophesee camera with the parameters passed in argument to `__init__`, stored in a `.yaml` file. To do so, we use the _Hal Python Bindings API_ to create and modify the `Device` object (doc [here](https://docs.prophesee.ai/stable/api/python/hal/bindings.html?highlight=get_i_roi)). Several parameters can be set:
    
    - Region of intereste (ROI) : with `roi_off` ((int,int)) for the top left corner of the region positions and `roi_size_x` (int) and `roi_size_x` (int) for the size of this region. If the `roi_size_x` is set to a value different than 0, the ROI is automatically activated. 
    - The accumulation time : the time during which we accumulate events before generating a frame, with the variable `accumulation_time` (int in micro seconds).
    - The biases : they control the parameters of the electronic circuit of the camera (filters, sensitivity ...), see [here](https://docs.prophesee.ai/stable/hw/manuals/biases.html?highlight=biases).

    This function also defines the `RawReader` object which retrieves the stream of events from the camera, and the `OnDemandFrameGenerationAlgorithm` tool to generate frame from a set of events.
    
    We need to limitate the rate of events, otherwise the treatment is too slow and it can even get stuck. An event rate of ~1e7 seems to be a good value to keep enough details while not taking too much time.

- **`close_cam`**

    This function simply deletes the `Device` object.

- **`get`**

    When called, this function generate a frame with the events of the last `accumulation_time` micro secondes. This frame is returned as a numpy array with the size of the camera. If a ROI was defined, the pixels not in this region are set to 0. 

- **`set_accumulation_time`**

    This function allows the user to change the accumulation time used to generate a frame.

#### Class ZeluxCamera

This class is designed to use a Zelux Thorlabs camera with the module `thorlabs_tsi_sdk.tl_camera import TLCameraSDK`. 

- **`init_cam`**
    
    This function initiate the Zelux camera with the parameters passed in argument to `__init__`, stored in the `config_sample.yaml` file in `Docs` folder. Several parameters can be set:
    
    - Region of intereste (ROI) : with `offset` ((int,int)) from the sensor center and `roi_size` (int) for the size of this region of interest (suqare region).
    - The exposure time : `exposure_time`
    - The size of the macro pixel used to reduce computation time: `macropixel`.
    - The gain of the camera: `gain`
    - The time out: `timeout`

- **`close_cam`**

    This function simply deletes close the Zelux camera.

- **`get`**

    When called, this function retrieve the last frame from the camera and output it as a numpy array. 

#### Class RedPitaya

## Optimizer


