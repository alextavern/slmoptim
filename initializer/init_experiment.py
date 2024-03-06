from ..utils.download import ZeluxCamera, RedPitaya, PropheseeCamera
from ..utils.upload import SpatialLightModulator


# inspired by this simple illustration of the factory method design pattern in python
#https://medium.com/@vadimpushtaev/python-choosing-subclass-cf5b1b67c696

""" This class is used to initialize the hardware parts ot the experiment.
    In order to make adjustable for different hardware components in the future, 
    a simple class (InitExperiment) is used to instantiate classes with the hardware
    parameters. The user needs only to provide the class and add it manually to the
    mapping. 
    Here, the SLM - using slmpy - the Zelux Thorcam - using the thorlabs SDK and 
    a RedPitaya card are implemented.
    
    This class can be used either using an external configuration yaml file or by 
    using the classmethod decorated create_hardware to create a component individually
    for testing purposes.
    
    e.g. to create a hardware component:
    
    init_experiment = InitExperiment()
    camera = init_experiment.create_hardware('ccd')
    camera.init_cam(**kwargs)
"""

    
class InitExperiment():
    
    def __init__(self, auto=False, **config):
        
            if auto:
            # this for-loop has the dictionary keys hard-coded
            # if there is a change in the config.yaml file
            # these changes must be made here too
                for component in config['hardware']:
                    class_name = config['hardware'][component]['name'] # extract the class names
                    hardware_params = config['hardware'][component]['params'] # extract hardware params
                    new_hardware = self.create_hardware(class_name, **hardware_params)

                    setattr(self, component, new_hardware) # sets the instanitated class as an attrinute of this class

            else: 
                pass
            
    @classmethod
    def create_hardware(cls, hardware_type, **params):
        """ Creates a hardware object by instantiating its corresponding class.
            The possible objects are included into a manually given map. If a new component
            needs to be implemented the user must include it into the map and
            provide its class (it should be located into utils.download/upload) 

        Parameters
        ----------
        hardware_type : string

        Returns
        -------
        instantiated class

        Raises
        ------
        ValueError
            if the hardware name is not included into the mapping
        """
        
        # here is the mapping
        HARDWARE_TYPE_TO_CLASS_MAP = {
            'slmpy': SpatialLightModulator,
            'ccd':  ZeluxCamera,
            'event-based': PropheseeCamera,
            'redpi': RedPitaya,
        }
        
        if hardware_type not in HARDWARE_TYPE_TO_CLASS_MAP:
            raise ValueError('Bad hardware type {}'.format(hardware_type))
        
        print('{} parameters are:'.format(hardware_type))
        print(params)
    
        return HARDWARE_TYPE_TO_CLASS_MAP[hardware_type](**params)
