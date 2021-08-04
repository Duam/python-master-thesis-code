
class Dataset:
    def __init__(self, filename, name="data", data=None):
        self.filename = filename
        self.name = name
        self.data = data if data is not None else []


class carousel_zcm_constants:
    """ === Address === """
    url = "udpm://239.255.76.67:7667?ttl=0"
    """ === Channel names === """
    control_channel = "car_control"
    control_aileron_channel = "VE2_KIN1_SET"
    control_flap_channel = "VE2_KIN2_SET"
    control_rudder_channel = "VE2_KIN3_SET"
    control_elevator_channel = "VE2_KIN4_SET"
    out_roll_channel = "VE1_SCAPULA_ELEVATION"
    out_pitch_channel = "VE1_SCAPULA_ROTATION"
    out_yaw_channel = "CAR_ENCODERPOSITION"
    out_acc_channel = "VE2_MPU_ACC"
    out_gyr_channel = "VE2_MPU_GYRO"
    measurements_sampled_channel = "MEASUREMENTS_sampled"
    controls_sampled_channel = "CONTROLS_sampled"
    """ === File names === """
    path_data_sim = "./data_sim/"
    path_data_phys = "./data_phys/"
    prefix_raw = "RAW_"
    prefix_preprocessed = "PREPROCESSED_"
    prefix_identified = "FINAL_"
    prefix_idle = "IDLE_"
    prefix_live = "LIVE_"
    prefix_virtual = "VIRTUAL_"
    prefix_physical = "PHYSICAL_"
    prefix_identification = "IDENTIFICATION_SET_"
    prefix_validation = "VALIDATION_SET_"

    @staticmethod
    def get_file_prefix(is_virtual:bool, is_identification_data:bool, is_live_experiment:bool, pipeline_step:int):
        """get_file_prefix Returns the prefix of the data file
        Args:
          virtual_experiment[bool] -- Flag: Virtual (simulated) or physical_experiment (real) experiment?
          identification_imu[bool] -- Flag: Identification dataset or validation dataset?
          live_data[bool] -- Flag: Data from a turning or an idling experiment?
          pipeline_step[int] -- Which step in the pipeline does this data come from/head to?
        Returns:
          The file prefix
        """

        prefix = ""

        # Data folder
        if is_virtual:
            prefix += carousel_zcm_constants.path_data_sim
        else:
            prefix += carousel_zcm_constants.path_data_phys

        # Raw/processed/identified
        if pipeline_step == 1:
            prefix += carousel_zcm_constants.prefix_raw
        elif pipeline_step == 2:
            prefix += carousel_zcm_constants.prefix_preprocessed
        elif pipeline_step == 3:
            prefix += carousel_zcm_constants.prefix_identified
        else:
            pass

        # Identification/Validation data
        if is_identification_data:
            prefix += carousel_zcm_constants.prefix_identification
        else:
            prefix += carousel_zcm_constants.prefix_validation

        # Carousel Idle/Running
        if is_live_experiment:
            prefix += carousel_zcm_constants.prefix_live
        else:
            prefix += carousel_zcm_constants.prefix_idle

        # Virtual/Physical
        if is_virtual:
            prefix += carousel_zcm_constants.prefix_virtual
        else:
            prefix += carousel_zcm_constants.prefix_physical

        return prefix
