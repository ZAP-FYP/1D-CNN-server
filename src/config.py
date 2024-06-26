from decouple import config, UndefinedValueError


class Config:
    def __init__(self):
        try:
            self.test_flag = config("TEST_FLAG", cast=bool)
            self.train_flag = config("TRAIN_FLAG", cast=bool)
            self.full_data_flag = config("FULL_DATA_FLAG", cast=bool)
            self.n_th_frame = config("N_TH_FRAME", cast=bool)
            self.prev_f = config("PREV_FRAMES", cast=int)
            self.future_f = config("FUTURE_FRAMES", cast=int)
            self.start_f = config("START_FUTURE", cast=int)
            self.frame_avg_rate = config("FRAME_AVG_RATE", cast=int)
            self.frame_rate = config("FRAME_RATE", cast=int)
            self.DRR = config("DATA_REDUCTION_RATE", cast=int)
            self.model_name = config("MODEL_NAME")
            self.collision_model_name = config("COLLISION_MODEL_NAME")
            self.dataset_type = config("DATASET_TYPE")
            self.dataset_path = config("DATASET_PATH")
            self.visualization_flag = config("VISUALIZATION_FLAG", cast=bool)
            self.collision_flag = config("COLLISION_FLAG", cast=bool)
            self.patience = config("PATIENCE", cast=int)
            self.pretrained_flag = config("PRETRAINED_FLAG", cast=bool)
            self.filtering_thresold = config("FILTERING_THRESOLD")
            self.custom_loss = config("CUSTOM_LOSS", cast=bool)


        except UndefinedValueError as e:
            raise ValueError(f"Environment variable {e} is not set.")
