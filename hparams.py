import argparse

class HParams(object):
    def __init__(self):
        self.dataset_path = '../media/bach3/seungheon/Music_Emotion_Recognition/7Mid_5000song/audio'
        self.feature_path= '../media/bach3/seungheon/Music_Emotion_Recognition/7Mid_5000song/feature'
        self.mel_feature_path= '../media/bach3/seungheon/Music_Emotion_Recognition/7Mid_5000song/mel_feature'
        self.meta5000 = '../media/bach3/seungheon/Music_Emotion_Recognition/7Mid_5000song/metadata_annotations'
        self.dataset_path_360 = '../media/bach3/seungheon/Music_Emotion_Recognition/SoundTrack/set1/mp3/Soundtrack360_mp3'
        self.label_path_360 = '../media/bach3/seungheon/Music_Emotion_Recognition/SoundTrack/set1/mean_ratings_set1.xls'
        self.stimulus_path_360 = '../media/bach3/seungheon/Music_Emotion_Recognition/SoundTrack/set1/stimulus_set_1.csv'
        self.annotations = '../media/bach3/seungheon/Music_Emotion_Recognition/annotations'
        self.mid_feature = ['melody','articulation','rhythm_complexity','rhythm_stability','dissonance','atonality','mode']
        self.emotion_label = ['valence','energy','tension','anger','fear','happy','sad','tender']
        # Feature Parameters
        self.sample_rate=22050
        self.fft_size = 2048
        self.win_size = 1024
        self.hop_size = 706
        self.num_mels = 149
        self.feature_length = 313

        # Training Parameters
        self.device = [1,2]  # 0: CPU, 1: GPU0, 2: GPU1, ...
        self.batch_size = 4
        self.num_epochs = 1000
        self.learning_rate = 1e-2
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-6
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 5

    # Function for pasing argument and set hParams
    def parse_argument(self, print_argument=True):
        parser = argparse.ArgumentParser()
        for var in vars(self):
            value = getattr(hparams, var)
            argument = '--' + var
            parser.add_argument(argument, type=type(value), default=value)

        args = parser.parse_args()
        for var in vars(self):
            setattr(hparams, var, getattr(args,var))

        if print_argument:
            print('----------------------')
            print('Hyper Paarameter Settings')
            print('----------------------')
            for var in vars(self):
                value = getattr(hparams, var)
                print(var + ":" + str(value))
            print('----------------------')

hparams = HParams()
hparams.parse_argument()