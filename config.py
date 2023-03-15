import torch

if torch.cuda.is_available():
    print('Cuda enabled device found! - Config')
    device = torch.device('cuda:2')
    print(f'Using {device}')
else:
    print('Cuda enabled device NOT found!')
    device = torch.device('cpu')





padding_idx = 0
embedding_size = 300
hidden_size = 768 # also caption dimension
decoder_n_layers = 1
# cap_encoder_n_layers = 1
# rev_encoder_n_layers = 1
# out_channels = [50, 50, 50]
# kernel_sizes = [2, 3, 4]
bidirectional = False
directions = 2 if bidirectional else 1

sum_birections = False
factor = 1 if sum_birections else 2




# continue_training = True
continue_training = False

# sentence_dim = (factor* hidden_size) + sum(out_channels)
# sentence_dim = factor * hidden_size # no conv used
# sentence_encoder_gru_input_dim = embedding_size

embedding_type = 'not pretrained'
# embedding_type = 'pretrained'

decoder_learning_ratio = 5.0
max_length = 16
batch_size = 4
batch_size_evaluate = 4

# decoder_input_size = 768
mfb_factor_number = 3
mfb_out_dim = 200
mfb_drop = 0.2

epochs = 10
start_epoch = 0
learning_rate = 0.00001
weight_decay = 0.0001


freq_threshold = 2
clip_gradients = True
clip_at = 20

save_counter = 0
save_config = False

exp_lr_decay = 0.99

patience = 2
seed = 3


path_to_dataset = '/home/aseemarora/Documents/Humor/Datasets/datasets/combined'
path_to_dataset = '/Data/asif/Aseem/humor/datasets/combined-2'
# train_file = 'train.csv'
# validation_file = 'val.csv'

# train_file = 'colbert_train.csv'
# validation_file = 'colbert_val.csv'

# train_file = 'humicroedit_train.csv'
# validation_file = 'humicroedit_val.csv'


train_file = 'shortjokes_train.csv'
validation_file = 'shortjokes_val.csv'


# test_file = 'colbert_test.csv'
test_file = 'humicroedit_test.csv'
# test_file = 'reddit_test.csv'
# test_file = 'shortjokes_test.csv'
# test_file = 'puns_test.csv'

path_to_train_folder = '/home/aseemarora/Downloads/Medical-domain-datasets/Path-VQA/Complete/split/split/qas/train'
# train_file = 'train_vqa_2.pkl'
path_to_train_image_features = '/home/aseemarora/Downloads/Medical-domain-datasets/Path-VQA/Complete/split/split/images/train/image-features-vit-50'


path_to_validation_folder = '/home/aseemarora/Downloads/Medical-domain-datasets/Path-VQA/Complete/split/split/qas/train'
# validation_file = 'val_vqa_2.pkl'
path_to_val_image_features = '/home/aseemarora/Downloads/Medical-domain-datasets/Path-VQA/Complete/split/split/images/val/image-features-vit-50'

path_to_test_folder = '/home/aseemarora/Downloads/Medical-domain-datasets/Path-VQA/Complete/split/split/qas/train'
# test_file = 'test_vqa_2.pkl'
path_to_test_image_features = '/home/aseemarora/Downloads/Medical-domain-datasets/Path-VQA/Complete/split/split/images/test/image-features-vit-50'



class Config:
    def __init__(self):
        self.CONTINUE_TRAINING = continue_training
        self.PADDING_IDX = padding_idx
        self.EMBEDDING_SIZE = embedding_size
        self.HIDDEN_SIZE = hidden_size
        self.DECODER_N_LAYERS = decoder_n_layers
        self.START_EPOCH = start_epoch
        # self.CAP_ENCODER_N_LAYERS = cap_encoder_n_layers
        # self.REV_ENCODER_N_LAYERS = rev_encoder_n_layers
        # self.OUT_CHANNELS = out_channels
        # self.KERNEL_SIZES = kernel_sizes
        self.BIDIRECTIONAL = bidirectional
        self.DIRECTIONS = directions
        self.SUM_BIRECTIONS = sum_birections
        self.FACTOR = factor
        # self.SENTENCE_DIM = sentence_dim
        self.EMBEDDING_TYPE = embedding_type
        # self.COMBINE_LEVEL = combine_level
        # self.IMAGE_DIM = image_dim
        self.MFB_FACTOR_NUMBER = mfb_factor_number
        self.MFB_OUT_DIM = mfb_out_dim
        self.MFB_DROP = mfb_drop
        # self.MFB_FLAG = mfb_flag
        self.BATCH_SIZE = batch_size
        self.BATCH_SIZE_EVALUATE = batch_size_evaluate
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        # self.WEIGHT_DECAY = weight_decay
        self.FREQ_THRESHOLD = freq_threshold
        self.CLIP_GRADIENTS = clip_gradients
        self.CLIP_AT = clip_at
        # self.CONV_DROPOUT = conv_dropout
        # self.CONV_P = conv_p
        # self.CAPTION_DROPOUT = caption_dropout
        # self.CAPTION_P = caption_p
        # self.SENTENCE_DROPOUT = sentence_dropout
        # self.SENTENCE_P = sentence_p
        # self.REVIEW_DROPOUT = review_dropout
        # self.REVIEW_P = review_p
        self.PATH_TO_DATASET = path_to_dataset
        self.PATH_TO_TRAIN_FOLDER = path_to_train_folder
        self.TRAIN_FILE = train_file
        self.PATH_TO_VALIDATION_FOLDER = path_to_validation_folder
        self.VALIDATION_FILE = validation_file
        self.PATH_TO_TEST_FOLDER = path_to_test_folder
        self.TEST_FILE = test_file
        self.PATH_TO_TRAIN_IMAGE_FEATURES = path_to_train_image_features
        self.PATH_TO_VAL_IMAGE_FEATURES = path_to_val_image_features
        self.PATH_TO_TEST_IMAGE_FEATURES = path_to_test_image_features
        # self.PHOTOS_DIR = photos_dir
        self.DEVICE = device
        self.SAVE_COUNTER = save_counter
        self.SAVE_CONFIG = save_config
        self.EXP_LR_DECAY = exp_lr_decay
        self.SEED = seed
        self.PATIENCE = patience
        self.MAX_LENGTH = max_length
        self.DECODER_LEARNING_RATIO = decoder_learning_ratio
        # self.DECODER_HIDDEN_SIZE = decoder_hidden_size
        # self.CONCAT_CAPTIONS = concat_captions
        # self.SENTENCE_ENCODER_GRU_INPUT_DIM = sentence_encoder_gru_input_dim
