

class Config:
    def __init__(self):
        self.lr = 1e-5
        self.batch_size = 30
        self.num_instances = 5
        self.dim = 512
        self.width = 224
        self.origin_width = 256
        self.ratio = 0.16
        self.alpha = 2
        self.beta = 50
        self.k = 16
        self.margin = 0.5
        self.log_name = 'VGG16_V5'
        self.init = 'random'
        self.freeze_BN = True
        self.data = 'cub'
        self.data_root = ''
        self.net = 'VGG16_V5'
        self.epochs = 1000
        self.save_step = 30
        self.resume = None        #pretrained model path
        self.print_freq = 5      #show loss
        self.save_dir = 'saved_model'        #model_save dir
        self.nThreads = 8
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.loss_base = 0.75
        self.loss = 'Binomial'
        self.use_reg = False
        self.pool_feature = False
        self.gallery_eq_query =True
