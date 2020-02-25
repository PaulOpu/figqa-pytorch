import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import figqa.utils.sequences as sequences

import torchvision.transforms as transforms

import time

class RelNet(nn.Module):

    def __init__(self, model_args):
        '''
        Implementation of a Relation Network for VQA that includes a basic
        late fusion model and text-only LSTM as special cases.
        '''
        super().__init__()
        self.model_args = model_args
        self.kind = model_args['model']
        if model_args.get('act_f') in [None, 'relu']:
            act_f = nn.ReLU()
        elif model_args['act_f'] == 'elu':
            act_f = nn.ELU()
        self.num_classes = 2
        # question embedding
        self.qembedding = nn.Embedding(model_args['vocab_size'],
                                       model_args['word_embed_dim'])
        self.qlstm = nn.LSTM(model_args['word_embed_dim'],
                             model_args['ques_rnn_hidden_dim'],
                             model_args['ques_num_layers'],
                             batch_first=True, dropout=0)
        ques_dim = model_args['ques_rnn_hidden_dim']
        # text-only classifier
        if self.kind == 'lstm':
            self.qclassifier = nn.Sequential(
                nn.Linear(ques_dim, 512),
                act_f,
                nn.Linear(512, 512),
                nn.Dropout(),
                act_f,
                nn.Linear(512, self.num_classes),
            )
        # image embedding
        if self.kind in ['cnn+lstm', 'rn']:
            img_net_dim = model_args.get('img_net_dim', 64)
            """ self.img_net = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                act_f,
                nn.Conv2d(64, img_net_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(img_net_dim),
                act_f,
                nn.Conv2d(img_net_dim, img_net_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(img_net_dim),
                act_f,
                nn.Conv2d(img_net_dim, img_net_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(img_net_dim),
                act_f,
                nn.Conv2d(img_net_dim, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                act_f,
            ) """

            #Chargrid: Img Net
            self.img_net = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                act_f,
                nn.Conv2d(64, img_net_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(img_net_dim),
                act_f
            )

            # Chargrid: OCR Embedding
            self.chargrid_net = nn.Sequential(
                nn.Conv2d(39, 10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                act_f,
                nn.Conv2d(10, img_net_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(img_net_dim),
                act_f,
                nn.Conv2d(img_net_dim, img_net_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(img_net_dim),
                act_f
            )

            #Chargrid: Img_Net + Chargrid Embedding
            self.entitygrid_net = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                act_f,
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                act_f,
                nn.Conv2d(128, img_net_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(img_net_dim),
                act_f,
            )
            img_net_out_dim = 64            


        # late fusion classifier
        if self.kind == 'cnn+lstm':
            self.cnn_lstm_classifier = nn.Sequential(
                nn.Linear(ques_dim + 8*8*img_net_out_dim, 512),
                act_f,
                nn.Linear(512, 512),
                nn.Dropout(),
                act_f,
                nn.Linear(512, self.num_classes),
            )
        # relation network modules
        if self.kind == 'rn':
            g_in_dim = 2 * (img_net_out_dim + 2) + ques_dim
            # maybe batchnorm
            if model_args.get('rn_bn', False):
                f_act = nn.Sequential(
                    nn.BatchNorm1d(model_args['rn_f_dim']),
                    act_f,
                )
                g_act = nn.Sequential(
                    nn.BatchNorm1d(model_args['rn_g_dim']),
                    act_f,
                )
            else:
                f_act = g_act = act_f
            self.g = nn.Sequential(
                nn.Linear(g_in_dim, model_args['rn_g_dim']),
                g_act,
                nn.Linear(model_args['rn_g_dim'], model_args['rn_g_dim']),
                g_act,
                nn.Linear(model_args['rn_g_dim'], model_args['rn_g_dim']),
                g_act,
                nn.Linear(model_args['rn_g_dim'], model_args['rn_g_dim']),
                g_act,
            )
            self.f = nn.Sequential(
                nn.Linear(model_args['rn_g_dim'], model_args['rn_f_dim']),
                f_act,
                nn.Linear(model_args['rn_f_dim'], model_args['rn_f_dim']),
                f_act,
                nn.Dropout(),
                nn.Linear(model_args['rn_f_dim'], self.num_classes),
            )
            self.loc_feat_cache = {}
        # random init
        self.apply(self.init_parameters)

    @staticmethod
    def init_parameters(mod):
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            #Chargrid: , nonlinearity='relu'
            nn.init.kaiming_uniform(mod.weight, nonlinearity='relu')
            if mod.bias is not None:
                nn.init.constant(mod.bias, 0)

    def img_to_pairs(self, img, ques):
        '''
        Take a small feature map `img` (say 8x8), treating each pixel
        as an object, and return a tensor with one feature
        per pair of objects.

        Arguments:
            img: tensor of size (N, C, H, W) with CNN features of an image
            ques: tensor of size (N, E) containing question embeddings

        Returns:
            Tensor of size (N, num_pairs=HW*HW, feature_dim=2C + E + 2)
        '''
        N, _, H, W = img.size()
        n_objects = H * W
        cells = img.view(N, -1, n_objects)
        # append location features to each object/cell
        loc_feat = self._loc_feat(img)
        cells = torch.cat([cells, loc_feat], dim=1)
        # accumulate pairwise object embeddings
        pairs = []
        three = ques.unsqueeze(2).repeat(1, 1, n_objects)
        for i in range(n_objects):
            one = cells[:, :, i].unsqueeze(2).repeat(1, 1, n_objects)
            two = cells
            # N x C x n_pairs
            i_pairs = torch.cat([one, two, three], dim=1)
            pairs.append(i_pairs)
        pairs = torch.cat(pairs, dim=2)
        result = pairs.transpose(1, 2).contiguous()
        return result

    def _loc_feat(self, img):
        '''
        Efficiently compute a feature specifying the numeric coordinates of
        each object (pair of pixels) in img.
        '''
        N, _, H, W = img.size()
        key = (N, H, W)
        if key not in self.loc_feat_cache:
            # constant features get appended to RN inputs, compute these here
            loc_feat = torch.FloatTensor(N, 2, W**2)
            if img.is_cuda:
                loc_feat = loc_feat.cuda()
            for i in range(W**2):
                loc_feat[:, 0, i] = i // W
                loc_feat[:, 1, i] = i % W
            self.loc_feat_cache[key] = Variable(loc_feat)
        return self.loc_feat_cache[key]

    def forward(self, batch):
        img = batch['img']
        ques_len = batch['question_len']
        ques_emb = self.qembedding(batch['question'])

        #Load Chargrid (chargrid on the fly)
        labels = batch['labels']
        bboxes = batch['bboxes']
        n_label = batch["n_label"]
        #BATCH SIZE
        chargrid = torch.zeros((labels.shape[0],256,256,39),device=torch.get_device(labels))
        #create chargrid on the fly
        #start = time.time()
        for batch_id in range(labels.shape[0]):
            for label_id in range(n_label[batch_id].item()):
                x,y,x2,y2 = bboxes[batch_id,label_id,:]
                chargrid[batch_id,y:y2,x:x2,:] = labels[batch_id,label_id]
        #print(f"chargrid: {time.time()-start:.4f}",)
        chargrid = chargrid.permute(0,3,1,2)

        #Load Chargrid (chargrid created beforehand
        #chargrid = batch['chargrid']


        self.qlstm.flatten_parameters()
        ques = sequences.dynamic_rnn(self.qlstm, ques_emb, ques_len)
        # answer using questions only
        if self.kind == 'lstm':
            scores = self.qclassifier(ques)
            return F.log_softmax(scores, dim=1)
        img = self.img_net(img)
        chargrid = self.chargrid_net(chargrid)
        # answer using questions + images; no relational structure
        if self.kind == 'cnn+lstm':
            ipt = torch.cat([ques, img.view(len(img), -1)], dim=1)
            scores = self.cnn_lstm_classifier(ipt)
            return F.log_softmax(scores, dim=1)
        # RN implementation treating pixels as objects
        # (f and g as in the RN paper)
        assert self.kind == 'rn'

        #Chargrid: Concat img and chargrid and conv
        entitygrid = torch.cat([img,chargrid],dim=1)
        entitygrid = self.entitygrid_net(entitygrid)
        context = 0
        pairs = self.img_to_pairs(entitygrid, ques)
        #pairs = self.img_to_pairs(img, ques)
        N, N_pairs, _ = pairs.size()
        context = self.g(pairs.view(N*N_pairs, -1))
        context = context.view(N, N_pairs, -1).mean(dim=1)
        scores = self.f(context)
        return F.log_softmax(scores, dim=1)
