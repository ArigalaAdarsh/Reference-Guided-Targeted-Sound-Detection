from itertools import zip_longest
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
DEBUG=0

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from convnext.convnext import convnext_tiny


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
 
class conv1d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding='VALID', dilation=1):
        super(conv1d, self).__init__()
        if padding == 'VALID':
            dconv_pad = 0
        elif padding == 'SAME':
            dconv_pad = dilation * ((kernel_size - 1) // 2)
        else:
            raise ValueError("Padding Mode Error!")
        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=dconv_pad)
        self.act = nn.ReLU()
        self.init_layer(self.conv)

    def init_layer(self, layer, nonlinearity='relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        out = self.act(self.conv(x))
        return out
class Fusion(nn.Module):
    def __init__(self, inputdim,n_fac):
        super().__init__()
        self.fuse_layer1 = conv1d(inputdim, inputdim*n_fac,1)
        self.fuse_layer2 = conv1d(inputdim, inputdim*n_fac,1)
        self.avg_pool = nn.AvgPool1d(n_fac, stride=n_fac)  

    def forward(self,embedding,mix_embed):
        embedding = embedding.permute(0,2,1)
        fuse1_out = self.fuse_layer1(embedding)  
        fuse1_out = fuse1_out.permute(0,2,1)

        mix_embed = mix_embed.permute(0,2,1)
        fuse2_out = self.fuse_layer2(mix_embed)  
        fuse2_out = fuse2_out.permute(0,2,1)
        as_embs = torch.mul(fuse1_out, fuse2_out)  
        # (10, 501, 512)
        as_embs = self.avg_pool(as_embs) 
        return as_embs
    
                
class FiLMFusion(nn.Module):
    def __init__(self, inputdim, n_fac):
        super().__init__()
        self.n_fac = n_fac

        # Predict gamma and beta from embedding
        self.gamma_layer = conv1d(inputdim, inputdim * n_fac, 1)
        self.beta_layer  = conv1d(inputdim, inputdim * n_fac, 1)

        # Project mix features
        self.mix_proj = conv1d(inputdim, inputdim * n_fac, 1)

        self.avg_pool = nn.AvgPool1d(n_fac, stride=n_fac)

    def forward(self, embedding, mix_embed):
        # embedding: [B, T, C]
        # mix_embed: [B, T, C]

        embedding = embedding.permute(0, 2, 1)
        mix_embed = mix_embed.permute(0, 2, 1)

        gamma = self.gamma_layer(embedding)  # [B, C*n_fac, T]
        beta  = self.beta_layer(embedding)
        mix   = self.mix_proj(mix_embed)

        # FiLM modulation

        out = gamma * mix + beta

        B, Cn, T = out.shape              # Cn = 768 * n_fac
        C = Cn // self.n_fac              # 768

        out = out.view(B, C, self.n_fac, T)  # (B, 768, 4, T)
        out = out.mean(dim=2)                # (B, 768, T)

        # now permute for GRU
        out = out.permute(0, 2, 1)           # (B, T, 768)
        #print(f'{out.shape=}')
        return out
                
class CDur_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
      
        self.gru = nn.GRU(768, 768, bidirectional=True, batch_first=True)  #768 multiplication  #1536 for concatenation
        self.fusion = Fusion(768,4) #768 for convnext
        #self.fusion = FiLMFusion(768, 4) # This is for FilM Based Fusion
        self.fc = nn.Linear(1536,1536)  #  Birdirectional GRU (786*2)
        
        self.outputlayer = nn.Linear(1536, outputdim)
         
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2)  
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2)  
       
        return decision_time[:,:,0],decision_up
 
 

class Join_fusion(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,load_pretrained=True,**kwargs): # input_dimension -> #mels in URBAN-SED feature extractor , output_dimension -> 2 always
        super().__init__()
        
        self.detection = CDur_fusion(inputdim,outputdim)
        self.AudioEncoder = convnext_tiny(pretrained=False, strict=False, drop_path_rate=0.0, after_stem_dim=[252, 56], use_speed_perturb=False)

        if load_pretrained:
            self.load_pretrained_weights_convnext()
        
        self.in_features = self.AudioEncoder.head_audioset.in_features    
        self.AudioEncoder.head_audioset = nn.Linear(self.in_features, 10)

         

    def load_pretrained_weights_convnext(self):
        state_dict = torch.load('convnext/convnext_tiny_471mAP.pth')
     
        self.AudioEncoder.load_state_dict(state_dict['model'])
    
    def forward(self,x,ref): # x = mixture file rep and ref = ref file rep
        
         
        output_dict = self.AudioEncoder(ref)
        embedding,logit  = output_dict["clipwise_output"], output_dict["clipwise_logits"] 
        batch, time, dim = x.shape
       
        x = self.AudioEncoder.forward_frame_embeddings(x)  # designed as such to give output as (B,768,31) 768 dim embedding, 31 frames
          
         
        x = x.transpose(1, 2).contiguous()   # (b,31,768)
        embedding = embedding.unsqueeze(1)   
      
        embedding = embedding.repeat(1, x.shape[1], 1)  
        
        #x = torch.cat((x, embedding), dim=2)  
        
        # print("Embedding shape:", embedding.shape,"x shape", x.shape)
        x = self.detection.fusion(embedding,x) # fuse the embedding , after fusion shape remains same  
        print('********************************', x.shape)
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) 
        x = self.detection.fc(x)  
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2)  
        
        decision_up = torch.nn.functional.interpolate(   
                decision_time.transpose(1, 2), 
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) 
       
        return decision_time[:,:,0],decision_up,logit 
                   

