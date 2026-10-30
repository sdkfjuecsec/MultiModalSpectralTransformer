"""BLIP-style contrastive model pairing ChemBERTa SMILES features with MMT spectra.

Pipeline overview
-----------------
``BLIPModel`` runs two encoders: ``ChembertaFingerprint`` encodes the target
SMILES string, and the pretrained ``MultimodalTransformer`` encodes the spectra.
Both outputs are fed through the same stack of Q-Former query tokens and a
projection head, and the two projected vectors are compared with an
InfoNCE-style contrastive loss.

Contents
--------
BLIP_make
    Loads both pretrained backbones and builds the optimizer.
get_extended_attention_mask
    Local broadcastable-mask helper used by ``ChembertaFingerprint``.
ChembertaFingerprint
    ChemBERTa + pooling + linear fingerprint head.
BLIPModel
    Q-Former fusion of the SMILES and spectra embeddings.
cross_entropy / ProjectionHead
    Contrastive loss and the MLP projection used by both branches.
BLIPMultiGPU
    ``pl.LightningModule`` wrapper.

Legacy note: this file lives under ``_old/``; the active training code is
``models_MMT_v15_4*.py`` + ``TransformerMultiGPU``.
"""

import argparse
import math
import os
import random
import sys
import time
import datetime
import json
import itertools

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import utils_MMT.MT_functions_v15_4 as mtf
import utils_MMT.helper_functions_pl_v15_4 as hf
from utils_MMT.models_MMT_v15_4 import MultimodalTransformer
from utils_MMT.Qformer_v15_4 import BertLMHeadModel
from transformers.models.bert.configuration_bert import BertConfig


from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Adding project-specific path to sys.path
#sys.path.append('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer')
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Build the path to the MultiModalTransformer directory relative to the script's location
base_path = os.path.abspath(os.path.join(script_dir, '../'))

# Add the MultiModalTransformer directory to sys.path
if base_path not in sys.path:
    sys.path.append(base_path)

    
def BLIP_make(config, stoi, stoi_MF, itos):
    """Load the two frozen backbones and build the BLIP model and optimizer.

    Args:
        config: Experiment configuration (``device``, ``model_version``,
            ``loss_fn``, ``CB_model_path``, ``MT_model_path`` and the per-group
            learning rates ``CB_lr`` / ``MT_lr`` / ``Qformer_lr`` /
            ``MT_projection_lr`` / ``weight_decay``).
        stoi (dict): SMILES token -> id mapping (accepted but unused here).
        stoi_MF (dict): Formula token -> id mapping (accepted but unused here).
        itos: SMILES id -> token mapping, handed to ``BLIPModel``.

    Returns:
        tuple: ``(BLIP_model, optimizer)`` where ``optimizer`` is an
        ``AdamW`` over four parameter groups.

    Side effects:
        Downloads / loads the pretrained ChemBERTa (``config.model_version``)
        and its tokenizer, loads the two ``.pth`` checkpoints, and *freezes*
        every parameter of ``CB_model`` and ``MT_model``. This means the
        ``CB_lr`` / ``MT_lr`` groups will produce ``requires_grad`` warnings and
        no updates unless some parameters are unfrozen afterwards.
    """
    # Select device
    device = config.device

    # Load dataloader
    #dataloaders = create_CLIP_dataloaders(config, stoi, stoi_MF)

    # Load Chemberta model
    chemberta = RobertaModel.from_pretrained(config.model_version, output_attentions=True)
    sigmoid = False if config.loss_fn=="BCEWithLogitsLoss" else True
    CB_model = ChembertaFingerprint(chemberta, config, sigmoid)
    CB_state = torch.load(config.CB_model_path)
    CB_model.load_state_dict(CB_state)
    CB_model = CB_model.to(device)

    # Load Chemberta tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.model_version)

    # Load MultimodalTransformer
    MT_model = MultimodalTransformer(config, src_pad_idx=0)

    model_state_dict = torch.load(config.MT_model_path)
    MT_model.load_state_dict(model_state_dict)
    MT_model = MT_model.to(device)


    ################################
    # Freeze parameters in the chemberta model
    for param in CB_model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the self.LL1 layer
    #for param in CB_model.LL1.parameters():
    #    param.requires_grad = True        
    ############################

    # Freeze parameters in the model
    for param in MT_model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the self.fp1 layer
    #for param in MT_model.fp1.parameters():
    #    param.requires_grad = True
        
    # Load CLIP model
    BLIP_model = BLIPModel(CB_model,
                            MT_model,
                            tokenizer,
                            itos,
                            config)

    #loss_meter = AvgMeter()

    # params = [
    #     {"params": CB_model.parameters(), "lr": config.CB_lr},
    #     {"params": MT_model.parameters(), "lr": config.MT_lr},
    #     {"params": itertools.chain(
    #         BLIP_model.CB_projection.parameters(), BLIP_model.MT_projection.parameters()),
    #      "lr": config.MT_projection_lr, "weight_decay": config.weight_decay}]

    params = [
        {"params": CB_model.parameters(), "lr": config.CB_lr},
        {"params": MT_model.parameters(), "lr": config.MT_lr},
        {"params": BLIP_model.Qformer.parameters(), "lr": config.Qformer_lr},
       # {"params": BLIP_model.Qformer_MT.parameters(), "lr": config.Qformer_MT_lr},
        {"params": itertools.chain(
            BLIP_model.CB_projection.parameters(), BLIP_model.MT_projection.parameters()),
        "lr": config.MT_projection_lr, "weight_decay": config.weight_decay}]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)    
        
    return BLIP_model, optimizer 
    #return dataloaders, BLIP_model, tokenizer, optimizer, loss_meter


####################################################################################
################################# ChemBerta ########################################
####################################################################################

def get_extended_attention_mask(attention_mask, input_shape):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

# Global dict used to store activations captured by ``getActivation`` hooks.
activation = {}
def getActivation(name):
    """Build a forward hook that stores the module output under ``name``.

    Args:
        name (str): Key used in the global ``activation`` dict.

    Returns:
        callable: ``hook(model, input, output)`` suitable for
        ``module.register_forward_hook``.

    Side effects:
        The returned hook mutates the module-level ``activation`` dict and
        detaches the captured tensor from the autograd graph.
    """
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class ChembertaFingerprint(nn.Module):
    """ChemBERTa encoder followed by a pooling + linear fingerprint head.

    Args:
        chemberta (RobertaModel): Pretrained ChemBERTa backbone.
        config: Configuration exposing ``fp_dim_reduction``,
            ``num_linear_layers``, ``use_relu``, ``use_dropout``,
            ``num_class`` and ``pca_components``.
        sigmoid (bool): Apply a sigmoid to the fingerprint (used when the loss
            is a plain BCE instead of ``BCEWithLogitsLoss``).
    """

    def __init__(self, chemberta, config, sigmoid=False):
        """Create the head layers and store the hyper-parameters."""
        super(ChembertaFingerprint, self).__init__()
        self.sigmoid = sigmoid
        self.fp_dim_reduction = config.fp_dim_reduction
        self.num_linear_layers = config.num_linear_layers
        self.use_relu = config.use_relu
        self.use_dropout = config.use_dropout
        
        if self.fp_dim_reduction == False:
            self.num_classes = config.num_class
        else:
            self.num_classes = config.pca_components

        self.chemberta = chemberta
        self.get_extended_attention_mask = get_extended_attention_mask
        self.pool = chemberta.pooler
        self.LL1 = nn.Linear(768, self.num_classes )
        
        linear_layers = []
        for i in range(self.num_linear_layers):
            linear_layers.append(nn.Linear(self.num_classes , self.num_classes ))
            if self.use_relu:
                linear_layers.append(nn.ReLU())
            if self.use_dropout:
                linear_layers.append(nn.Dropout(0.2))
        self.linear_layers = nn.Sequential(*linear_layers)
        
        self.classifier = nn.Linear(self.num_classes , self.num_classes )
        self.sig_fn = nn.Sigmoid()  

    def forward(self, x, mask):
        """Encode SMILES ids and return fingerprint, pooled and memory features.

        Args:
            x: ``LongTensor`` of ChemBERTa token ids ``(batch, seq)``.
            mask: Attention mask (1 = attend, 0 = padding).

        Returns:
            tuple: ``(fingerprint, out_1, enc_out)`` where ``fingerprint`` is
            the (optionally sigmoid-activated) prediction, ``out_1`` is the
            detached pooled projection handed to the contrastive branch, and
            ``enc_out`` is the full encoder memory used as Q-Former keys.
        """

        # Generate embeddings
        embeddings = self.chemberta.embeddings(x) 
        # Generate adapted mask
        get_extended_attention_mask = self.get_extended_attention_mask(mask, embeddings.size())
        # Encoder Step
        out = self.chemberta.encoder(embeddings,get_extended_attention_mask) 

        enc_out = out.last_hidden_state

        out = self.chemberta.pooler(out.last_hidden_state)
        # out = self.dropout(out)
        out = self.LL1(out)
        
        out_1 = out.detach()  ### FOR CLIP MODEL TO USE
        
        out = self.linear_layers(out)

        out = self.classifier(out)
        if self.sigmoid:
            out = self.sig_fn(out)
        return out, out_1, enc_out


####################################################################################
################################# BLIP MODEL #######################################
####################################################################################



class BLIPModel(nn.Module):
    """BLIP-style two-tower model with a shared Q-Former.

    Args:
        CB_model (ChembertaFingerprint): Frozen SMILES encoder.
        MT_model (MultimodalTransformer): Frozen spectra encoder.
        tokenizer (RobertaTokenizer): ChemBERTa tokenizer for SMILES strings.
        itos (dict): SMILES id -> token mapping.
        config: Experiment configuration (device, embedding widths,
            ``BLIP_temperature``, ``projection_dim``, dropout).

    The Q-Former is built in ``init_Qformer`` with 32 query tokens, a
    cross-attention width of 768 and a cross-attention block every second layer;
    the query-token parameters are re-initialised from the corresponding
    text-attention weights.
    """

    def __init__(
        self,
        CB_model,
        MT_model,
        tokenizer,
        itos,
        config,
        ):
        """Store the backbones and build the projections, upscale layer and Q-Former."""
        super().__init__()
        self.CB_model = CB_model
        self.MT_model = MT_model
        self.tokenizer = tokenizer
        self.itos = itos
        self.config = config
        self.CB_projection = ProjectionHead(config=config, embedding_type="CB").to(config.device)
        self.MT_projection = ProjectionHead(config=config, embedding_type="MT").to(config.device)
        # Upscale MT model's dimensions to 768
        self.MT_upscale = nn.Linear(128, 768).to(config.device)

        self.temperature = config.BLIP_temperature
        self.Qformer, self.query_tokens = self.init_Qformer(32, #num_query_token
                                                            768, #self.config.MT_embedding, 
                                                            cross_attention_freq=2)
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])         

        
    def forward(self, x):
        """Compute the contrastive loss between the SMILES and spectra branches.

        Args:
            x (dict): Batch dictionary. The modality tensors are taken only for
                modalities listed in ``config.training_mode``.

        Returns:
            tuple: ``(loss.mean(), loss)`` where ``loss`` has one entry per
            batch element and is the average of the text and image directions
            of the InfoNCE loss.

        Side effects:
            Writes ``x['trg_SMI_input']`` into the *caller's* batch dict, i.e.
            it mutates its input.

        NOTE (known issue, left unchanged): the computation ends with a literal
        ``import IPython; IPython.embed();``, so every forward call (training
        and validation alike) stops in an interactive debugger instead of
        returning. In a non-interactive run this raises and kills the process.
        """
        # Getting Image and Text Features
        #smi_input_ids, smi_attention_mask, spectra_padded, spectra_mask = batch
        src_1H = x['src_1H'].to(self.config.device) if "1H" in self.config.training_mode else None
        mask_1H = x['mask_1H'].to(self.config.device) if "1H" in self.config.training_mode else None
        src_13C = x['src_13C'].to(self.config.device) if "13C" in self.config.training_mode else None
        mask_13C = x['mask_13C'].to(self.config.device) if "13C" in self.config.training_mode else None
        src_HSQC = x['src_HSQC'].to(self.config.device) if "HSQC" in self.config.training_mode else None
        mask_HSQC = x['mask_HSQC'].to(self.config.device) if "HSQC" in self.config.training_mode else None
        src_COSY = x['src_COSY'].to(self.config.device) if "COSY" in self.config.training_mode else None
        mask_COSY = x['mask_COSY'].to(self.config.device) if "COSY" in self.config.training_mode else None
        src_IR = x['src_IR'].to(self.config.device) if "IR" in self.config.training_mode else None
        mask_IR = x['mask_IR'].to(self.config.device) if "IR" in self.config.training_mode else None
        src_MF = x['src_MF'].to(self.config.device) if "MF" in self.config.training_mode else None
        mask_MF = x['mask_MF'].to(self.config.device) if "MF" in self.config.training_mode else None
        src_MS = x['src_MS'].to(self.config.device) if "MS" in self.config.training_mode else None
        mask_MS = x['mask_MS'].to(self.config.device) if "MS" in self.config.training_mode else None      
        trg_MW = x['trg_MW'].to(self.config.device) if "MW" in self.config.training_mode else None  
        trg_enc_SMI = x['trg_enc_SMI'].to(self.config.device)
        trg_FP = x['trg_FP'].to(self.config.device)
        
        trg_enc_SMI = trg_enc_SMI.transpose(0, 1)
        trg_SMI_input = trg_enc_SMI[:-1, :] # Remove <EOS> token from target sequence
        x['trg_SMI_input'] = trg_SMI_input
        

        # Prepare data for ChemBerta
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:,:], self.itos)          
        iutput_tokenizer  = self.tokenizer(trg_conv_SMI, return_tensors='pt', padding=True)
        smi_input_ids, smi_attention_mask = iutput_tokenizer['input_ids'], iutput_tokenizer['attention_mask'],
        smi_input_ids, smi_attention_mask = smi_input_ids.to(self.config.device), smi_attention_mask.to(self.config.device)        
        
        CB_pooled, CB_features, CB_memory = self.CB_model(smi_input_ids, smi_attention_mask)
        CB_atts = torch.ones(CB_memory.size()[:-1], dtype=torch.long).to(self.config.device)


        # ChemBerta processing
        CB_query_tokens = self.query_tokens.expand(CB_memory.shape[0], -1, -1)
        CB_query_output = self.Qformer.bert(
            query_embeds=CB_query_tokens,
            encoder_hidden_states=CB_memory,
            encoder_attention_mask=CB_atts,
            use_cache=True,
            return_dict=True
        )
        #CB_embeddings = F.normalize(self.CB_projection(CB_query_output.last_hidden_state[:, 0, :]), dim=-1)
        CB_embeddings = self.CB_projection(CB_query_output.last_hidden_state[:, 0, :])

        MT_memory, MT_embedding_src, MT_src_padding_mask, MT_features = self.MT_model(src_1H, mask_1H, 
                                            src_13C, mask_13C, 
                                            src_HSQC, mask_HSQC, 
                                            src_COSY, mask_COSY, 
                                            src_IR, mask_IR, 
                                            src_MF, mask_MF, 
                                            src_MS, mask_MS, 
                                            trg_MW)
        MT_memory = MT_memory.permute(1, 0, 2)  # Shape adjustment
        MT_memory_upscaled  = self.MT_upscale(MT_memory)  # Upscale to 768 dimensions
        MT_atts = torch.ones(MT_memory.size()[:-1], dtype=torch.long).to(self.config.device)

        MT_query_tokens = self.query_tokens.expand(MT_atts.shape[0], -1, -1)

        MT_query_output  = self.Qformer.bert(
                    query_embeds=MT_query_tokens,
                    encoder_hidden_states=MT_memory_upscaled,
                    encoder_attention_mask=MT_atts,
                    use_cache=True,
                    return_dict=True)

        #MT_embeddings = F.normalize(self.MT_projection(MT_query_output.last_hidden_state[:, 0, :]), dim=-1)
        MT_embeddings = self.MT_projection(MT_query_output.last_hidden_state[:, 0, :])

        # Calculating the Loss
        logits = (CB_embeddings @ MT_embeddings.T) / self.temperature
        MT_similarity = MT_embeddings @ MT_embeddings.T
        CB_similarity = CB_embeddings @ CB_embeddings.T
        targets = F.softmax((MT_similarity + CB_similarity) / 2 * self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        # NOTE (known issue, left unchanged): leftover debugger breakpoint (see
        # the method docstring); it prevents ``loss.mean()`` from ever being
        # returned in a normal (non-interactive) run.
        import IPython; IPython.embed();
        return loss.mean(), loss

    def inference(self, x, gen_enc_SMI_list):
        """Evaluate the contrastive objective on generated SMILES.

        Unlike ``forward`` this runs under ``torch.no_grad()`` and takes the
        generated SMILES as strings.

        Args:
            x (dict): Spectra / target batch, read like in ``forward``.
            gen_enc_SMI_list (list[str]): Generated SMILES to encode with
                ChemBERTa.

        Returns:
            tuple: ``(mean_loss, losses, logits, targets, logits[:, 0])``.

        Side effects:
            Temporarily switches every sub-module to eval mode and unconditionally
            switches them back to train mode before returning (so a caller
            running inference inside validation ends up in train mode).
        """
        # Set all components to evaluation mode
        self.CB_model.eval()
        self.MT_model.eval()
        self.CB_projection.eval()
        self.MT_projection.eval()
        self.Qformer.eval()

        with torch.no_grad():
            # Prepare your data like src_1H, mask_1H, src_13C, etc.
            src_1H = x['src_1H'].to(self.config.device) if "1H" in self.config.training_mode else None
            mask_1H = x['mask_1H'].to(self.config.device) if "1H" in self.config.training_mode else None
            src_13C = x['src_13C'].to(self.config.device) if "13C" in self.config.training_mode else None
            mask_13C = x['mask_13C'].to(self.config.device) if "13C" in self.config.training_mode else None
            src_HSQC = x['src_HSQC'].to(self.config.device) if "HSQC" in self.config.training_mode else None
            mask_HSQC = x['mask_HSQC'].to(self.config.device) if "HSQC" in self.config.training_mode else None
            src_COSY = x['src_COSY'].to(self.config.device) if "COSY" in self.config.training_mode else None
            mask_COSY = x['mask_COSY'].to(self.config.device) if "COSY" in self.config.training_mode else None
            src_IR = x['src_IR'].to(self.config.device) if "IR" in self.config.training_mode else None
            mask_IR = x['mask_IR'].to(self.config.device) if "IR" in self.config.training_mode else None
            src_MF = x['src_MF'].to(self.config.device) if "MF" in self.config.training_mode else None
            mask_MF = x['mask_MF'].to(self.config.device) if "MF" in self.config.training_mode else None
            src_MS = x['src_MS'].to(self.config.device) if "MS" in self.config.training_mode else None
            mask_MS = x['mask_MS'].to(self.config.device) if "MS" in self.config.training_mode else None      
            trg_MW = x['trg_MW'].to(self.config.device) if "MW" in self.config.training_mode else None  
            trg_enc_SMI = x['trg_enc_SMI'].to(self.config.device)
            trg_FP = x['trg_FP'].to(self.config.device)

            # Prepare your ChemBerta data
            input_tokenizer = self.tokenizer(gen_enc_SMI_list, return_tensors='pt', padding=True)
            smi_input_ids, smi_attention_mask = input_tokenizer['input_ids'].to(self.config.device), input_tokenizer['attention_mask'].to(self.config.device)

            CB_pooled, CB_features, CB_memory = self.CB_model(smi_input_ids, smi_attention_mask)
            CB_atts = torch.ones(CB_memory.size()[:-1], dtype=torch.long).to(self.config.device)

            # ChemBerta processing
            CB_query_tokens = self.query_tokens.expand(CB_memory.shape[0], -1, -1)
            CB_query_output = self.Qformer.bert(
                query_embeds=CB_query_tokens,
                encoder_hidden_states=CB_memory,
                encoder_attention_mask=CB_atts,
                use_cache=True,
                return_dict=True
            )
            #CB_embeddings = F.normalize(self.CB_projection(CB_query_output.last_hidden_state[:, 0, :]), dim=-1)
            CB_embeddings = self.CB_projection(CB_query_output.last_hidden_state[:, 0, :])

            # Prepare your MT model data
            # (Assuming x already contains the data for MT model like src_1H, mask_1H, etc.)
            MT_memory, MT_embedding_src, MT_src_padding_mask, MT_features = self.MT_model(src_1H, mask_1H, 
                                                src_13C, mask_13C, 
                                                src_HSQC, mask_HSQC, 
                                                src_COSY, mask_COSY, 
                                                src_IR, mask_IR, 
                                                src_MF, mask_MF, 
                                                src_MS, mask_MS, 
                                                trg_MW)

            MT_memory = MT_memory.permute(1, 0, 2)  # Shape adjustment
            MT_memory_upscaled = self.MT_upscale(MT_memory)  # Upscale to 768 dimensions
            MT_atts = torch.ones(MT_memory.size()[:-1], dtype=torch.long).to(self.config.device)

            MT_query_tokens = self.query_tokens.expand(MT_atts.shape[0], -1, -1)
            MT_query_output = self.Qformer.bert(
                query_embeds=MT_query_tokens,
                encoder_hidden_states=MT_memory_upscaled,
                encoder_attention_mask=MT_atts,
                use_cache=True,
                return_dict=True
            )
            #MT_embeddings = F.normalize(self.MT_projection(MT_query_output.last_hidden_state[:, 0, :]), dim=-1)
            MT_embeddings = self.MT_projection(MT_query_output.last_hidden_state[:, 0, :])

            # Calculating the Loss
            logits = (CB_embeddings @ MT_embeddings.T) / self.temperature
            MT_similarity = MT_embeddings @ MT_embeddings.T
            CB_similarity = CB_embeddings @ CB_embeddings.T
            targets = F.softmax((MT_similarity + CB_similarity) / 2 * self.temperature, dim=-1)
            texts_loss = cross_entropy(logits, targets, reduction='none')
            images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            losses =  (images_loss + texts_loss) / 2.0 
        self.CB_model.train()
        self.MT_model.train()
        self.CB_projection.train()
        self.MT_projection.train()
        self.Qformer.train()
        return  losses.mean(), losses, logits, targets, logits[:,0]



    # def inference(self, x, gen_enc_SMI_list):
    #     # Set all components to evaluation mode
    #     self.CB_model.eval()
    #     self.MT_model.eval()
    #     self.CB_projection.eval()
    #     self.MT_projection.eval()
    #     self.Qformer_CB.eval()  # Set Qformer for ChemBerta to evaluation mode
    #     self.Qformer_MT.eval()  # Set Qformer for MT_model to evaluation mode

        
    #     with torch.no_grad():
    #         # Prepare your data like src_1H, mask_1H, src_13C, etc.
    #         src_1H = x['src_1H'].to(self.config.device) if "1H" in self.config.training_mode else None
    #         mask_1H = x['mask_1H'].to(self.config.device) if "1H" in self.config.training_mode else None
    #         src_13C = x['src_13C'].to(self.config.device) if "13C" in self.config.training_mode else None
    #         mask_13C = x['mask_13C'].to(self.config.device) if "13C" in self.config.training_mode else None
    #         src_HSQC = x['src_HSQC'].to(self.config.device) if "HSQC" in self.config.training_mode else None
    #         mask_HSQC = x['mask_HSQC'].to(self.config.device) if "HSQC" in self.config.training_mode else None
    #         src_COSY = x['src_COSY'].to(self.config.device) if "COSY" in self.config.training_mode else None
    #         mask_COSY = x['mask_COSY'].to(self.config.device) if "COSY" in self.config.training_mode else None
    #         src_IR = x['src_IR'].to(self.config.device) if "IR" in self.config.training_mode else None
    #         mask_IR = x['mask_IR'].to(self.config.device) if "IR" in self.config.training_mode else None
    #         src_MF = x['src_MF'].to(self.config.device) if "MF" in self.config.training_mode else None
    #         mask_MF = x['mask_MF'].to(self.config.device) if "MF" in self.config.training_mode else None
    #         src_MS = x['src_MS'].to(self.config.device) if "MS" in self.config.training_mode else None
    #         mask_MS = x['mask_MS'].to(self.config.device) if "MS" in self.config.training_mode else None      
    #         trg_MW = x['trg_MW'].to(self.config.device) if "MW" in self.config.training_mode else None  
    #         trg_enc_SMI = x['trg_enc_SMI'].to(self.config.device)
    #         trg_FP = x['trg_FP'].to(self.config.device)

    #         # Prepare data for ChemBerta
    #         input_tokenizer = self.tokenizer(gen_enc_SMI_list, return_tensors='pt', padding=True)
    #         smi_input_ids, smi_attention_mask = input_tokenizer['input_ids'].to(self.config.device), input_tokenizer['attention_mask'].to(self.config.device)

    #         CB_pooled, CB_features, CB_memory = self.CB_model(smi_input_ids, smi_attention_mask)
    #         CB_atts = torch.ones(CB_memory.size()[:-1], dtype=torch.long).to(self.config.device)
    #         CB_query_tokens = self.query_tokens_CB.expand(CB_atts.shape[0], -1, -1)

    #         CB_query_output = self.Qformer_CB.bert(
    #             query_embeds=CB_query_tokens,
    #             encoder_hidden_states=CB_memory,
    #             encoder_attention_mask=CB_atts,
    #             use_cache=True,
    #             return_dict=True)

    #         CB_embeddings = F.normalize(self.CB_projection(CB_query_output.last_hidden_state[:, 0, :]), dim=-1)

    #         # MT_model processing
    #         MT_memory, MT_embedding_src, MT_src_padding_mask, MT_features = self.MT_model(src_1H, mask_1H, 
    #                                                                                     src_13C, mask_13C, 
    #                                                                                     src_HSQC, mask_HSQC, 
    #                                                                                     src_COSY, mask_COSY, 
    #                                                                                     src_IR, mask_IR, 
    #                                                                                     src_MF, mask_MF, 
    #                                                                                     src_MS, mask_MS, 
    #                                                                                     trg_MW)
    #         MT_memory = MT_memory.permute(1, 0, 2)
    #         MT_atts = torch.ones(MT_memory.size()[:-1], dtype=torch.long).to(self.config.device)
    #         MT_query_tokens = self.query_tokens_MT.expand(MT_atts.shape[0], -1, -1)

    #         MT_query_output = self.Qformer_MT.bert(
    #             query_embeds=MT_query_tokens,
    #             encoder_hidden_states=MT_memory,
    #             encoder_attention_mask=MT_atts,
    #             use_cache=True,
    #             return_dict=True)

    #         MT_embeddings = F.normalize(self.MT_projection(MT_query_output.last_hidden_state[:, 0, :]), dim=-1)

    #         # Calculating the Loss
    #         logits = (CB_embeddings @ MT_embeddings.T) / self.temperature
    #         MT_similarity = MT_embeddings @ MT_embeddings.T
    #         CB_similarity = CB_embeddings @ CB_embeddings.T
    #         targets = F.softmax((MT_similarity + CB_similarity) / 2 * self.temperature, dim=-1)
    #         texts_loss = cross_entropy(logits, targets, reduction='none')
    #         images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    #         losses =  (images_loss + texts_loss) / 2.0 
    #     self.CB_model.train()
    #     self.MT_model.train()
    #     self.CB_projection.train()
    #     self.MT_projection.train()
    #     return  losses.mean(), losses, logits, targets, logits[:,0]
    
    def init_Qformer(self, num_query_token, encoder_width, cross_attention_freq=2):
        """Build a pretrained BERT language model adapted as a Q-Former.

        Args:
            num_query_token (int): Number of learnable query tokens.
            encoder_width (int): Hidden size of the cross-attended memory.
            cross_attention_freq (int): Insert one cross-attention block every
                this many layers.

        Returns:
            tuple: ``(Qformer, query_tokens)`` where ``Qformer`` is a
            ``BertLMHeadModel`` and ``query_tokens`` has shape
            ``(1, num_query_token, hidden_size)``.

        Side effects:
            Downloads / loads ``bert-base-uncased`` (config and weights).
        """
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = encoder_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


    ######################generate me the cosine similarity of every  output to every input..

def cross_entropy(preds, targets, reduction='none'):
    """Soft-target cross entropy used by the contrastive loss.

    Args:
        preds: Logits ``(batch, batch)``.
        targets: Soft targets of the same shape (rows sum to one).
        reduction (str): ``"none"`` returns the per-row losses, ``"mean"``
            their average.

    Returns:
        Tensor | None: The requested reduction, or ``None`` for any other value.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


####################################################################################
############################## Projection Head #####################################
####################################################################################

class ProjectionHead(nn.Module):
    """Residual MLP head projecting an embedding to ``projection_dim``.

    Args:
        config: Configuration exposing ``CB_embedding``, ``MT_embedding``,
            ``projection_dim`` and ``dropout``.
        embedding_type (str): ``"CB"`` or ``"MT"``; selects the input width.

    NOTE (known issue, left unchanged): the default value is the *list*
    ``["CB", "MT"]``, which matches neither branch, so ``self.embedding_dim``
    is never assigned and construction fails with ``AttributeError`` unless
    the caller always passes a string.
    """

    def __init__(self, config, embedding_type=["CB","MT"]):
        """Select the input width and build the residual MLP."""
        super().__init__()
        if embedding_type=="CB":
            self.embedding_dim = config.CB_embedding
        if embedding_type=="MT":
            self.embedding_dim = config.MT_embedding
        self.projection_dim = config.projection_dim
        self.dropout = config.dropout
        
        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)
    
    def forward(self, x):
        """Apply projection -> GELU -> fc -> dropout with a residual + LayerNorm."""
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


####################################################################################
############################# Multi GPU CLIP ####################################
####################################################################################



class BLIPMultiGPU(pl.LightningModule):
    """Lightning wrapper around :class:`BLIPModel`.

    Args:
        config: Experiment configuration. Reads ``itos_path``, ``stoi_path``,
            ``stoi_MF_path``, ``BLIP_model_save_dir`` and
            ``model_save_interval``, and forwards the rest to ``BLIP_make``.
    """

    def __init__(self, config):
        """Load the vocabularies, build ``BLIPModel`` and store the optimizer."""
        super(BLIPMultiGPU, self).__init__()
        self.start_time = time.time()
        self.config=config
        with open(config.itos_path, 'r') as f:
            itos = json.load(f)
        with open(config.stoi_path, 'r') as f:
            stoi = json.load(f)
        with open(config.stoi_MF_path, 'r') as f:
            stoi_MF = json.load(f)        
        self.BLIP_model, self.optimizer = BLIP_make(config, stoi, stoi_MF, itos)
            
    
    def forward(self, x):
        """Return the scalar contrastive loss for the batch."""
        loss, losses = self.BLIP_model(x)
        return loss

    
    def training_step(self, batch, batch_idx):
        """Run one training step and log the scalar loss.

        Args:
            batch (dict): Batch dictionary.
            batch_idx (int): Index of the current batch (unused).

        Returns:
            dict: ``{'loss': loss, 'log': {'train_loss': loss}}``.

        Side effects:
            Pushes ``train_loss`` straight to the experiment logger
            (Weights & Biases).
        """
        loss = self.forward(batch)
        #### update this part
        tensorboard_logs = {'train_loss': loss}   
        self.logger.experiment.log({'train_loss': loss})   ### test_loss
        return {'loss': loss, 'log': tensorboard_logs}

    
    def validation_step(self, batch, batch_idx):
        """Run one validation step and log the scalar loss under ``test_loss``.

        Args:
            batch (dict): Batch dictionary.
            batch_idx (int): Index of the current batch (unused).

        Returns:
            dict: ``{'loss': loss, 'log': {'test_loss': loss}}``.
        """
        loss = self.forward(batch)
        tensorboard_logs = {'test_loss': loss}   
        self.logger.experiment.log({'test_loss': loss})   ### test_loss
        return {'loss': loss, 'log': tensorboard_logs}


    def on_epoch_end(self):
        """Save a ``.ckpt`` and a raw ``.pth`` state dict at the end of each epoch.

        Side effects:
            Writes ``MultimodalBLIP_Epoch_{epoch}_Loss{loss}.pth`` and the
            matching ``.ckpt`` into ``config.BLIP_model_save_dir``.

        NOTE (known issue, left unchanged): ``train_loss`` is only pushed
        directly to the experiment logger (``self.logger.experiment.log``) and
        never through ``self.log``, so ``trainer.callback_metrics`` usually has
        no such key. ``loss`` is then ``None`` and ``f'{loss:.3f}'`` raises
        ``TypeError`` at the end of the first epoch.
        """
        epoch = self.current_epoch
        loss = self.trainer.callback_metrics.get('train_loss', None)  
        save_path = os.path.join(self.config.BLIP_model_save_dir, f"MultimodalBLIP_Epoch_{epoch}_Loss{loss:.3f}.pth")
        ckp_save_path = os.path.join(self.config.BLIP_model_save_dir, f"MultimodalBLIP_Epoch_{epoch}_Loss{loss:.3f}.ckpt")
        self.trainer.save_checkpoint(ckp_save_path)
        torch.save(self.BLIP_model.state_dict(), save_path)
        
    def on_batch_end(self):
        """Save a checkpoint whenever ``model_save_interval`` seconds elapsed.

        Side effects:
            Writes a timestamped ``.pth`` state dict and ``.ckpt`` into
            ``config.BLIP_model_save_dir``, prints the path and resets the
            timer. Same ``callback_metrics['train_loss']`` caveat as
            ``on_epoch_end`` applies to the loss formatting.
        """
        elapsed_time = time.time() - self.start_time
        time_obj = datetime.timedelta(seconds=elapsed_time)
        time_str = str(time_obj).split('.')[0]  # This will give you a string formatted as 'H:M:S'


        if elapsed_time  >= self.config.model_save_interval:  
            epoch = self.current_epoch
            loss = self.trainer.callback_metrics.get('train_loss', None)  
            save_path = os.path.join(self.config.BLIP_model_save_dir, f"MultimodalBLIP__time_{str(time.time())}_Loss_{loss:.3f}.pth")
            ckp_save_path = os.path.join(self.config.BLIP_model_save_dir, f"MultimodalBLIP__time_{str(time.time())}_Loss_{loss:.3f}.ckpt")
            torch.save(self.BLIP_model.state_dict(), save_path)
            self.trainer.save_checkpoint(ckp_save_path)
            print(f'Model saved at {save_path}')
            self.start_time = time.time()
    
    def configure_optimizers(self):
        """Return the pre-built optimizer plus a reduce-on-plateau scheduler.

        Returns:
            dict: Lightning's optimizer configuration; the scheduler monitors
            ``'train_loss'`` with mode ``min``.

        NOTE (known issue, left unchanged): the monitored key ``'train_loss'``
        is produced by ``training_step`` only in the legacy ``'log'`` dict, not
        via ``self.log``, so the scheduler may never receive a metric and never
        reduce the learning rate.
        """
        optimizer = self.optimizer #torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
            }
        }

# from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

#     def configure_optimizers(self):
#         optimizer = self.optimizer  # Assuming this is defined as torch.optim.AdamW or similar
#         num_warmup_steps = self.config.num_warmup_steps  # Set this in your configuration
#         total_steps = self.config.total_steps  # Set total training steps in your configuration

#         # Warmup scheduler
#         warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(step / num_warmup_steps, 1.0))

#         # ReduceLROnPlateau scheduler
#         plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

#         # Scheduler sequence
#         scheduler_sequence = {
#             'scheduler': warmup_scheduler,
#             'interval': 'step',  # Assuming step-wise updates; change to 'epoch' if otherwise
#             'total_steps': total_steps
#         }

#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': scheduler_sequence,
#                 'plateau_scheduler': plateau_scheduler,
#                 'monitor': 'train_loss',  # Monitoring for plateau_scheduler
#             }
#         }