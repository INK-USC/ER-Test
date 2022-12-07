import os, pickle, warnings
from typing import Optional, List
from timeit import default_timer as timer

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from transformers import AutoModel, AutoTokenizer, AutoConfig

from src.model.base_model import BaseModel
from src.model.mlp import MLP_factory
from src.utils.data import dataset_info
from src.utils.losses import calc_task_loss, calc_pos_expl_loss, calc_neg_expl_loss, calc_comp_loss, calc_suff_loss
from src.utils.metrics import init_best_metrics, init_perf_metrics, calc_preds, calc_comp, calc_suff, calc_aopc, calc_plaus
from src.utils.expl import attr_algos, baseline_required, calc_expl
from src.utils.optim import setup_optimizer_params, setup_scheduler, freeze_layers
from src.utils.logging import log_step_losses, log_step_metrics, log_epoch_losses, log_epoch_metrics

# torch.autograd.set_detect_anomaly(True)

class LanguageModel(BaseModel):
    def __init__(self,
                 arch: str, dataset: str, optimizer: DictConfig, num_classes: int,
                 scheduler: DictConfig, num_freeze_layers: int = 0, freeze_epochs=-1, neg_weight=1,
                 expl_reg: bool = False, expl_reg_freq: int = 1, task_wt: float = None,
                 explainer_type: str = None, attr_algo: str = None,
                 pos_expl_wt: float = 0.0, pos_expl_criterion: str = None, pos_expl_margin: float = None,
                 neg_expl_wt: float = 0.0, neg_expl_criterion: str = None, neg_expl_margin: float = None,
                 ig_steps: int = 3, internal_batch_size: int = None, gradshap_n_samples: int = 3, gradshap_stdevs: float = 0.0,
                 train_topk: List[int] = [1, 5, 10, 20, 50], eval_topk: List[int] = [1, 5, 10, 20, 50],
                 comp_wt: float = 0.0, comp_criterion: str = None, comp_margin: float = None, comp_target: bool = False,
                 suff_wt: float = 0.0, suff_criterion: str = None, suff_margin: float = None, suff_target: bool = False,
                 plaus_wt: float = 0.0,
                 compute_attr: bool = False, save_outputs: bool = False, exp_id: str = None, attr_scaling: float = 1, anno_method:str='instance',data_type:str='original',
                 **kwargs):

        super().__init__()

        self.save_hyperparameters()

        self.arch = arch
        self.dataset = dataset
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.max_length = dataset_info[dataset]['max_length'][arch]

        self.scheduler = scheduler
        self.freeze_epochs = freeze_epochs
        self.neg_weight = neg_weight

        self.expl_reg = expl_reg
        self.expl_reg_freq = expl_reg_freq
        self.task_wt = task_wt

        assert explainer_type in ['attr_algo', 'dlm', 'slm']
        self.explainer_type = explainer_type
        self.attr_algo = attr_algo
        assert attr_algo in list(attr_algos.keys()) + [None]
        self.data_type=data_type
        self.pos_expl_wt = pos_expl_wt
        self.pos_expl_criterion = pos_expl_criterion
        self.pos_expl_margin = pos_expl_margin

        self.neg_expl_wt = neg_expl_wt
        self.neg_expl_criterion = neg_expl_criterion
        self.neg_expl_margin = neg_expl_margin

        self.tokenizer = AutoTokenizer.from_pretrained(arch)

        self.attr_dict = {
            'explainer_type': None,
            'attr_algo': attr_algo, }
        self.compute_attr = compute_attr
        assert attr_algo in list(attr_algos.keys()) + [None]
        self.attr_algo = attr_algo

        config = AutoConfig.from_pretrained(
            arch, output_attentions = attr_algo == 'attention',
        )
        self.task_encoder = AutoModel.from_pretrained(
            arch, config=config
        )
        self.anno_method=anno_method
        self.task_head = nn.Linear(
            self.task_encoder.config.hidden_size,
            num_classes if self.dataset != 'cose' else 1
        )
        self.model_dict = {
            'task_encoder': self.task_encoder,
            'task_head': self.task_head,
        }

        if self.expl_reg or self.compute_attr:

            if self.explainer_type == 'attr_algo':
                self.attr_dict['baseline_required'] = baseline_required[attr_algo]
                self.attr_dict['attr_func'] = attr_algos[attr_algo](self) if attr_algo != 'attention' else attr_algos[attr_algo]
                self.attr_dict['tokenizer'] = AutoTokenizer.from_pretrained(arch)
                if attr_algo == 'integrated-gradients':
                    self.attr_dict['ig_steps'] = ig_steps
                    self.attr_dict['internal_batch_size'] = internal_batch_size
                elif attr_algo == 'gradient-shap':
                    self.attr_dict['gradshap_n_samples'] = gradshap_n_samples
                    self.attr_dict['gradshap_stdevs'] = gradshap_stdevs

                config = AutoConfig.from_pretrained(
                    arch, output_attentions = attr_algo == 'attention',
                )
                self.task_encoder = AutoModel.from_pretrained(
                    arch, config=config
                )
                self.task_head = nn.Linear(
                    self.task_encoder.config.hidden_size,
                    num_classes if self.dataset != 'cose' else 1
                )

                self.model_dict = {
                    'task_encoder': self.task_encoder,
                    'task_head': self.task_head,
                }

            elif self.explainer_type in ['dlm', 'slm']:
                assert attr_algo is None
                assert expl_reg
                assert len(train_topk) > 0 and all([0 < x <= 100 for x in train_topk])
                assert len(eval_topk) > 0 and all([0 < x <= 100 for x in eval_topk])
                assert comp_wt > 0 or suff_wt > 0 or plaus_wt > 0

                self.topk = {'train': train_topk, 'dev': eval_topk, 'test': eval_topk}

                self.comp_wt = comp_wt
                self.comp_criterion = comp_criterion
                self.comp_margin = comp_margin
                self.comp_target = comp_target
                assert comp_criterion in ['diff', 'margin']
                assert comp_margin >= 0

                self.suff_wt = suff_wt
                self.suff_criterion = suff_criterion
                self.suff_margin = suff_margin
                self.suff_target = suff_target
                assert suff_criterion in ['diff', 'margin', 'mae', 'mse', 'kldiv']
                assert suff_margin >= 0

                self.plaus_wt = plaus_wt

                if self.explainer_type == 'dlm':
                    self.task_encoder = AutoModel.from_pretrained(arch)
                    self.expl_encoder = AutoModel.from_pretrained(arch)
                elif self.explainer_type == 'slm':
                    self.encoder = AutoModel.from_pretrained(arch)
                    self.task_encoder = self.encoder
                    self.expl_encoder = self.encoder
                self.task_head = nn.Linear(
                    self.task_encoder.config.hidden_size,
                    num_classes if self.dataset != 'cose' else 1
                )
                self.expl_head = nn.Linear(self.expl_encoder.config.hidden_size, 1)

                self.model_dict = {
                    'task_encoder': self.task_encoder,
                    'task_head': self.task_head,
                    'expl_encoder': self.expl_encoder,
                    'expl_head': self.expl_head,
                }

        self.compute_attr = compute_attr

        self.best_metrics = init_best_metrics()
        self.perf_metrics = init_perf_metrics(num_classes)

        self.register_buffer('empty_tensor', torch.LongTensor([]))
        if num_classes == 2:
            self.register_buffer('class_weights', torch.FloatTensor([neg_weight, 1]))
        else:
            self.class_weights = None

        assert num_freeze_layers >= 0
        if num_freeze_layers > 0:
            freeze_layers(self, num_freeze_layers)

        if save_outputs:
            assert exp_id is not None
        self.save_outputs = save_outputs
        self.exp_id = exp_id
        self.attr_scaling = attr_scaling
        

    def calc_attrs(self, input_ids, attn_mask, targets=None):
        # Compute attrs via grad-based attr algo
        if self.explainer_type == 'attr_algo' and self.attr_algo in attr_algos.keys():

            # If dataset is CoS-E, use zeros as targets
            if self.dataset == 'cose':
                assert targets is None
                targets = torch.zeros(len(input_ids)).long().to(input_ids.device)

            # Compute input embs and baseline embs
            input_embeds, baseline_embeds = self.get_attr_func_inputs(
                input_ids,
                self.attr_dict['baseline_required'],
            )

            # Compute dim-level attrs via attr algo
            if self.attr_dict['attr_algo'] == 'integrated-gradients':
                attrs = self.attr_dict['attr_func'].attribute(
                    inputs=input_embeds.requires_grad_(), baselines=baseline_embeds,
                    target=targets, additional_forward_args=(attn_mask, 'captum'),
                    n_steps=self.attr_dict['ig_steps'], internal_batch_size=self.attr_dict['internal_batch_size'],
                ).float()
            elif self.attr_dict['attr_algo'] == 'gradient-shap':
                attrs = self.attr_dict['attr_func'].attribute(
                    inputs=input_embeds.requires_grad_(), baselines=baseline_embeds,
                    target=targets, additional_forward_args=(attn_mask, 'captum'),
                    n_samples=self.attr_dict['gradshap_n_samples'], stdevs=self.attr_dict['gradshap_stdevs'],
                ).float()
            elif self.attr_dict['attr_algo'] == 'deep-lift':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    attrs = self.attr_dict['attr_func'].attribute(
                        inputs=input_embeds.requires_grad_(), baselines=baseline_embeds,
                        target=targets, additional_forward_args=(attn_mask, 'captum'),
                    ).float()
            elif self.attr_dict['attr_algo'] in ['input-x-gradient', 'saliency']:
                attrs = self.attr_dict['attr_func'].attribute(
                    inputs=input_embeds.requires_grad_(),
                    target=targets, additional_forward_args=(attn_mask, 'captum'),
                ).float()

            # Pool dim-level attrs into token-level attrs
            attrs = torch.sum(attrs, dim=-1)

        # Compute attrs via LM
        elif self.explainer_type in ['dlm', 'slm']:
            attrs = self.forward(input_ids, attn_mask, mode='expl')

        else:
            raise NotImplementedError

        # Mask out attrs for non-pad tokens
        attrs = attrs * attn_mask

        # Make sure no attr scores are NaN
        assert not torch.any(torch.isnan(attrs))

        return attrs

    def get_attr_func_inputs(self, input_ids, baseline_required):
        word_emb_layer = self.task_encoder.embeddings.word_embeddings
        tokenizer = self.attr_dict['tokenizer']
        input_embeds = word_emb_layer(input_ids)
        if baseline_required:
            baseline = torch.full(input_ids.shape, tokenizer.pad_token_id, device=input_ids.device).long()
            baseline[:, 0] = tokenizer.cls_token_id
            sep_token_locs = torch.nonzero(input_ids == tokenizer.sep_token_id)
            baseline[sep_token_locs[:, 0], sep_token_locs[:, 1]] = tokenizer.sep_token_id
            baseline_embeds = word_emb_layer(baseline)
        else:
            baseline_embeds = None
        return input_embeds, baseline_embeds

    def forward(self, inputs, attention_mask, mode='task'):
        assert mode in ['task', 'expl', 'captum']

        if mode == 'task':
            outputs = {}

            enc_outputs = self.task_encoder(input_ids=inputs, attention_mask=attention_mask)
            enc = enc_outputs.pooler_output
            
            logits = self.task_head(enc)
            if self.dataset == 'cose':
                logits = logits.reshape(-1, self.num_classes)
            outputs['logits'] = logits

            if self.attr_dict['attr_algo'] == 'attention':
                 attns = torch.stack(enc_outputs.attentions, dim=1)[:, -1, :, 0, :] # Use CLS token attn from last layer
                 outputs['attns'] = torch.mean(attns, dim=1) # Compute mean attn scores across all heads

            return outputs

        elif mode == 'expl':
            enc = self.expl_encoder(input_ids=inputs, attention_mask=attention_mask).last_hidden_state
            logits = self.expl_head(enc).squeeze(-1)
            if self.dataset == 'cose':
                logits = logits.reshape(-1, self.max_length)
            return logits

        elif mode == 'captum':
            enc = self.task_encoder(inputs_embeds=inputs, attention_mask=attention_mask).pooler_output
            logits = self.task_head(enc)
            return logits

    def attr_forward(self, input_ids, attn_mask):
        if self.attr_dict['attr_algo'] == 'attention':
            return None
        elif self.dataset == 'cose':
            return self.calc_attrs(input_ids, attn_mask)
        else:
            batch_size = input_ids.shape[0]
            input_ids_ = input_ids.unsqueeze(1).expand(-1, self.num_classes, -1).reshape(-1, self.max_length)
            attn_mask_ = attn_mask.unsqueeze(1).expand(-1, self.num_classes, -1).reshape(-1, self.max_length)
            all_classes = torch.arange(self.num_classes).to(input_ids.device).unsqueeze(0).expand(batch_size, -1).flatten()
            return self.calc_attrs(input_ids_, attn_mask_, all_classes).reshape(batch_size, self.num_classes, self.max_length)

    def unirex_forward(self, attrs, input_ids, attn_mask, targets, topk, expl_keys, mode):
        assert mode in ['loss', 'metric']
        batch_size, max_length = input_ids.shape
        if self.dataset == 'cose':
            batch_size = int(batch_size / self.num_classes)

        prev_end = 0
        expls = torch.stack([calc_expl(attrs, k, attn_mask) for k in topk]).reshape(-1, max_length)
        inv_expls = (1 - expls) * attn_mask.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
        inv_expls[:, 0] = 1 # always treat CLS token as positive token

        if 'task' in expl_keys:
            input_ids_expand = input_ids
            attn_mask_expand = attn_mask
            task_start, task_end = prev_end, prev_end + batch_size
            prev_end = task_end

        else:
            attn_mask_expand = self.empty_tensor.clone()
            input_ids_expand = self.empty_tensor.clone()
            prev_end = 0
            
        if 'comp' in expl_keys:
            if mode == 'loss':
                comp_input_ids = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                input_ids_expand = torch.cat((input_ids_expand, comp_input_ids), dim=0)
                attn_mask_expand = torch.cat((attn_mask_expand, inv_expls), dim=0)
            elif mode == 'metric':
                comp_input_ids_ = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                comp_attn_mask_ = attn_mask.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                comp_input_ids, comp_attn_mask = [], []
                for i, cur_inv_expl in enumerate(inv_expls):
                    inv_expls_nonzero = torch.nonzero(cur_inv_expl).flatten()
                    num_pad_tokens = max_length-len(inv_expls_nonzero)
                    comp_input_ids.append(torch.cat((comp_input_ids_[i][inv_expls_nonzero], self.tokenizer.pad_token_id*torch.ones(num_pad_tokens).long().to(input_ids.device))))
                    comp_attn_mask.append(torch.cat((comp_attn_mask_[i][inv_expls_nonzero], 0*torch.ones(num_pad_tokens).long().to(attn_mask.device))))
                comp_input_ids = torch.stack(comp_input_ids)
                comp_attn_mask = torch.stack(comp_attn_mask)
                    
                input_ids_expand = torch.cat((input_ids_expand, comp_input_ids), dim=0)
                attn_mask_expand = torch.cat((attn_mask_expand, comp_attn_mask), dim=0)

            comp_targets = None if targets is None else targets.unsqueeze(0).expand(len(topk), -1).reshape(-1)
            comp_start, comp_end = prev_end, prev_end + batch_size*len(topk)
            prev_end = comp_end

        else:
            comp_targets = None

        if 'suff' in expl_keys:
            if mode == 'loss':
                suff_input_ids = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                input_ids_expand = torch.cat((input_ids_expand, suff_input_ids), dim=0)
                attn_mask_expand = torch.cat((attn_mask_expand, expls), dim=0)
            elif mode == 'metric':
                suff_input_ids_ = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                suff_attn_mask_ = attn_mask.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                suff_input_ids, suff_attn_mask = [], []
                for i, cur_expl in enumerate(expls):
                    expls_nonzero = torch.nonzero(cur_expl).flatten()
                    num_pad_tokens = max_length-len(expls_nonzero)
                    suff_input_ids.append(torch.cat((suff_input_ids_[i][expls_nonzero], self.tokenizer.pad_token_id*torch.ones(num_pad_tokens).long().to(input_ids.device))))
                    suff_attn_mask.append(torch.cat((suff_attn_mask_[i][expls_nonzero], 0*torch.ones(num_pad_tokens).long().to(attn_mask.device))))
                suff_input_ids = torch.stack(suff_input_ids)
                suff_attn_mask = torch.stack(suff_attn_mask)
                    
                input_ids_expand = torch.cat((input_ids_expand, suff_input_ids), dim=0)
                attn_mask_expand = torch.cat((attn_mask_expand, suff_attn_mask), dim=0)

            suff_targets = None if targets is None else targets.unsqueeze(0).expand(len(topk), -1).reshape(-1)
            suff_start, suff_end = prev_end, prev_end + batch_size*len(topk)
            prev_end = suff_end

        else:
            suff_targets = None

        logits_expand = self.forward(input_ids_expand.detach(), attn_mask_expand.detach())['logits']
        task_logits = logits_expand[task_start:task_end, :] if 'task' in expl_keys else None
        comp_logits = logits_expand[comp_start:comp_end, :] if 'comp' in expl_keys else None
        suff_logits = logits_expand[suff_start:suff_end, :] if 'suff' in expl_keys else None
        
        logits_dict = {
            'task': task_logits,
            'comp': comp_logits,
            'suff': suff_logits,
        }

        targets_dict = {
            'task': targets,
            'comp': comp_targets,
            'suff': suff_targets,
        }

        return logits_dict, targets_dict

    def run_step(self, batch, split, batch_idx):
        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        rationale = batch['rationale']
        has_rationale = batch['has_rationale']
        targets = batch['label']
        batch_size = len(input_ids)

        if self.dataset == 'cose':
            input_ids = input_ids.reshape(-1, self.max_length)
            attn_mask = attn_mask.reshape(-1, self.max_length)
            rationale = rationale.reshape(-1, self.max_length)
            has_rationale = has_rationale.reshape(-1)

        eval_split: str = batch['split']
        if split == 'train':
            assert split == eval_split
        ret_dict, loss_dict, metric_dict = {}, {}, {}

        do_expl_reg = self.expl_reg and (batch_idx % self.expl_reg_freq == 0)

        if self.explainer_type == 'attr_algo' and (do_expl_reg or self.compute_attr):

            # Compute attributions for all classes
            attrs = self.attr_forward(input_ids, attn_mask)

            if do_expl_reg:
                # Compute task logits
                outputs = self.forward(input_ids, attn_mask)
                logits = outputs['logits']
                if self.attr_dict['attr_algo'] == 'attention':
                    attrs = outputs['attns']
                    attrs = attrs.unsqueeze(1).expand(-1, self.num_classes, -1)

                # Compute task loss
                task_loss = self.task_wt * calc_task_loss(logits, targets)

                # Initialize expl loss as zero
                expl_loss = torch.tensor(0.0).to(self.device)

                # Compute positive expl loss (w.r.t target class)
                if self.pos_expl_wt > 0:
                    assert self.dataset not in ['amazon', 'yelp']
                    if self.anno_method!='instance':
                        attr_mask = rationale.int()
                    else:
                        attr_mask=attn_mask
                    if self.dataset != 'cose':
                        pos_classes = targets.unsqueeze(1).expand(-1, self.max_length).unsqueeze(1)
                        pos_attrs = torch.gather(attrs, dim=1, index=pos_classes).squeeze(1)
                    pos_expl_loss = self.pos_expl_wt * calc_pos_expl_loss(
                        attrs=pos_attrs if self.dataset != 'cose' else attrs,
                        rationale=rationale,
                        attn_mask=attr_mask,
                        criterion=self.pos_expl_criterion,
                        margin=self.pos_expl_margin,
                        has_rationale=has_rationale,
                        attr_scaling=self.attr_scaling,
                        attr_algo=self.attr_dict['attr_algo'],
                    )
                    expl_loss += pos_expl_loss
                    loss_dict['pos_expl_loss'] = pos_expl_loss
        
                # Compute negative expl loss (w.r.t. non-target classes)
                if self.neg_expl_wt > 0:
                    assert self.dataset not in ['amazon', 'yelp', 'cose']
                    neg_expl_loss = self.neg_expl_wt * calc_neg_expl_loss(
                        attrs=attrs,
                        attn_mask=attn_mask,
                        criterion=self.neg_expl_criterion,
                        targets=targets,
                        preds=calc_preds(logits),
                        attr_algo=self.attr_dict['attr_algo'],
                    )
                    expl_loss += neg_expl_loss
                    loss_dict['neg_expl_loss'] = neg_expl_loss

        # Compute expl loss
        elif self.explainer_type in ['dlm', 'slm'] and (do_expl_reg or self.compute_attr):
            logits = None
            topk = self.topk[eval_split]
            attrs = self.calc_attrs(input_ids, attn_mask)

            if self.anno_method!='instance':
                attr_mask = rationale.int()
            else:
                attr_mask=attn_mask

            # Initialize expl loss as zero
            expl_loss = torch.tensor(0.0).to(self.device)
            if self.comp_wt > 0 or self.suff_wt > 0:

                logits_dict, targets_dict = self.unirex_forward(attrs, input_ids, attn_mask, targets, topk, expl_keys=['task', 'comp', 'suff'], mode='loss')
                task_losses = calc_task_loss(logits_dict['task'], targets_dict['task'], reduction='none', class_weights=self.class_weights)
                logits = logits_dict['task']
                task_loss = self.task_wt * torch.mean(task_losses)
                task_losses = task_losses.unsqueeze(0).expand(len(topk), -1)
                
                if self.comp_wt > 0: # Compute comp loss
                    comp_losses = calc_comp_loss(
                        comp_logits=logits_dict['comp'],
                        comp_targets=targets_dict['comp'],
                        task_losses=task_losses,
                        comp_criterion=self.comp_criterion,
                        topk=topk,
                        comp_margin=self.comp_margin,
                    )
                    comp_loss = self.comp_wt * torch.mean(comp_losses)
                    expl_loss += comp_loss
                    loss_dict['comp_loss'] = comp_loss
                    loss_dict['comp_losses'] = comp_losses
                else:
                    loss_dict['comp_loss'] = torch.tensor(0.0).to(self.device)

                if self.suff_wt > 0: # Compute suff loss
                    suff_losses = calc_suff_loss(
                        suff_logits=logits_dict['suff'],
                        suff_targets=targets_dict['suff'],
                        task_losses=task_losses,
                        suff_criterion=self.suff_criterion,
                        topk=topk,
                        suff_margin=self.suff_margin,
                        task_logits = logits_dict['task'] if self.suff_criterion == 'kldiv' else None,
                    )
                    suff_loss = self.suff_wt * torch.mean(suff_losses)
                    expl_loss += suff_loss
                    loss_dict['suff_loss'] = suff_loss
                    loss_dict['suff_losses'] = suff_losses
                else:
                    loss_dict['suff_loss'] = torch.tensor(0.0).to(self.device)

            else:
                logits = self.forward(input_ids, attn_mask)['logits']
                task_loss = calc_task_loss(logits, targets, class_weights=self.class_weights)

            if self.plaus_wt > 0 and rationale is not None: # Compute plaus loss
                plaus_loss = self.plaus_wt * calc_pos_expl_loss(
                    attrs=attrs,
                    rationale=rationale,
                    attn_mask=attn_mask,
                    criterion=self.pos_expl_criterion,
                    margin=self.pos_expl_margin,
                    has_rationale=has_rationale,
                    attr_scaling=self.attr_scaling,
                    attr_algo=self.attr_dict['attr_algo'],
                )
                expl_loss += plaus_loss
                loss_dict['plaus_loss'] = plaus_loss

            # Compute expl metrics
            with torch.no_grad():
                # Compute preds
                preds = calc_preds(logits)

                # Set models to eval mode
                for model in self.model_dict.values():
                    if model is not None:
                        model.eval()

                attrs = self.calc_attrs(input_ids, attn_mask, preds)
                
                # Perform comp/suff forward pass
                if self.comp_wt > 0 or self.suff_wt > 0:
                    expl_keys = ['task', 'comp', 'suff']
                    logits_dict, _ = self.unirex_forward(attrs, input_ids, attn_mask, None, topk, expl_keys=expl_keys, mode='metric')

            if self.comp_wt > 0 or self.suff_wt > 0:
                comp_logits = logits_dict['comp'].reshape(len(topk), batch_size, self.num_classes)
                metric_dict['comps'] = torch.stack([calc_comp(logits_dict['task'], comp_logits[i], targets, self.comp_target) for i, k in enumerate(topk)])
                metric_dict['comp_aopc'] = calc_aopc(metric_dict['comps'])

                suff_logits = logits_dict['suff'].reshape(len(topk), batch_size, self.num_classes)
                metric_dict['suffs'] = torch.stack([calc_suff(logits_dict['task'], suff_logits[i], targets, self.suff_target) for i, k in enumerate(topk)])
                metric_dict['suff_aopc'] = calc_aopc(metric_dict['suffs'])

            if rationale is not None:
                metric_dict['plaus_auprc'], metric_dict['plaus_token_f1'] = calc_plaus(rationale, attrs, attn_mask, has_rationale)

        else:
            logits = self.forward(input_ids, attn_mask)['logits']
            task_loss = calc_task_loss(logits, targets, class_weights=self.class_weights)
            loss = task_loss

        if do_expl_reg:
            loss_dict['expl_loss'] = expl_loss
            loss = task_loss + expl_loss

        loss_dict['task_loss'] = task_loss
        loss_dict['loss'] = loss

        # Log step losses
        ret_dict = log_step_losses(self, loss_dict, ret_dict, do_expl_reg, eval_split)
        ret_dict['logits'] = logits.detach()
        ret_dict['targets'] = targets.detach()
        ret_dict['eval_split'] = eval_split

        # Log step metrics
        ret_dict = log_step_metrics(self, metric_dict, ret_dict, eval_split)

        # Save attrs
        if self.compute_attr or do_expl_reg:
            ret_dict['attrs'] = attrs.detach()

        return ret_dict

    def aggregate_epoch(self, outputs, split, current_epoch=None):
        if split == 'train':
            splits = ['train']
        elif split == 'dev':
            splits = ['dev', 'test']
        elif split == 'test':
            splits = [outputs[0]['eval_split']]
        outputs_list = outputs if split == 'dev' else [outputs]
        
        for dataset_idx, eval_split in enumerate(splits):
            outputs = outputs_list[dataset_idx]
            log_epoch_losses(self, outputs, eval_split) # Log epoch losses
            log_epoch_metrics(self, outputs, eval_split) # Log epoch metrics
        # Save outputs to file            
        if self.save_outputs:
            out_dir = f'{get_original_cwd()}/../save/{self.exp_id}/model_outputs/{self.dataset}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            keys = ['preds', 'targets', 'logits']
            if self.expl_reg or self.compute_attr:
                keys.append('attrs')
            for dataset_idx, eval_split in enumerate(splits):
                outputs = outputs_list[dataset_idx]
                for key in keys:
                    if key == 'preds':
                        logits = torch.cat([x['logits'] for x in outputs])
                        out_data = calc_preds(logits)
                    else:
                        out_data = torch.cat([x[key] for x in outputs])
                    out_data = out_data.cpu().detach()
                    if current_epoch:
                        out_file = os.path.join(out_dir, f'{eval_split}_epoch_{current_epoch}_{key}.pkl')
                    else:
                        if self.data_type=='original':
                            out_file = os.path.join(out_dir, f'{eval_split}_{key}.pkl')
                        elif self.data_type=='contrast':
                            out_file = os.path.join(out_dir, f'{eval_split}_{key}_contrast.pkl')
                    print(out_file)
                    pickle.dump(out_data.squeeze(), open(out_file, 'wb'))

    def configure_optimizers(self):
        optimizer_params = setup_optimizer_params(self.model_dict, self.optimizer)
        self.optimizer['lr'] = self.optimizer['lr'] * self.trainer.world_size
        optimizer = instantiate(
            self.optimizer, params=optimizer_params,
            _convert_="partial"
        )
        if self.scheduler.lr_scheduler == 'linear_with_warmup':
            scheduler = setup_scheduler(self.scheduler, self.total_steps, optimizer)
            return [optimizer], [scheduler]
        elif self.lr_scheduler == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError