import torch
import torch.nn.functional as F

DIST_CRITERION_DICT = {
    'mse': F.mse_loss,
    'l1': F.l1_loss,
    'huber': F.smooth_l1_loss,
}
MIN_VAL = 1e-4

def calc_task_loss(logits, targets, reduction='mean', class_weights=None):
    assert len(logits) == len(targets)
    return F.cross_entropy(logits, targets, weight=class_weights, reduction=reduction)

def calc_pos_expl_loss(attrs, rationale, attn_mask, criterion, margin=None, has_rationale=None, attr_scaling=1, attr_algo=None):
    attrs = attr_scaling * attrs
    assert criterion in ['bce', 'kldiv', 'margin', 'mse', 'l1', 'huber', 'order', 'gate']
    max_length = attn_mask.shape[1]
    has_rationale_ = has_rationale.unsqueeze(1).repeat(1, max_length) * attn_mask
    rationale = rationale * has_rationale_
    num_tokens = has_rationale_.sum()

    if attr_algo == 'attention' or criterion == 'kldiv':
        # Transform attrs to distribution
        # attrs[attn_mask == 0] = -float('inf') # This line creates a problem as per -- https://discuss.pytorch.org/t/logbackward-returned-nan-values-in-its-0th-output/92820/9
        attrs = attrs.clone().float()
        attrs[attn_mask == 0] = -1e6
        attrs = F.softmax(attrs, dim=1)

        # Transform rationale to distribution (i.e., scale rationale uniformly by rationale_sum)
        rationale_sum = rationale.sum(dim=1)
        rationale = rationale / rationale_sum[:, None]

    elif criterion in ['mse', 'l1', 'huber', 'order']:
        attrs = F.sigmoid(attrs)

    if criterion == 'bce':
        assert has_rationale is not None
        pos_wt = (num_tokens - rationale.sum()) / rationale.sum()
        loss = F.binary_cross_entropy_with_logits(attrs, rationale, pos_weight=pos_wt, reduction='none') * has_rationale_

    elif criterion == 'kldiv':
        # attrs[attrs == 0] = MIN_VAL
        # attrs *= attn_mask # Edited because of this - https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/4
        attrs = attrs.clone() * attn_mask
        attrs = attrs.clamp(min=MIN_VAL)
        logged_attrs = torch.log(attrs)
        loss = F.kl_div(logged_attrs, rationale, reduction='none', log_target=False) * has_rationale_
 
    elif criterion == 'margin':
        raise NotImplementedError
        margins = attn_mask * margin
        inv_rationale = (1 - rationale) * attn_mask
        loss = (-rationale + inv_rationale) * attrs
        assert not torch.any(torch.isnan(loss))
        loss = torch.maximum(-margins, loss) + margins
        loss = torch.sum(loss) / torch.sum(attn_mask)

    elif criterion == 'mse' or criterion == 'l1' or criterion == 'huber':
        loss = DIST_CRITERION_DICT[criterion](attrs, rationale, reduction='none') * has_rationale_

    elif criterion == 'order':
        max_non_rationale_attr = torch.max((1-rationale)*attrs, dim=1).values.unsqueeze(1).expand(-1, attrs.shape[1])
        ordered_attr = torch.where(rationale==1, torch.sub(torch.div((rationale * attrs), max_non_rationale_attr), torch.tensor(1.0).to(rationale.device)), torch.tensor(0.0).to(rationale.device))
        loss = torch.square(torch.minimum(ordered_attr, torch.zeros(size=ordered_attr.shape).to(rationale.device)))

    elif criterion == 'gate':
        raise NotImplementedError
        max_length = attn_mask.shape[1]
        has_rationale_ = has_rationale.unsqueeze(1).repeat(1, max_length) * attn_mask
        rationale = rationale * has_rationale_
        num_tokens = has_rationale_.sum()
        attrs = F.softmax(attrs, dim=1)
        max_non_rationale_attr = torch.max((1-rationale)*attrs, dim=1).values.unsqueeze(1).expand(-1, attrs.shape[1])
        ordered_attr = torch.where(rationale==1, torch.sub(torch.div((rationale * attrs), max_non_rationale_attr), torch.tensor(1.0).to(rationale.device)), torch.tensor(0.0).to(rationale.device))
        order_loss = torch.sum(torch.square(torch.minimum(ordered_attr, torch.zeros(size=ordered_attr.shape).to(rationale.device))), dim=1)
        bern_probability = torch.add(-torch.sum(torch.where(rationale==1, attrs, torch.tensor(0.0).to(rationale.device)), dim=1),torch.tensor(1.0).to(rationale.device))
        gate_loss = torch.bernoulli(bern_probability) * order_loss
        loss = gate_loss.mean()
        assert not torch.any(torch.isnan(loss))

    assert not torch.any(torch.isnan(loss))
    
    if criterion == 'order':
        loss = torch.sum(loss, dim=1)
        loss = loss.mean()
    else:
        loss = loss.sum()
        if num_tokens > 0:
            loss /= num_tokens
        else:
            assert loss == 0
    assert not torch.isnan(loss)
    
    return loss

def calc_neg_expl_loss(attrs, attn_mask, criterion, targets, preds=None, attr_algo=None):
    if attr_algo == 'attention':
        raise NotImplementedError

    assert criterion in ['l1', 'l1_incorrect']
    batch_size, num_classes, max_length = attrs.shape
    all_classes = torch.arange(num_classes).to(targets.device).unsqueeze(0).expand(batch_size, -1)

    if criterion == 'l1':
        targets_ = targets.unsqueeze(1).expand(-1, num_classes)
        neg_nz = torch.nonzero(all_classes != targets_)
        neg_classes = all_classes[neg_nz[:, 0], neg_nz[:, 1]].reshape(batch_size, -1).unsqueeze(2).expand(-1, -1, max_length)
        neg_attrs = torch.gather(attrs, dim=1, index=neg_classes)

        attn_mask_ = attn_mask.unsqueeze(1).expand(-1, num_classes-1, -1)
        num_tokens = attn_mask_.sum()
        loss = torch.abs(neg_attrs * attn_mask_).sum() / num_tokens

    elif criterion == 'l1_incorrect':
        assert preds is not None

        preds_ = preds.unsqueeze(1).expand(-1, num_classes)
        pred_nz = torch.nonzero(all_classes == preds_)
        pred_classes = all_classes[pred_nz[:, 0], pred_nz[:, 1]].reshape(batch_size, -1).unsqueeze(2).expand(-1, -1, max_length)
        neg_attrs = torch.gather(attrs, dim=1, index=pred_classes).squeeze(1)

        incorrect_mask = (targets != preds).long().unsqueeze(1).expand(-1, max_length) * attn_mask
        num_tokens = incorrect_mask.sum()
        loss = torch.abs(neg_attrs * incorrect_mask).sum()
        if num_tokens != 0:
            loss = loss / num_tokens

    else:
        raise NotImplementedError

    assert not torch.isnan(loss)
    return loss

def calc_comp_loss(comp_logits, comp_targets, task_losses, comp_criterion, topk, comp_margin=None):
    inv_expl_losses = calc_task_loss(comp_logits, comp_targets, reduction='none').reshape(len(topk), -1)
    if comp_criterion == 'diff':
        comp_losses = task_losses - inv_expl_losses
    elif comp_criterion == 'margin':
        assert comp_margin is not None
        comp_margins = comp_margin * torch.ones_like(inv_expl_losses)
        comp_losses = torch.maximum(-comp_margins, task_losses - inv_expl_losses) + comp_margins
    else:
        raise NotImplementedError

    assert not torch.any(torch.isnan(comp_losses))
    return torch.mean(comp_losses, dim=1)

def calc_suff_loss(suff_logits, suff_targets, task_losses, suff_criterion, topk, suff_margin=None, task_logits=None):
    if suff_criterion == 'kldiv':
        assert task_logits is not None
        batch_size = len(task_logits)
        task_distr = F.log_softmax(task_logits, dim=1).unsqueeze(0).expand(len(topk), -1, -1).reshape(len(topk) * batch_size, -1)
        suff_distr = F.softmax(suff_logits, dim=1)
        suff_losses = F.kl_div(task_distr, suff_distr, reduction='none').reshape(len(topk), -1)
    else:
        expl_losses = calc_task_loss(suff_logits, suff_targets, reduction='none').reshape(len(topk), -1)
        if suff_criterion == 'diff':
            suff_losses = expl_losses - task_losses
        elif suff_criterion == 'margin':
            suff_margins = suff_margin * torch.ones_like(expl_losses)
            suff_losses = torch.maximum(-suff_margins, expl_losses - task_losses) + suff_margins
        elif suff_criterion == 'mae':
            suff_losses = F.l1_loss(expl_losses, task_losses, reduction='none')
        elif suff_criterion == 'mse':
            suff_losses = F.mse_loss(expl_losses, task_losses, reduction='none')
        else:
            raise NotImplementedError

    assert not torch.any(torch.isnan(suff_losses))
    return torch.mean(suff_losses, dim=1)
