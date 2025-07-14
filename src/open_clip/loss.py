import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features
def gather_features_da(
        image_features,
        text_features,
        valid_caption_mask,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_valid_caption_mask=torch.cat(torch.distributed.nn.all_gather(valid_caption_mask), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            gathered_valid_caption_mask = [torch.zeros_like(valid_caption_mask) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_valid_caption_mask, valid_caption_mask)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                gathered_valid_caption_mask[rank] = valid_caption_mask
                
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            all_valid_caption_mask = torch.cat(gathered_valid_caption_mask, dim=0)

    return all_image_features, all_text_features, all_valid_caption_mask

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text[:len(logits_per_image)], labels)
            ) / 2
        return total_loss


class Clip_DA_all_hn_far_Loss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            cmr_loss=False,
            imc_loss=False,
            hndc_loss=False,
            hardnegative=False,
            imc_loss_weight=0.2,
            cmr_loss_weight=0.2,
            hndc_loss_weight=0.2,
            threshold_type='mean',
          
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.cmr_loss=cmr_loss
        self.imc_loss=imc_loss
        self.hndc_loss = hndc_loss
        
        self.imc_loss_weight=imc_loss_weight
        self.cmr_loss_weight=cmr_loss_weight
        self.hndc_loss_weight=hndc_loss
        self.threshold_type=threshold_type
        self.hardnegative=hardnegative
        
    def forward(self, image_features, text_features, valid_caption_mask, logit_scale, thresholds):
        if isinstance(logit_scale, tuple):
            raise NotImplementedError("siglip not supported")

        device = image_features.device
        total_loss, itc_loss, cmr_loss, imc_loss, hndc_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        if self.world_size > 1:
            all_image_features, all_text_features, all_valid_caption_mask = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )
            caption_types = torch.tensor(
                ([1] * image_features.shape[0] + [2] * image_features.shape[0] * 4) * self.world_size,
                device=device
            )

            gt_all_text_features = all_text_features[caption_types == 1] #gt text
            da_all_text_features = all_text_features[caption_types == 2] #hard negative text
            gt_len, feature_size = all_image_features.shape[0], all_image_features.shape[-1]

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                if self.hardnegative:
                    all_text_features = torch.cat([gt_all_text_features, da_all_text_features]) #gt와 hn concat
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T #image 당 logit 계산 (gt & hn)
    
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T

                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T

                if self.cmr_loss:
                    da_logits_per_image = logit_scale * (
                        da_all_text_features.reshape(gt_len, -1, feature_size)
                        @ all_image_features.unsqueeze(-1)
                    ).squeeze() * all_valid_caption_mask
                    cmr_loss, thresholds = self.get_cmr_loss(
                        logits_per_image, da_logits_per_image, all_valid_caption_mask, thresholds
                    )

                if self.imc_loss:
                    text_embedding_matrix = logit_scale * gt_all_text_features @ da_all_text_features.T
                    imc_loss = self.get_imc_loss(logits_per_image, text_embedding_matrix)
                if self.hndc_loss:
                    hndc_loss = self.get_hndc_loss(da_text_features=da_all_text_features, temperature=0.07)
                
        else:
            gt_len, feature_size = image_features.shape[0], image_features.shape[-1]
            gt_text_features = text_features[:gt_len]
            da_text_features = text_features[gt_len:]
            all_text_features = torch.cat([gt_text_features, da_text_features])

            if self.hardnegative:
                logits_per_image = logit_scale * image_features @ all_text_features.T
            else:
                logits_per_image = logit_scale * image_features @ gt_text_features.T

            logits_per_text = logit_scale * gt_text_features @ image_features.T

            if self.cmr_loss:
                da_logits_per_image = logit_scale * (
                    da_text_features.reshape(gt_len, -1, feature_size)
                    @ image_features.unsqueeze(-1)
                ).squeeze() * valid_caption_mask
                cmr_loss, thresholds = self.get_cmr_loss(
                    logits_per_image, da_logits_per_image, valid_caption_mask, thresholds
                )

            if self.imc_loss:
                text_embedding_matrix = logit_scale * gt_text_features @ da_text_features.T
                imc_loss = self.get_imc_loss(logits_per_image, text_embedding_matrix)
            
            if self.hndc_loss :
                hndc_loss = self.get_hndc_loss(da_text_features=da_text_features, temperature=0.07)
                

        # label setup
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        itc_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        total_loss += itc_loss
        if self.cmr_loss:
            total_loss += cmr_loss * self.cmr_loss_weight
        if self.imc_loss:
            total_loss += imc_loss * self.imc_loss_weight
        if self.hndc_loss:
            total_loss += hndc_loss * self.hndc_loss_weight
        # return total_loss, thresholds, itc_loss, cmr_loss, imc_loss, hndc_loss   
        return (
            total_loss,
            thresholds if self.cmr_loss else None,
            itc_loss,
            cmr_loss if self.cmr_loss else None,
            imc_loss if self.imc_loss else None,
            hndc_loss if self.hndc_loss else None
        )  
    def get_cmr_loss(self,gt_logits_per_image:torch.Tensor,da_logits_per_image:torch.Tensor,valid_caption_mask,thresholds:torch.Tensor) -> torch.Tensor:
        # calculating cmr loss
        gt_similarity=gt_logits_per_image.diag().reshape(-1,1).expand(da_logits_per_image.shape)
        # gt_similarity=gt_logits_per_image.gather(0,torch.arange(min(gt_logits_per_image.shape),device=gt_logits_per_image.device).reshape(1,-1)).reshape(min(gt_logits_per_image.shape),1).expand(da_logits_per_image.shape)
        cmr_loss=nn.functional.relu((thresholds+da_logits_per_image-gt_similarity))*valid_caption_mask


        # updating thresholds
        if self.threshold_type=='mean':
            mask = da_logits_per_image!=0
            average_similarity_for_types = (da_logits_per_image*mask).sum(dim=0)/mask.sum(dim=0)
            thresholds=(gt_similarity.mean(0)-average_similarity_for_types).expand(gt_similarity.shape)
            thresholds=thresholds.detach()
        elif self.threshold_type=='max':
            thresholds,max_indices=(gt_similarity*valid_caption_mask-da_logits_per_image).max(0)
            thresholds=thresholds.expand(gt_similarity.shape)/5
            thresholds=thresholds.detach()
        return cmr_loss.mean(),thresholds

    def get_imc_loss(self,gt_logits_per_image:torch.Tensor,embedding_matrix:torch.Tensor):
        """
        gt_logits_per_image: standard clip similarity matrix, diag is true gt similarity value : shape [batch_size,5xbatch_size]
        embedding_matrix: extra similarity matrix served as denominator in clip loss
        """
        
        logtis_matrix = embedding_matrix
        labels=torch.zeros(logtis_matrix.shape[0],device=logtis_matrix.device,dtype=torch.long)
        imc_loss=F.cross_entropy(logtis_matrix,labels)
        return imc_loss
        
    def get_hndc_loss(self, da_text_features: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """
        - Computes similarity matrix excluding self-similarity
        - Uses fake positive at index 0 in each row for cross-entropy loss
        """
        N, D = da_text_features.shape
        x = F.normalize(da_text_features, dim=-1)  # [N, D]
        sim_matrix = torch.matmul(x, x.T) / temperature  # [N, N]
        mask = torch.eye(N, dtype=torch.bool, device=x.device)
        sim_matrix_no_self = sim_matrix[~mask].view(N, N - 1)  # [N, N-1]
        labels = torch.zeros(N, dtype=torch.long, device=x.device)

        loss = F.cross_entropy(sim_matrix_no_self, labels)
        return loss   



class Siglip_DA_all_hn_far_Loss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        cmr_loss=False,
        imc_loss=False,
        hndc_loss=False,
        hardnegative=False,
        imc_loss_weight=0.2,
        cmr_loss_weight=0.2,
        hndc_loss_weight=0.2,
        threshold_type='mean',
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.prev_num_logits = 0
        self.labels = {}
        self.cmr_loss = cmr_loss
        self.imc_loss = imc_loss
        self.hndc_loss = hndc_loss
        self.hardnegative = hardnegative

        self.imc_loss_weight = imc_loss_weight
        self.cmr_loss_weight = cmr_loss_weight
        self.hndc_loss_weight = hndc_loss_weight
        self.threshold_type = threshold_type

    def forward(self, image_features, text_features, valid_caption_mask, logit_scale, thresholds):
        device = image_features.device
        total_loss = 0.0
        itc_loss = cmr_loss = imc_loss = hndc_loss = 0.0

        # 🚩 Clamp logit_scale for stability
        with torch.no_grad():
            logit_scale = logit_scale.clamp(max=5)

        # 🚩 Normalize embeddings for stability
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        if self.world_size > 1:
            raise NotImplementedError("Multi-GPU distributed training support not implemented in this version.")
        else:
            gt_len = image_features.shape[0]
            gt_text_features = text_features[:gt_len]
            da_text_features = text_features[gt_len:]
            all_text_features = torch.cat([gt_text_features, da_text_features], dim=0)

            if self.hardnegative:
                logits_per_image = logit_scale * image_features @ all_text_features.T
            else:
                logits_per_image = logit_scale * image_features @ gt_text_features.T

            logits_per_text = logit_scale * gt_text_features @ image_features.T

            # Compute CMR_Loss if enabled
            if self.cmr_loss:
                da_logits_per_image = logit_scale * (
                    da_text_features.reshape(gt_len, -1, gt_text_features.shape[-1])
                    @ image_features.unsqueeze(-1)
                ).squeeze() * valid_caption_mask

                cmr_loss, thresholds = self.get_cmr_loss(
                    logits_per_image, da_logits_per_image, valid_caption_mask, thresholds
                )

            # Compute IMC_Loss if enabled
            if self.imc_loss:
                # 🚩 Normalize before IMC_Loss again for safety
                gt_text_features = F.normalize(gt_text_features, dim=-1)
                da_text_features = F.normalize(da_text_features, dim=-1)

                text_embedding_matrix = logit_scale * gt_text_features @ da_text_features.T

                try:
                    imc_loss = self.get_imc_loss(logits_per_image, text_embedding_matrix)
                    if torch.isnan(imc_loss):
                        print("⚠️ NaN detected in IMC_Loss, setting loss to zero for this batch.")
                        imc_loss = torch.tensor(0.0, device=device)
                except Exception as e:
                    print(f"⚠️ Exception in IMC_Loss: {e}, setting loss to zero for this batch.")
                    imc_loss = torch.tensor(0.0, device=device)

            # Compute HNDC_Loss if enabled
            if self.hndc_loss:
                hndc_loss = self.get_hndc_loss(da_text_features=da_text_features, temperature=0.07)

        # Labels
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # ITC_Loss
        itc_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        total_loss += itc_loss

        # Add other losses
        if self.cmr_loss:
            total_loss += cmr_loss * self.cmr_loss_weight
        if self.imc_loss:
            total_loss += imc_loss * self.imc_loss_weight
        if self.hndc_loss:
            total_loss += hndc_loss * self.hndc_loss_weight

        return (
            total_loss,
            thresholds if self.cmr_loss else None,
            itc_loss,
            cmr_loss if self.cmr_loss else None,
            imc_loss if self.imc_loss else None,
            hndc_loss if self.hndc_loss else None,
        )

    def get_cmr_loss(self, gt_logits_per_image, da_logits_per_image, valid_caption_mask, thresholds):
        gt_similarity = gt_logits_per_image.diag().reshape(-1, 1).expand(da_logits_per_image.shape)
        cmr_loss = F.relu((thresholds + da_logits_per_image - gt_similarity)) * valid_caption_mask

        if self.threshold_type == 'mean':
            mask = da_logits_per_image != 0
            avg_similarity = (da_logits_per_image * mask).sum(dim=0) / mask.sum(dim=0)
            thresholds = (gt_similarity.mean(0) - avg_similarity).expand(gt_similarity.shape).detach()
        elif self.threshold_type == 'max':
            thresholds, _ = (gt_similarity * valid_caption_mask - da_logits_per_image).max(0)
            thresholds = (thresholds.expand(gt_similarity.shape) / 5).detach()

        return cmr_loss.mean(), thresholds

    def get_imc_loss(self, gt_logits_per_image, embedding_matrix):
        labels = torch.zeros(embedding_matrix.shape[0], device=embedding_matrix.device, dtype=torch.long)
        imc_loss = F.cross_entropy(embedding_matrix, labels)
        return imc_loss

    def get_hndc_loss(self, da_text_features, temperature=0.07):
        N, D = da_text_features.shape
        x = F.normalize(da_text_features, dim=-1)
        sim_matrix = torch.matmul(x, x.T) / temperature
        mask = torch.eye(N, dtype=torch.bool, device=x.device)
        sim_matrix_no_self = sim_matrix[~mask].view(N, N - 1)
        labels = torch.zeros(N, dtype=torch.long, device=x.device)
        loss = F.cross_entropy(sim_matrix_no_self, labels)
        return loss

