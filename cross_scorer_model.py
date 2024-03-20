import transformers
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers import BertForMaskedLM
import torch.nn.functional as F
import spacy
import torch.nn as nn    
class Similarity(nn.Module):    
    """    
    Dot product or cosine similarity    
    """    
    def __init__(self, temp):    
        super().__init__()    
        self.temp = temp    
        self.cos = nn.CosineSimilarity(dim=-1)    
    def forward(self, x, y):    
        return self.cos(x, y) / self.temp    
    def temp_forward(self, x, y, temp):    
        return self.cos(x, y) / temp    
class CrossScorer(nn.Module):
    """
    Note: This is the bi encoder (separate) model.
          TODO: Fix name accordingly
    """
    def __init__(self, p_encoder, r_encoder, use_aux_loss=False): 
        """
        """
        super(CrossScorer, self).__init__()
        self.p_encoder = p_encoder
        self.r_encoder = r_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = 0.1
        self.sim = Similarity(temp=self.temp)
        self.hn_weight = 1.0
        self.lamb_1 = 0.5
        self.freeze_response = False
        if self.freeze_response:
            for param in self.r_encoder.parameters():
                param.requires_grad = False
        self.use_aux_loss = use_aux_loss
        self.encoder_type = "bi"
    def get_pos_embeds(self, pos_masks, last_hidden):
        pos_embeds = pos_masks.unsqueeze(-1) * last_hidden
        pos_embeds = torch.sum(pos_embeds, dim=1)
        return pos_embeds
    def score_forward(
        self,
        p_batch=None,
        r_batch=None,
        pv=None,
        pn=None,
        rv=None,
        rn=None,
        ):  
        """
        TODO: decide where to prepare the strings into the batch and masks
              and do demo
        """
        p_output, verb_p, noun_p = self.p_encoder.emb_forward(
                **p_batch, verb_mask=pv, noun_mask=pn
                )        
        r_output, verb_r, noun_r = self.r_encoder.emb_forward(
                **r_batch, verb_mask =rv, noun_mask=rn
                )       
        p_z = p_output 
        r_z = r_output 
        cos_sim, cos_denom = self.flat_sim(p_z, r_z)
        if self.use_aux_loss:
            noun_sim, denom_n = self.flat_sim(noun_p,noun_r)
            verb_sim, denom_v = self.flat_sim(verb_p,verb_r)
            cos_sim = cos_sim 
            verb_sim = verb_sim 
            noun_sim = noun_sim 
            score = cos_sim + verb_sim + noun_sim
            score = score /3.0
        else:
            score = cos_sim
        return score
    def flat_sim(self, z1, z2):
        return self.sim(z1,z2), z1.size(0)
    def cl_loss(self, z1, z2, hn_batch=None, hn_output=None):
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        ce_loss_fct = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fct(cos_sim, labels)
        if hn_batch:
            hn_z = hn_output.last_hidden_state[:,0,:]
            p_hn_cos = self.sim(z_1.unsqueeze(1), hn_z.unsqueeze(0))
            cos_sim = torch.cat([cos_sim,p_hn_cos],1)
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        if hn_batch:
            weights = torch.tensor([[0.0] * (cos_sim.size(-1) - p_hn_cos.size(-1)) + [0.0] * i + [self.hn_weight] + [0.0] * (p_hn_cos.size(-1) - i - 1) for i in range(p_hn_cos.size(-1))]).to(self.device)
            cos_sim = cos_sim + weights
        return ce_loss, cos_sim
    def forward(
        self,
        p_batch=None,
        r_batch=None,
        pv=None,
        pn=None,
        rv=None,
        rn=None,
        hn_batch=None
        ):  
        p_output, verb_p, noun_p = self.p_encoder.emb_forward(
                **p_batch, verb_mask=pv, noun_mask=pn
                )        
        r_output, verb_r, noun_r = self.r_encoder.emb_forward(
                **r_batch, verb_mask =rv, noun_mask=rn
                )        
        if hn_batch:
            hn_output = self.r_encoder.emb_forward(
                    **hn_batch
                    )    
        p_z = p_output
        r_z = r_output 
        ce_loss, cos_sim = self.cl_loss(p_z, r_z, hn_batch, None)
        if self.use_aux_loss:
            noun_loss, noun_sim = self.cl_loss(noun_p,noun_r)
            verb_loss, verb_sim = self.cl_loss(verb_p,verb_r)
            loss = ce_loss + verb_loss + noun_loss 
        else:
            loss = ce_loss
        return SequenceClassifierOutput(
                loss=loss,
                )
    def cl_loss_with_hn(self, prompts, responses):
        BSZ = prompts.size(0)
        responses = list(responses.tensor_split(BSZ, dim=0) )
        responses = torch.stack(responses)
        assert prompts.size(0) == responses.size(0)
        ce_loss_fct = nn.CrossEntropyLoss()
        loss = 0
        sim = 0
        for prompt, response_set in zip(prompts, responses):
            cos_sim = self.sim(prompt.unsqueeze(0), response_set)
            label = torch.LongTensor([0]).to(self.device) 
            cos_sim = cos_sim.unsqueeze(0)
            ce_loss = ce_loss_fct(cos_sim, label)
            loss += ce_loss
            sim += cos_sim
        return loss, sim
    def hard_forward(
        self,
        p_batch=None,
        r_batches=None,
        pv=None,
        pn=None,
        rv=None,
        rn=None,
        ):  
        p_output, verb_p, noun_p = self.p_encoder.emb_forward(
                **p_batch, verb_mask=pv, noun_mask=pn
                )        
        r_outputs, verb_rs, noun_rs = self.r_encoder.emb_forward(
                **r_batches, verb_mask =rv, noun_mask=rn
                )        
        p_z = p_output 
        r_zs = r_outputs 
        ce_loss, cos_sim = self.cl_loss_with_hn(p_z, r_zs)
        if self.use_aux_loss:
            noun_loss, noun_sim = self.cl_loss_with_hn(noun_p,noun_rs)
            verb_loss, verb_sim = self.cl_loss_with_hn(verb_p,verb_rs)
            loss = ce_loss + verb_loss + noun_loss 
        else:
            loss = ce_loss
        return SequenceClassifierOutput(
                loss=loss,
                )
class CrossScorerCrossEncoder(nn.Module):
    def __init__(self, transformer): 
        super(CrossScorerCrossEncoder, self).__init__()
        self.cross_encoder = transformer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.l1 = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ELU()
        self.l2 = torch.nn.Linear(512,1)
        self.encoder_type = "cross"    
    def saved_score_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):  
        output = self.cross_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pair_reps = output.last_hidden_state[:,0,:]
        logits = self.l2_classify(self.relu(self.l1(pair_reps)))
        return logits
    def score_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_attentions=False
        ):  
        output = self.cross_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pair_reps = output.last_hidden_state[:,0,:]
        score = self.l2(self.relu(self.l1(pair_reps)))
        if output_attentions and return_attentions:
            return score.sigmoid().squeeze(), output.attentions
        return score
    def cl_loss_all_random(self, pair_scores, labels):
        BSZ = pair_scores.size(0) 
        BSZ = int(BSZ/5)
        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
        lq_scores = pair_scores[:,1:] 
        hq_scores = pair_scores[:,0] 
        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        loss = hq_lq_loss
        return loss
    def cl_loss(self, pair_scores, labels):
        BSZ = pair_scores.size(0) 
        BSZ = int(BSZ/(5))
        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        gap_1_loss_fct = nn.MarginRankingLoss(margin=0.5)
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
        mq_scores = pair_scores[:,1] 
        lq_scores = pair_scores[:,2:-1] 
        hq_scores = pair_scores[:,0] 
        hq_mq_loss = gap_1_loss_fct(
                hq_scores.flatten(), 
                mq_scores.flatten(), 
                torch.ones(mq_scores.flatten().size()).to(self.device))
        mq_lq_loss = gap_1_loss_fct(
                mq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        mismatch_scores = pair_scores[:,-1]
        hq_mismatch_loss =  gap_2_loss_fct(
                        hq_scores.flatten(), 
                        mismatch_scores.flatten(), 
                        torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mq_mismatch_loss = gap_1_loss_fct(
                mq_scores.flatten(), 
                mismatch_scores.flatten(), 
                torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mismatch_loss = hq_mismatch_loss + mq_mismatch_loss 
        import wandb
        wandb.log({"hq_mq_loss": hq_mq_loss})
        wandb.log({"mq_lq_loss": mq_lq_loss})
        wandb.log({"hq_lq_loss": hq_lq_loss})
        wandb.log({"hq_mismatch_loss": hq_mismatch_loss})
        wandb.log({"mq_mismatch_loss": mq_mismatch_loss})
        loss = hq_mq_loss + mq_lq_loss + hq_lq_loss  + mismatch_loss 
        return loss
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        random = False
        ):
        pair_scores = self.score_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).sigmoid().squeeze()
        labels = None
        if True: 
            cl_loss = self.cl_loss(pair_scores, labels)
        else:
            pass
        loss =   cl_loss 
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
class CrossScorerBiEncoder(nn.Module):
    def __init__(self): 
        super(CrossScorerBiEncoder, self).__init__()
        self.p_encoder = AutoModel.from_pretrained("roberta-base")
        self.r_encoder = AutoModel.from_pretrained("roberta-base")
        self.attention = nn.MultiheadAttention(768, 1)
        self.l1 = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ELU()
        self.l2 = torch.nn.Linear(512,1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = 0.1
        self.sim = Similarity(temp=self.temp)
        self.encoder_type = "bi"    
    def score_forward(
        self,
        p_batch=None,
        r_batch=None
        ):  
        p_output = self.p_encoder(
                **p_batch
        )
        r_output = self.r_encoder(
                **r_batch
        )        
        p_pooled = p_output.last_hidden_state[:,0,:].unsqueeze(0)
        r_hiddens = r_output.last_hidden_state.transpose(1,0)
        attn_output, attn_output_weights = self.attention(p_pooled, r_hiddens, r_hiddens)
        attn_output = attn_output.transpose(1,0)
        score = self.l2(self.relu(self.l1(attn_output)))
        return score
    def cl_loss_all_random(self, pair_scores, labels):
        BSZ = pair_scores.size(0) 
        BSZ = int(BSZ/4)
        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
        lq_scores = pair_scores[:,1:] 
        hq_scores = pair_scores[:,0] 
        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        loss = hq_lq_loss
        return loss
    def cl_loss(self, pair_scores, labels):
        BSZ = pair_scores.size(0) 
        BSZ = int(BSZ/4)
        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        gap_1_loss_fct = nn.MarginRankingLoss(margin=0.5)
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
        mq_scores = pair_scores[:,1] 
        lq_scores = pair_scores[:,2:-1] 
        hq_scores = pair_scores[:,0] 
        hq_mq_loss = gap_1_loss_fct(
                hq_scores.flatten(), 
                mq_scores.flatten(), 
                torch.ones(mq_scores.flatten().size()).to(self.device))
        mq_lq_loss = gap_1_loss_fct(
                mq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        mismatch_scores = pair_scores[:,-1]
        hq_mismatch_loss =  gap_2_loss_fct(
                        hq_scores.flatten(), 
                        mismatch_scores.flatten(), 
                        torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mq_mismatch_loss = gap_1_loss_fct(
                mq_scores.flatten(), 
                mismatch_scores.flatten(), 
                torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mismatch_loss = hq_mismatch_loss + mq_mismatch_loss 
        loss = hq_mq_loss + mq_lq_loss + hq_lq_loss + mismatch_loss 
        return loss
    def forward(
        self,
        p_batch = None,
        r_batch = None,
        random = False
        ):
        pair_scores = self.score_forward(
                p_batch, r_batch
        ).squeeze()
        BSZ = pair_scores.size(0) 
        BSZ = int(BSZ/4)
        label = torch.zeros(4).long()
        label[0] = 1
        labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
        if not random:
            cl_loss = self.cl_loss(pair_scores, labels)
        else:
            cl_loss = self.cl_loss_all_random(pair_scores, labels)
        loss =  cl_loss 
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
    def saved_forward(
        self,
        p_batch = None,
        r_batch = None,
        random = False
        ):
        pair_scores = self.score_forward(
                p_batch, r_batch
        ).squeeze()
        BSZ = pair_scores.size(0) 
        BSZ = int(BSZ/4)
        label = torch.zeros(4).long()
        label[0] = 1
        labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
        if not random:
            cl_loss = self.cl_loss(pair_scores, labels)
        else:
            cl_loss = self.cl_loss_all_random(pair_scores, labels)
        loss =  cl_loss 
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
    def hard_forward(
        self,
        p_batch=None,
        r_batch=None
        ):
        return self.forward(
                p_batch, r_batch
        )
class CrossScorerWithHead(nn.Module):
    def __init__(self, p_encoder, r_encoder):
        super(CrossScorerWithHead, self).__init__()
        self.p_encoder = p_encoder
        self.r_encoder = r_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.l1 = torch.nn.Linear(768*3, 512)
        self.relu1 = torch.nn.ELU()
        self.l2 = torch.nn.Linear(512,256)
        self.relu2 = torch.nn.ELU()
        self.l3 = torch.nn.Linear(256,3)
    def score_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        return None
    def forward(
        self,
        p_batch=None,
        r_batch=None,
        labels = None
        ):  
        p_output = self.p_encoder.emb_forward(
                **p_batch
                )        
        r_output = self.r_encoder.emb_forward(
                **r_batch
                )       
        p_z = p_output.last_hidden_state[:,0,:]
        r_z = r_output.last_hidden_state[:,0,:]
        z = torch.cat([p_z,r_z,torch.abs(p_z-r_z)],dim=-1)       
        z = self.l3(self.relu2(self.l2(self.relu1(self.l1(z)))))
        if labels is None:
            return SequenceClassifierOutput(logits=z)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(z, labels)
        return SequenceClassifierOutput(
                loss=loss,
                logits=z,
                )
