import mathimport torchimport torch.nn as nnimport torch.nn.functional as Ffrom DeepTarget.distribute_utils import is_main_processfrom DeepTarget.utils import TAPETokenizer, concat_all_gather, tie_encoder_decoder_weights, load_pretrainfrom modeling_bert.tape.models.modeling_bert import ProteinBertModelclass Mol_Transformer(nn.Module):    def __init__(self, vocabulary, config, multi_mode=False, work='encoder'):        super(Mol_Transformer, self).__init__()        self.max_len = config.max_len        self.model_type = 'Transformer'        self.multi_mode = multi_mode        self.src_mask = None        self.pad_mask = None        self.vocabulary = vocabulary        self.hidden_size = config.hidden        self.nhead = config.head        self.num_layers = config.num_layers        self.dropout = config.dropout        self.ninp = config.ninp        self.vocab_size = len(vocabulary)        self.encoder = nn.Embedding(self.vocab_size, self.ninp, padding_idx=vocabulary.pad)        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.ninp))        self.drop = nn.Dropout(config.dropout)        self.is_casual = False if work == 'encoder' else True        # transformer        self.encoder_blocks = nn.Sequential(            *[Block(config, multi_mode=self.multi_mode, is_casual=self.is_casual) for _ in range(config.num_layers)])        # decoder head        self.ln_f = nn.LayerNorm(config.ninp)        self.decoder = nn.Linear(self.ninp, self.vocab_size)        self.init_decoder_weights()    @property    def device(self):        return next(self.parameters()).device    def init_decoder_weights(self):        initrange = 0.1        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)        nn.init.zeros_(self.decoder.bias)        nn.init.uniform_(self.decoder.weight, -initrange, initrange)    def init_weights(self):        """ Initialize and prunes weights if needed. """        # Initialize weights        self.apply(self._init_weights)        # Prune heads if needed        if getattr(self.config, 'pruned_heads', False):            self.prune_heads(self.config.pruned_heads)    def prune_heads(self, heads_to_prune):        """ Prunes heads of the base model.            Arguments:                heads_to_prune: dict with keys being selected layer indices (`int`) and                    associated values being the list of heads to prune in said layer                    (list of `int`).        """        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed        base_model._prune_heads(heads_to_prune)    def forward(self, x, encoder_hidden_states=None, mol_padding_mask=None, prot_padding_mask=None):        b, t = x.size()        self.src_mask = None        self.pad_mask = None        x = self.encoder(x)  # 字符编码+位置编码        x = self.drop(x + self.pos_emb[:, :t, :])        for layer in self.encoder_blocks:            x = layer(x, encoder_hidden_states, mol_padding_mask=mol_padding_mask, prot_padding_mask=prot_padding_mask)        x = self.ln_f(x)        mol_feature = x[:, 0]        logits = self.decoder(x)        return logits, mol_feature    def string2tensor(self, string, max_len=100, device='model'):        ids = self.vocabulary.string2ids(string, add_eos=True)        ids = ids[:max_len] if len(ids) > max_len else ids        tensor = torch.tensor(ids, dtype=torch.long,                              device=self.device                              if device == 'model' else device)        return tensor    def tensor2string(self, tensor):        ids = tensor.tolist()        try:            end_first_index = ids.index(self.vocabulary.eos)            ids = ids[:end_first_index + 1]        except:            pass        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)        return string    def sample(self, n_batch, max_length=100):        print('max_len:', self.max_len)        max_length = self.max_len        starts = [torch.tensor([self.vocabulary.bos],                               dtype=torch.long,                               device=self.device)                  for _ in range(n_batch)]        starts = torch.tensor(starts, dtype=torch.long,                              device=self.device).unsqueeze(1)        input = starts        with torch.no_grad():            for i in range(max_length):                output = self.forward(input, encoder_hidden_states=None)[0]                output = F.softmax(output, dim=-1)                word_weights = output[:, -1].cpu()                word_idx = torch.multinomial(word_weights, 1)                input = torch.cat([input, word_idx.cuda()], 1)            out = [self.tensor2string(t) for t in input]        return outclass Block(nn.Module):    """ an unassuming Transformer block """    def __init__(self, config, multi_mode=False, is_casual=False):        super().__init__()        self.ln1 = nn.LayerNorm(config.ninp)        self.ln2 = nn.LayerNorm(config.ninp)        self.self_attn = CausalSelfAttention(config, is_cross_attention=False, is_casual=is_casual)        self.multi_mode = multi_mode        self.cross_attn = CausalSelfAttention(config, is_cross_attention=True)        self.mlp = nn.Sequential(            nn.Linear(config.ninp, 4 * config.ninp),            nn.GELU(),            nn.Linear(4 * config.ninp, config.ninp),            nn.Dropout(config.dropout),        )    def forward(self, x, encoder_hidden_states=None, mol_padding_mask=None, prot_padding_mask=None):        x = self.ln1(x)        self_attention_output = self.self_attn(x, encoder_hidden_states=None, mol_padding_mask=mol_padding_mask,                                               prot_padding_mask=prot_padding_mask)        x = x + self_attention_output        if self.multi_mode and encoder_hidden_states is not None:            cross_attention_output = self.cross_attn(x, encoder_hidden_states=encoder_hidden_states,                                                     mol_padding_mask=mol_padding_mask,                                                     prot_padding_mask=prot_padding_mask)            x = x + cross_attention_output        x = x + self.mlp(self.ln2(x))        return xclass CausalSelfAttention(nn.Module):    """    A vanilla multi-head masked self-attention layer with a projection at the end.    It is possible to use torch.nn.MultiheadAttention here but I am including an    explicit implementation here to show that there is nothing too scary here.    """    def __init__(self, config, is_cross_attention=False, is_casual=False):        super().__init__()        assert config.ninp % config.head == 0        self.is_cross_attention = is_cross_attention        self.is_casual = is_casual        self.max_len = config.max_len        self.attn_dim = config.attn_dim        # key, query, value projections for all heads        self.query = nn.Linear(config.ninp, self.attn_dim)        if self.is_cross_attention:            self.key = nn.Linear(config.prot_dim, self.attn_dim)            self.value = nn.Linear(config.prot_dim, self.attn_dim)        else:            self.key = nn.Linear(config.ninp, self.attn_dim)            self.value = nn.Linear(config.ninp, self.attn_dim)        # regularization        self.attn_drop = nn.Dropout(config.dropout)        self.resid_drop = nn.Dropout(config.dropout)        # output projection        self.proj = nn.Linear(self.attn_dim, config.ninp)        # causal mask to ensure that attention is only applied to the left in the input sequence        self.n_head = config.head    def extend_mask(self, attention_mask, is_cross_attention=False):        if is_cross_attention:            extended_attention_mask = attention_mask[:, None, None, :]        else:  # SA            if self.is_casual:                batch_size, seq_length = attention_mask.size()                seq_ids = torch.arange(seq_length, device=attention_mask.device)                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]                causal_mask = causal_mask.to(attention_mask.dtype)                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]            else:                extended_attention_mask = attention_mask[:, None, None, :]        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0        return extended_attention_mask    def forward(self, x, encoder_hidden_states=None, mol_padding_mask=None, prot_padding_mask=None):        B, T, C = x.size()        if not self.is_cross_attention:            # calculate query, key, values for all heads in batch and move head forward to be the batch dim            q = self.query(x).view(B, T, self.n_head, self.attn_dim // self.n_head).transpose(1, 2)  # (B, nh, T, hs)            k = self.key(x).view(B, T, self.n_head, self.attn_dim // self.n_head).transpose(1, 2)  # (B, nh, T, hs)            v = self.value(x).view(B, T, self.n_head, self.attn_dim // self.n_head).transpose(1, 2)  # (B, nh, T, hs)        else:            # If this is instantiated as a cross-attention module, the keys            # and values come from an encoder; the attention mask needs to be            # such that the encoder's padding tokens are not attended to.            q = self.query(x).view(B, T, self.n_head, self.attn_dim // self.n_head).transpose(1, 2)            _, _T, _C = encoder_hidden_states.size()            k = self.key(encoder_hidden_states).view(B, _T, self.n_head, self.attn_dim // self.n_head).transpose(1, 2)            v = self.value(encoder_hidden_states).view(B, _T, self.n_head, self.attn_dim // self.n_head).transpose(1, 2)        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))        if mol_padding_mask is None:            mol_padding_mask = torch.ones((B, T), device=x.device)        if self.is_cross_attention:            attention_mask = prot_padding_mask        else:            attention_mask = mol_padding_mask        mask = self.extend_mask(attention_mask=attention_mask, is_cross_attention=self.is_cross_attention)        att = att + mask        att = F.softmax(att, dim=-1)        # attn_save = att        att = self.attn_drop(att)        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)        y = y.transpose(1, 2).contiguous().view(B, T, self.attn_dim)  # re-assemble all head outputs side by side        # output projection        y = self.resid_drop(self.proj(y))        return y  # , attn_saveprot_pretrained_model_name_or_path = '/home/s2136015/Code/TransVAE/ProteinLM/tape/models/out'class DeepTarget(nn.Module):    def __init__(self, vocab, config, queue_size=1024, momentum=0.995):        super().__init__()        self.config = config        self.emb = config.ninp        self.momentum = momentum        self.queue_size = queue_size        self.temp = nn.Parameter(0.07 * torch.ones([]))        self.prot_tokenizer = TAPETokenizer()        self.init_model(vocab, config)        if self.config.CL_model:            self.init_m_model(vocab, config)            self.model_pairs = [[self.mol_model_encoder, self.mol_model_encoder_m],                                [self.prot_model, self.prot_model_m],                                [self.mol2emb, self.mol2emb_m],                                [self.prot2emb, self.prot2emb_m]]            self.register_buffer("prot_queue", torch.randn(config.ninp, queue_size))            self.register_buffer("mol_queue", torch.randn(config.ninp, queue_size))            self.register_buffer("prot_idx_queue", torch.full((1, queue_size), -100))            self.register_buffer("mol_idx_queue", torch.full((1, queue_size), -100))            self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))            self.prot_queue = nn.functional.normalize(self.prot_queue, dim=0)            self.mol_queue = nn.functional.normalize(self.mol_queue, dim=0)        if self.config.Matching_model:            self.protmol_match_head = nn.Linear(config.ninp, 2)            self.protmol_affinity_head = nn.Linear(config.ninp, 1)            self.protmol_hid2match = nn.Linear(config.ninp, config.ninp)            self.protmol_hid2affinity = nn.Linear(config.ninp, config.ninp)    @property    def device(self):        return next(self.parameters()).device    def init_model(self, vocab, config):        if self.config.MLM_model:            self.mol_model_encoder = Mol_Transformer(vocab[0], config, multi_mode=True, work='encoder')            return        self.prot_model = ProteinBertModel.from_pretrained(prot_pretrained_model_name_or_path)        for name, child in self.prot_model.named_children():            for param in child.parameters():                param.requires_grad = False        if self.config.Language_model:            self.mol_model_decoder = Mol_Transformer(vocab[0], config, multi_mode=True, work='decoder')            # self.mol_model_decoder = load_pretrain(self.mol_model_decoder,self.config.MLM_model_path)        if self.config.CL_model:            self.mol_model_encoder = Mol_Transformer(vocab[0], config, multi_mode=True, work='encoder')            # self.mol_model_encoder = load_pretrain(self.mol_model_encoder, self.config.MLM_model_path)            tie_encoder_decoder_weights(self.mol_model_encoder, self.mol_model_decoder, '', skip_key='self_attn')            self.mol2emb = nn.Linear(config.ninp, self.emb)            self.prot2emb = nn.Linear(1024, self.emb)    # create momentum encoders    def init_m_model(self, vocab, config):        self.mol_model_encoder_m = Mol_Transformer(vocab[0], config, multi_mode=True)        self.prot_model_m = ProteinBertModel.from_pretrained(prot_pretrained_model_name_or_path)        for name, child in self.prot_model_m.named_children():            for param in child.parameters():                param.requires_grad = False        self.mol2emb_m = nn.Linear(config.ninp, self.emb)        self.prot2emb_m = nn.Linear(1024, self.emb)    @torch.no_grad()    def copy_params(self):        for model_pair in self.model_pairs:            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):                param_m.data.copy_(param.data)  # initialize                param_m.requires_grad = False  # not update by gradient    @torch.no_grad()    def _momentum_update(self):        for model_pair in self.model_pairs:            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)    @torch.no_grad()    def _dequeue_and_enqueue(self, mol_feats, prot_feats, mol_idxs, prot_idxs):        # gather keys before updating queue        if self.config.multi_gpu:            mol_feats = concat_all_gather(mol_feats)            prot_feats = concat_all_gather(prot_feats)  # 分布式的时候用        batch_size = mol_feats.shape[0]        ptr = int(self.ptr_queue)        assert self.queue_size % batch_size == 0  # for simplicity        # replace the keys at ptr (dequeue and enqueue)        self.mol_queue[:, ptr:ptr + batch_size] = mol_feats.T        self.prot_queue[:, ptr:ptr + batch_size] = prot_feats.T        self.mol_idx_queue[:, ptr:ptr + batch_size] = mol_idxs.T        self.prot_idx_queue[:, ptr:ptr + batch_size] = prot_idxs.T        ptr = (ptr + batch_size) % self.queue_size  # move pointer        self.ptr_queue[0] = ptr    def forward(self, mol_mask, mol_target,                mol_padding_mask=None,                prot_input_ids=None,                prot_input_mask=None,                mol_idx=None,                prot_idx=None,                affinity_score=None,                epoch=0):        batch = len(mol_mask)        loss = 0        with torch.no_grad():            prot_outputs = self.prot_model(prot_input_ids, prot_input_mask)        prot_encoder_feature, prot_feature = prot_outputs[0], prot_outputs[-1]        prot_feature = F.normalize(self.prot2emb(prot_feature), dim=-1)        _, mol_feature = self.mol_model_encoder(mol_target,                                                mol_padding_mask=mol_padding_mask)  # batch x len x dim,[cls]        mol_feature = F.normalize(self.mol2emb(mol_feature), dim=-1)        if self.config.Language_model:            # ###======================= LM ===========================###            lm_mol_target = mol_target.clone()            lm_mol_target[:, 0] = self.mol_model_decoder.vocabulary.bos            lm_input_ids = lm_mol_target[:, :-1]            lm_output_ids = mol_target[:, 1:].clone()            padding_mask = mol_padding_mask.clone()[:, :-1]            lm_output = \                self.mol_model_decoder(lm_input_ids, encoder_hidden_states=prot_encoder_feature,                                       mol_padding_mask=padding_mask, prot_padding_mask=prot_input_mask)[0]            lm_loss = nn.CrossEntropyLoss()(lm_output.view(-1, lm_output.shape[-1]), lm_output_ids.view(-1))            loss += lm_loss        if self.config.CL_model:            # ###============== mol-protein Contrastive Learning ===================###            # get momentum features            mol_idx = mol_idx.view(-1, 1).cuda()            mol_idx_all = torch.cat([mol_idx.t(), self.mol_idx_queue.clone().detach()], dim=1)            mol_idx_all = torch.eq(mol_idx, mol_idx_all)            m2p_sim_targets = mol_idx_all / mol_idx_all.sum(1, keepdim=True)            prot_idx = prot_idx.view(-1, 1).cuda()            prot_idx_all = torch.cat([prot_idx.t(), self.prot_idx_queue.clone().detach()], dim=1)            prot_idx_all = torch.eq(prot_idx, prot_idx_all)            p2m_sim_targets = prot_idx_all / prot_idx_all.sum(1, keepdim=True)            with torch.no_grad():                self._momentum_update()                prot_feature_m = self.prot_model_m(prot_input_ids, prot_input_mask)[-1]                prot_feature_m = F.normalize(self.prot2emb_m(prot_feature_m), dim=-1)                prot_feat_m_all = torch.cat([prot_feature_m.t(), self.prot_queue.clone().detach()], dim=1)                _, mol_feature_m = self.mol_model_encoder_m(mol_mask)                mol_feature_m = F.normalize(self.mol2emb_m(mol_feature_m), dim=-1)                mol_feat_m_all = torch.cat([mol_feature_m.t(), self.mol_queue.clone().detach()], dim=1)                sim_m2p_m = mol_feature_m @ prot_feat_m_all / self.temp                sim_p2m_m = prot_feature_m @ mol_feat_m_all / self.temp                alpha = self.config.alpha if epoch > 0 else 0.0                sim_m2p_targets = alpha * F.softmax(sim_m2p_m, dim=1) + (1 - alpha) * m2p_sim_targets                sim_p2m_targets = alpha * F.softmax(sim_p2m_m, dim=1) + (1 - alpha) * p2m_sim_targets            sim_m2p = mol_feature @ prot_feat_m_all / self.temp            sim_p2m = prot_feature @ mol_feat_m_all / self.temp            loss_m2p = -torch.sum(F.log_softmax(sim_m2p, dim=1) * sim_m2p_targets, dim=1).mean()            loss_p2m = -torch.sum(F.log_softmax(sim_p2m, dim=1) * sim_p2m_targets, dim=1).mean()            cls_loss = (loss_m2p + loss_p2m) / 2            loss += cls_loss            if self.config.multi_gpu:                prot_idxs = concat_all_gather(prot_idx)  # 分布式的时候用                mol_idxs = concat_all_gather(mol_idx)            self._dequeue_and_enqueue(mol_feature_m, prot_feature_m, mol_idxs, prot_idxs)        if self.config.Matching_model:            # ###======================= mol-protein Matching ===========================###            m_mol_target = mol_target.clone()            m_mol_target[:, 0] = self.mol_model_decoder.vocabulary.enc            # forward the positve image-text pair            output_pos = self.mol_model_encoder(m_mol_target, encoder_hidden_states=prot_encoder_feature,                                                mol_padding_mask=mol_padding_mask, prot_padding_mask=prot_input_mask)            # forward the negetive image-text pair            with torch.no_grad():                m2p_mask = torch.eq(mol_idx, mol_idx.t()).cuda()                p2m_mask = torch.eq(prot_idx, prot_idx.t()).cuda()                sim_m2p = mol_feature @ prot_feature.t() / self.temp                sim_p2m = prot_feature @ mol_feature.t() / self.temp                weights_m2p = F.softmax(sim_m2p, dim=1)                weights_m2p.masked_fill_(m2p_mask, 0)                weights_p2m = F.softmax(sim_p2m, dim=1)                weights_p2m.masked_fill_(p2m_mask, 0)            # select a negative prot (from all ranks) for each mol            prot_embeds_neg = []            for b in range(self.config.n_batch):                neg_idx = torch.multinomial(weights_m2p[b], 1).item()                prot_embeds_neg.append(prot_encoder_feature[neg_idx])            prot_embeds_neg = torch.stack(prot_embeds_neg, dim=0)            # select a negative mol (from all ranks) for each prot            mol_neg = []            for b in range(self.config.n_batch):                neg_idx = torch.multinomial(weights_p2m[b], 1).item()                mol_neg.append(m_mol_target[neg_idx])            mol_neg = torch.stack(mol_neg, dim=0)            mol_ids_all = torch.cat([m_mol_target, mol_neg], dim=0)            prot_embeds_all = torch.cat([prot_encoder_feature, prot_embeds_neg], dim=0)            output_neg = self.mol_model_encoder(mol_ids_all, encoder_hidden_states=prot_embeds_all, is_causal=False)            match_embeddings = torch.cat([output_pos[-1], output_neg[-1]], dim=0)            # DTI            match_embeddings = self.protmol_hid2match(match_embeddings)            match_output = self.protmol_match_head(match_embeddings)            mpm_labels = torch.cat([torch.ones(batch, dtype=torch.long), torch.zeros(2 * batch, dtype=torch.long)],                                   dim=0).cuda()            mpI_loss = nn.CrossEntropyLoss()(match_output, mpm_labels)            # DTA            affinity_embeddings = self.protmol_hid2affinity(output_pos[-1])            affinity_output = self.protmol_affinity_head(affinity_embeddings)            mpA_loss = nn.MSELoss()(affinity_output, affinity_score.view(-1, 1))            mpm_loss = (mpI_loss / 3 + mpA_loss) / 2            loss += mpm_loss        loss_dict = {'CL_loss': cls_loss.item() if self.config.CL_model else 0,                     'matching_loss': mpm_loss.item() if self.config.Matching_model else 0,                     'lm_loss': lm_loss.item() if self.config.Language_model else 0}        return loss, loss_dict    def pretrain_lm(self, mol_target):        lm_mol_target = mol_target.clone()        lm_mol_target[:, 0] = self.mol_model.vocabulary.bos        lm_input_ids = lm_mol_target[:, :-1]        lm_output_ids = mol_target[:, 1:].clone()        lm_output = self.mol_model_decoder(lm_input_ids, encoder_hidden_states=None, is_causal=True)[0]        lm_loss = nn.CrossEntropyLoss()(lm_output.view(-1, lm_output.shape[-1]), lm_output_ids.view(-1))        return lm_loss    def pretrain_mlm(self, mol_mask, mol_target, mol_padding_mask=None):        mol_mlm_outputs, _ = self.mol_model_encoder(mol_mask, mol_padding_mask=mol_padding_mask)        labels = mol_target.clone()        labels[mol_mask != self.mol_model_encoder.vocabulary.mask] = -100        loss = nn.CrossEntropyLoss()(mol_mlm_outputs.view(-1, mol_mlm_outputs.shape[-1]),                                     labels.view(-1))        loss_dict = {"MLM_loss": loss.item()}        return loss, loss_dict    def sample(self, prot, prot_mask, n_batch=1):        prot_output = self.prot_model(prot, prot_mask)        encoder_hidden_states = prot_output[0]        encoder_hidden_states = encoder_hidden_states.repeat(n_batch, 1, 1)        # print('max_len:', self.config.max_len)        max_length = self.config.max_len        starts = [torch.tensor([self.mol_model_decoder.vocabulary.bos],                               dtype=torch.long,                               device=self.device)                  for _ in range(encoder_hidden_states.size(0))]        starts = torch.tensor(starts, dtype=torch.long,                              device=self.device).unsqueeze(1)        input = starts        with torch.no_grad():            for i in range(max_length):                output = \                    self.mol_model_decoder.forward(input, encoder_hidden_states=encoder_hidden_states,                                                   prot_padding_mask=prot_mask)[0]                output = F.softmax(output, dim=-1)                word_weights = output[:, -1].cpu()                word_idx = torch.multinomial(word_weights, 1)                input = torch.cat([input, word_idx.cuda()], 1)            out = [self.mol_model.tensor2string(t) for t in input]        return out    def test_mpm(self, mol, prot, prot_mask):        prot_outputs = self.prot_model(prot, prot_mask)        prot_encoder_feature, prot_feature = prot_outputs[0], prot_outputs[-1]        output = self.mol_model_decoder(mol, encoder_hidden_states=prot_encoder_feature, is_causal=False)[-1]        # DTI        match_output = self.protmol_hid2match(output)        match_output = self.protmol_match_head(match_output)        match_output = match_output.argmax(dim=1)        # DTA        affinity_output = self.protmol_hid2affinity(output)        affinity_output = self.protmol_affinity_head(affinity_output)        return match_output, affinity_output