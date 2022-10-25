from abc import ABC, abstractmethodimport torchimport torch.nn as nnimport torch.optim as optimfrom torch.utils.data import DataLoader, Datasetfrom tqdm.auto import tqdmimport torch.nn.functional as Fimport numpy as npimport pandas as pdfrom DeepTarget.distribute_utils import is_main_process, reduce_valuefrom DeepTarget.utils import set_torch_seed_to_all_gens, WordVocab, Logger, mol_random_mask, TAPETokenizer, IUPAC_VOCAB, \    prot_random_maskfrom torch.nn.utils.rnn import pad_sequencefrom pathlib import Pathimport pandas as pdclass BaseDataset(Dataset):    def __init__(self, data):        self.data = data    def __getitem__(self, index):        return self.data[index]    def __len__(self):        return len(self.data)class BaseTrainer(ABC):    @property    def n_workers(self):        n_workers = self.config.n_workers        return n_workers if n_workers != 1 else 0    def get_collate_device(self, model):        n_workers = self.n_workers        return 'cpu' if n_workers > 0 else model.device    def get_dataloader(self, model, data, collate_fn=None, shuffle=True):        if collate_fn is None:            collate_fn = self.get_collate_fn(model, type='train')        return DataLoader(BaseDataset(self.prepare_data(data)),                          batch_size=self.config.n_batch,                          shuffle=shuffle,                          drop_last=True,                          num_workers=self.n_workers, collate_fn=collate_fn,                          worker_init_fn=set_torch_seed_to_all_gens                          if self.n_workers > 0 else None)    def get_distribute_dataloader(self, model, data, config, collate_fn=None, shuffle=True):        if collate_fn is None:            collate_fn = self.get_collate_fn(model)        ###        train_sampler = torch.utils.data.distributed.DistributedSampler(            BaseDataset(self.prepare_data(data)),            num_replicas=config.world_size,            rank=config.rank        )        train_batch_sampler = torch.utils.data.BatchSampler(            train_sampler, self.config.n_batch, drop_last=True)        ###        return DataLoader(BaseDataset(self.prepare_data(data)),                          batch_sampler=train_batch_sampler,                          pin_memory=False,                          num_workers=self.n_workers, collate_fn=collate_fn,                          )    def get_genloader(self, model, data, collate_fn=None, shuffle=True):        if collate_fn is None:            collate_fn = self.get_collate_fn(model, type='sample')        return DataLoader(BaseDataset(self.prepare_data(data, type='sample')),                          batch_size=1,                          shuffle=shuffle,                          drop_last=True,                          num_workers=self.n_workers, collate_fn=collate_fn,                          worker_init_fn=set_torch_seed_to_all_gens                          if self.n_workers > 0 else None)    def prepare_data(self, data, type='train'):        if type is 'train':            mols, prots, mols_idx, prots_idx, affinity_scores = data[0], data[1], data[2], data[3], data[4]            data = [{'mol': smiles, 'prot': proteins, 'mol_idx': mol_idx, 'prot_idx': prot_idx,                     'affinity_score': affinity_score} for                    smiles, proteins, mol_idx, prot_idx, affinity_score in                    zip(mols, prots, mols_idx, prots_idx, affinity_scores)]        if type is 'sample':            prots, prots_idx = data[0], data[1]            data = [{'prot': proteins, 'prot_idx': prot_idx} for proteins, prot_idx in zip(prots, prots_idx)]        return data    def get_collate_fn(self, model, type):        return None    @abstractmethod    def get_vocabulary(self, mol, protein):        pass    @abstractmethod    def fit(self, model, train_data, val_data=None):        passclass DeepTargetTrainer(BaseTrainer):    def __init__(self, config):        self.config = config        self.temp = nn.Parameter(0.07 * torch.ones([]))    def get_loop_data(self, source, i):        data = source[:, :i]        return data    def _bootstrap(self, model, data, optimizer=None):        data_loader = self.get_dataloader(model, data, shuffle=True)        if optimizer is None:            model.eval()        res_df = pd.DataFrame(columns=['mol_idx', 'pro_idx'])        for i, (mol_mask, mol_target, mol_lens, prot_input_ids, prot_input_mask, prot_labels, mol_idx,                pro_idx) in enumerate(data_loader):            mol_mask, mol_target = mol_mask.to(model.device), mol_target.to(model.device)            prot_input_ids, prot_input_mask = prot_input_ids.to(model.device), prot_input_mask.to(model.device)            output = model.mol_prot_match_predict(mol_target, prot_input_ids, prot_input_mask)            match_output = model.protmol_match_head(output)            match_output_score = torch.nn.functional.softmax(match_output, dim=1)[:, 1]            match_idx = match_output_score.greater(self.config.bootstrap_threshold)            mol_idx, pro_idx = mol_idx[match_idx].tolist(), pro_idx[match_idx].tolist()            temp = pd.Series({'mol_idx': mol_idx, 'pro_idx': pro_idx})            res_df = res_df.append(temp, ignore_index=True)        res_df.to_csv(self.config.bootstrap_save_path, index=False)    def sample(self, model, data):        model.eval()        data_loader = self.get_genloader(model, data, shuffle=True)        res_df = pd.DataFrame(columns=['mol', 'pro_idx'])        for i, (prot_input_ids, prot_input_mask, pro_idx) in enumerate(data_loader):            prot_input_ids, prot_input_mask = prot_input_ids.to(model.device), prot_input_mask.to(model.device)            output = model.sample(prot_input_ids, prot_input_mask, self.config.sample_num)            mol = model.mol_model.tensor2string(output)            temp = pd.Series({'mol': mol, 'pro_idx': pro_idx})            res_df = res_df.append(temp, ignore_index=True)        res_df.to_csv(self.config.gen_savepath, index=False)    def _train_epoch(self, model, tqdm_data, optimizer=None, epoch=0):        if optimizer is None:            model.eval()        else:            model.train()        postfix = {'loss': 0,                   'running_loss': 0}        for i, (mol_mask, mol_target, mol_lens, prot_input_ids, prot_input_mask, prot_labels, mol_idx,                pro_idx, affinity_score) in enumerate(tqdm_data):            mol_mask, mol_target = mol_mask.to(model.device), mol_target.to(model.device)            prot_input_ids, prot_input_mask = prot_input_ids.to(model.device), prot_input_mask.to(model.device)            affinity_score = affinity_score.to(model.device)            loss, loss_dict = model(mol_mask, mol_target, prot_input_ids, prot_input_mask, mol_idx, pro_idx,                                    affinity_score, epoch)            # ###============== all loss ===================###            if optimizer is not None:                optimizer.zero_grad()                loss.backward()                loss = reduce_value(loss, average=True)                optimizer.step()            if is_main_process():                postfix['loss'] = loss.item()                postfix['running_loss'] += (loss.item() -                                            postfix['running_loss']) / (i + 1)                postfix['module_loss'] = loss_dict                tqdm_data.set_postfix(postfix)            # 测试affinity            if self.config.Matching_model:                if is_main_process() and epoch % 10 == 0 and optimizer is None:                    with torch.no_grad():                        if self.config.multi_gpu:                            match_output, affinity_output = model.module.test_mpm(mol_target, prot_input_ids,                                                                                  prot_input_mask)                        else:                            match_output, affinity_output = model.test_mpm(mol_target, prot_input_ids, prot_input_mask)                        test_res_file = Path(self.config.model_save[:-3] + '_{0:03d}.csv'.format(epoch))                        df = pd.DataFrame(columns=['affinity_score', 'pred_affinity_score', 'pred_label'])                        df['affinity_score'] = affinity_score.tolist()                        df['pred_affinity_score'] = affinity_output.tolist()                        df['pred_label'] = match_output.tolist()                        if test_res_file.exists():                            pre_df = pd.read_csv(self.config.model_save[:-3] + '_{0:03d}.csv'.format(epoch))                            df = pre_df.append(df)                        df.to_csv(self.config.model_save[:-3] + '_{0:03d}.csv'.format(epoch), index=False)        postfix['mode'] = 'Eval' if optimizer is None else 'Train'        # 等待所有进程计算完毕        if model.device != torch.device("cpu"):            torch.cuda.synchronize(model.device)        return postfix    def _train(self, model, train_loader, val_loader=None, logger=None):        def get_params():            return (p for p in model.parameters() if p.requires_grad)        device = model.device        optimizer = optim.AdamW(get_params(), lr=self.config.lr, weight_decay=0.05)        scheduler = optim.lr_scheduler.StepLR(optimizer,                                              self.config.step_size,                                              self.config.gamma)        model.zero_grad()        for epoch in range(self.config.train_epochs):            scheduler.step()            tqdm_data = tqdm(train_loader,                             desc='Training (epoch #{})'.format(epoch))            postfix = self._train_epoch(model, tqdm_data, optimizer, epoch)            if logger is not None:                logger.append(postfix)                logger.save(self.config.log_file)            if val_loader is not None:                tqdm_data = tqdm(val_loader,                                 desc='Validation (epoch #{})'.format(epoch))                postfix = self._train_epoch(model, tqdm_data)                if logger is not None:                    logger.append(postfix)                    logger.save(self.config.log_file)            if (self.config.model_save is not None) and \                    (epoch % self.config.save_frequency == 0):                model = model.to('cpu')                torch.save(                    model.state_dict(),                    self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch)                )                model = model.to(device)    def _distribute_train(self, model, train_loader, val_loader=None, logger=None):        def get_params():            return (p for p in model.parameters() if p.requires_grad)        device = model.device        print(self.config.gpu)        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.config.gpu],                                                          find_unused_parameters=True)        optimizer = optim.AdamW(get_params(), lr=self.config.lr, weight_decay=0.05)        scheduler = optim.lr_scheduler.StepLR(optimizer,                                              self.config.step_size,                                              self.config.gamma)        model.zero_grad()        for epoch in range(self.config.train_epochs):            scheduler.step()            # 在进程0中打印训练进度            if is_main_process():                train_loader = tqdm(train_loader,                                    desc='Training (epoch #{})'.format(epoch))            postfix = self._train_epoch(model, train_loader, optimizer, epoch)            if logger is not None and is_main_process():                logger.append(postfix)                logger.save(self.config.log_file)            if val_loader is not None:                if is_main_process():                    val_loader = tqdm(val_loader,                                      desc='Validation (epoch #{})'.format(epoch))                postfix = self._train_epoch(model, val_loader, epoch=epoch)                if logger is not None and is_main_process():                    logger.append(postfix)                    logger.save(self.config.log_file)            if (self.config.model_save is not None) and \                    (epoch % self.config.save_frequency == 0) \                    and is_main_process():                # model = model.to('cpu')                torch.save(                    model.module.state_dict(),                    self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch)                )                model = model.to(device)    def get_vocabulary(self, mol, protein):        mol_vocab = WordVocab.from_data(mol)        prot_vocab = IUPAC_VOCAB        return mol_vocab, prot_vocab    def get_collate_fn(self, model, type='train'):        device = self.get_collate_device(model)        def collate_train(data):            """            :param data: n_batch 个data            n x {'mol':mol, 'prot':prot, 'mol_idx':mol_idx, 'prot_idx':prot_index}            :return:            """            # 处理分子            mol_tensors = [model.mol_model.string2tensor(string['mol'], device=device)                           for string in data]            pad = model.mol_model.vocabulary.pad            res = [mol_random_mask(t, model.mol_model.vocabulary.mask) for t in mol_tensors]            mol_mask = pad_sequence([t['mask'] for t in res],                                    batch_first=True, padding_value=pad)            mol_target = pad_sequence([t['real'] for t in res],                                      batch_first=True, padding_value=pad)            mol_lens = torch.tensor([len(t) - 1 for t in mol_tensors],                                    dtype=torch.long, device=device)            # 处理蛋白            proteins = [string['prot'] for string in data]            proteins_tensor = [torch.tensor(model.prot_tokenizer.encode(protein)) for protein in proteins]            res = [prot_random_mask(protein_tensor, IUPAC_VOCAB) for protein_tensor in proteins_tensor]            prot_input_ids, prot_input_mask, prot_labels = [], [], []            for r in res:                prot_input_ids.append(r['input_ids'])                prot_input_mask.append(r['input_mask'])                prot_labels.append(r['labels'])            prot_input_ids = pad_sequence(prot_input_ids, batch_first=True, padding_value=0).long()            prot_input_mask = pad_sequence(prot_input_mask, batch_first=True, padding_value=0).long()            # ignore_index is -1            prot_labels = pad_sequence(prot_labels, batch_first=True, padding_value=-1).long()            # 处理分子idx            mol_idx = [string['mol_idx'] for string in data]            mol_idx = torch.Tensor(mol_idx)            # 处理蛋白idx            pro_idx = [string['prot_idx'] for string in data]            pro_idx = torch.Tensor(pro_idx)            # 处理affinity_score            affinity_score = [string['affinity_score'] for string in data]            affinity_score = torch.Tensor(affinity_score)            return mol_mask, mol_target, mol_lens, prot_input_ids, prot_input_mask, prot_labels, mol_idx, pro_idx, affinity_score        def collate_sample(data):            # 处理蛋白            proteins = [string['prot'] for string in data]            proteins_tensor = [torch.tensor(model.prot_tokenizer.encode(protein)) for protein in proteins]            res = [prot_random_mask(protein_tensor, IUPAC_VOCAB) for protein_tensor in proteins_tensor]            prot_input_ids, prot_input_mask = [], []            for r in res:                prot_input_ids.append(r['input_ids'])                prot_input_mask.append(r['input_mask'])            prot_input_ids = pad_sequence(prot_input_ids, batch_first=True, padding_value=0).long()            prot_input_mask = pad_sequence(prot_input_mask, batch_first=True, padding_value=0).long()            # 处理蛋白idx            pro_idx = [string['prot_idx'] for string in data]            pro_idx = torch.Tensor(pro_idx)            return prot_input_ids, prot_input_mask, pro_idx        return collate_train if type is 'train' else collate_sample    def fit(self, model, train_data, val_data=None):        logger = Logger() if self.config.log_file is not None else None        if self.config.multi_gpu:            train_loader = self.get_distribute_dataloader(model, train_data, config=self.config, shuffle=True)            val_loader = None if val_data is None else self.get_distribute_dataloader(                model, val_data, config=self.config, shuffle=False)            self._distribute_train(model, train_loader, val_loader, logger)        else:            train_loader = self.get_dataloader(model, train_data, shuffle=True)            val_loader = None if val_data is None else self.get_dataloader(                model, val_data, shuffle=False)            self._train(model, train_loader, val_loader, logger)        return model