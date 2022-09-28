import randomimport refrom multiprocessing import Poolfrom collections import UserList, defaultdictimport numpy as npimport pandas as pdfrom matplotlib import pyplot as pltimport torchfrom rdkit import rdBasefrom rdkit import Chemfrom typing import List, Sequenceimport loggingfrom copy import copyfrom collections import OrderedDictdef set_torch_seed_to_all_gens(_):    seed = torch.initial_seed() % (2 ** 32 - 1)    random.seed(seed)    np.random.seed(seed)def batch_to_device(batch, device):    return [        x.to(device) if isinstance(x, torch.Tensor) else x        for x in batch    ]def mol_random_mask(sentence, mask):    real_token = sentence    mask_tokens = sentence    for i, token in enumerate(mask_tokens):        if i == 0 or i == 1 or i == len(mask_tokens) - 1:            continue        prob = random.random()        if prob < 0.15:            mask_tokens[i] = mask    return {'mask': mask_tokens, 'real': real_token}def prot_random_mask(sentence, vocab):    masked_sentence = copy(sentence)    input_mask = torch.ones(len(sentence))    labels = torch.zeros(len(sentence)) - 1    for i, token in enumerate(sentence):        # Tokens begin and end with start_token and stop_token, ignore these        if i == 0 or i == len(sentence) - 1:            continue        prob = 1  # random.random()        if prob < 0.15:            prob /= 0.15            labels[i] = token            if prob < 0.8:                # 80% random change to mask token                token = vocab['<mask>']            elif prob < 0.9:                # 10% chance to change to random token                token = list(vocab.values())[random.randint(0, len(vocab) - 1)]            else:                # 10% chance to keep current token                pass            masked_sentence[i] = token            input_mask[i] = -1    # return masked_tokens, tokens, labels    return {'input_ids': masked_sentence, 'input_mask': input_mask, 'labels': labels}def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:    batch_size = len(sequences)    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()    if dtype is None:        dtype = sequences[0].dtype    if isinstance(sequences[0], np.ndarray):        array = np.full(shape, constant_value, dtype=dtype)    elif isinstance(sequences[0], torch.Tensor):        array = torch.full(shape, constant_value, dtype=dtype)    for arr, seq in zip(array, sequences):        arrslice = tuple(slice(dim) for dim in seq.shape)        arr[arrslice] = seq    return arrayclass SpecialTokens:    bos = '<bos>'    eos = '<eos>'    pad = '<pad>'    unk = '<unk>'    cls = '<cls>'    mask = '<mask>'def smiles_tokenizer(smile):    "Tokenizes SMILES string"    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"    regezz = re.compile(pattern)    smile = smile.strip()    tokens = [token for token in regezz.findall(smile)]    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))    return tokensdef get_tokens(data):    all_tokens = set()    for smi in data:        token = set(smiles_tokenizer(smi))        all_tokens.update(token)    return all_tokenslogger = logging.getLogger(__name__)IUPAC_CODES = OrderedDict([    ('Ala', 'A'),    ('Asx', 'B'),    ('Cys', 'C'),    ('Asp', 'D'),    ('Glu', 'E'),    ('Phe', 'F'),    ('Gly', 'G'),    ('His', 'H'),    ('Ile', 'I'),    ('Lys', 'K'),    ('Leu', 'L'),    ('Met', 'M'),    ('Asn', 'N'),    ('Pro', 'P'),    ('Gln', 'Q'),    ('Arg', 'R'),    ('Ser', 'S'),    ('Thr', 'T'),    ('Sec', 'U'),    ('Val', 'V'),    ('Trp', 'W'),    ('Xaa', 'X'),    ('Tyr', 'Y'),    ('Glx', 'Z')])IUPAC_VOCAB = OrderedDict([    ("<pad>", 0),    ("<mask>", 1),    ("<cls>", 2),    ("<sep>", 3),    ("<unk>", 4),    ("A", 5),    ("B", 6),    ("C", 7),    ("D", 8),    ("E", 9),    ("F", 10),    ("G", 11),    ("H", 12),    ("I", 13),    ("K", 14),    ("L", 15),    ("M", 16),    ("N", 17),    ("O", 18),    ("P", 19),    ("Q", 20),    ("R", 21),    ("S", 22),    ("T", 23),    ("U", 24),    ("V", 25),    ("W", 26),    ("X", 27),    ("Y", 28),    ("Z", 29)])UNIREP_VOCAB = OrderedDict([    ("<pad>", 0),    ("M", 1),    ("R", 2),    ("H", 3),    ("K", 4),    ("D", 5),    ("E", 6),    ("S", 7),    ("T", 8),    ("N", 9),    ("Q", 10),    ("C", 11),    ("U", 12),    ("G", 13),    ("P", 14),    ("A", 15),    ("V", 16),    ("I", 17),    ("F", 18),    ("Y", 19),    ("W", 20),    ("L", 21),    ("O", 22),    ("X", 23),    ("Z", 23),    ("B", 23),    ("J", 23),    ("<cls>", 24),    ("<sep>", 25)])class TAPETokenizer():    r"""TAPE Tokenizer. Can use different vocabs depending on the model.    """    def __init__(self, vocab: str = 'iupac'):        if vocab == 'iupac':            self.vocab = IUPAC_VOCAB        elif vocab == 'unirep':            self.vocab = UNIREP_VOCAB        self.tokens = list(self.vocab.keys())        self._vocab_type = vocab        assert self.start_token in self.vocab and self.stop_token in self.vocab    @property    def vocab_size(self) -> int:        return len(self.vocab)    @property    def start_token(self) -> str:        return "<cls>"    @property    def stop_token(self) -> str:        return "<sep>"    @property    def mask_token(self) -> str:        if "<mask>" in self.vocab:            return "<mask>"        else:            raise RuntimeError(f"{self._vocab_type} vocab does not support masking")    def tokenize(self, text: str) -> List[str]:        return [x for x in text]    def convert_token_to_id(self, token: str) -> int:        """ Converts a token (str/unicode) in an id using the vocab. """        try:            return self.vocab[token]        except KeyError:            raise KeyError(f"Unrecognized token: '{token}'")    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:        return [self.convert_token_to_id(token) for token in tokens]    def convert_id_to_token(self, index: int) -> str:        """Converts an index (integer) in a token (string/unicode) using the vocab."""        try:            return self.tokens[index]        except IndexError:            raise IndexError(f"Unrecognized index: '{index}'")    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:        return [self.convert_id_to_token(id_) for id_ in indices]    def convert_tokens_to_string(self, tokens: str) -> str:        """ Converts a sequence of tokens (string) in a single string. """        return ''.join(tokens)    def add_special_tokens(self, token_ids: List[str]) -> List[str]:        """        Adds special tokens to the a sequence for sequence classification tasks.        A BERT sequence has the following format: [CLS] X [SEP]        """        cls_token = [self.start_token]        sep_token = [self.stop_token]        return cls_token + token_ids + sep_token    def encode(self, text: str) -> np.ndarray:        tokens = self.tokenize(text)        tokens = self.add_special_tokens(tokens)        token_ids = self.convert_tokens_to_ids(tokens)        return np.array(token_ids, np.int64)    @classmethod    def from_pretrained(cls, **kwargs):        return cls()class WordVocab:    @classmethod    def from_data(cls, data, *args, **kwargs):        chars = get_tokens(data)        return cls(chars, *args, **kwargs)    def __init__(self, chars, ss=SpecialTokens):        if (ss.bos in chars) or (ss.eos in chars) or \                (ss.pad in chars) or (ss.unk in chars):            raise ValueError('SpecialTokens in chars')        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk, ss.cls, ss.mask]        self.ss = ss        self.c2i = {c: i for i, c in enumerate(all_syms)}        self.i2c = {i: c for i, c in enumerate(all_syms)}    def __len__(self):        return len(self.c2i)    @property    def bos(self):        return self.c2i[self.ss.bos]    @property    def eos(self):        return self.c2i[self.ss.eos]    @property    def pad(self):        return self.c2i[self.ss.pad]    @property    def unk(self):        return self.c2i[self.ss.unk]    @property    def cls(self):        return self.c2i[self.ss.cls]    @property    def mask(self):        return self.c2i[self.ss.mask]    def char2id(self, char):        if char not in self.c2i:            return self.unk        return self.c2i[char]    def id2char(self, id):        if id not in self.i2c:            return self.ss.unk        return self.i2c[id]    def string2ids(self, string, add_bos=False, add_eos=False, add_cls=False):        ids = [self.char2id(c) for c in smiles_tokenizer(string)]        if add_bos:            if add_cls:                ids = [self.cls] + ids            ids = [self.bos] + ids        if add_eos:            ids = ids + [self.eos]        return ids    def ids2string(self, ids, rem_bos=True, rem_eos=True):        if len(ids) == 0:            return ''        if rem_bos and ids[0] == self.bos:            ids = ids[1:]        if rem_eos and ids[-1] == self.eos:            ids = ids[:-1]        string = ''.join([self.id2char(id) for id in ids])        return stringclass CharVocab:    @classmethod    def from_data(cls, data, *args, **kwargs):        chars = set()        for string in data:            chars.update(string)        return cls(chars, *args, **kwargs)    def __init__(self, chars, ss=SpecialTokens):        if (ss.bos in chars) or (ss.eos in chars) or \                (ss.pad in chars) or (ss.unk in chars):            raise ValueError('SpecialTokens in chars')        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]        self.ss = ss        self.c2i = {c: i for i, c in enumerate(all_syms)}        self.i2c = {i: c for i, c in enumerate(all_syms)}    def __len__(self):        return len(self.c2i)    @property    def bos(self):        return self.c2i[self.ss.bos]    @property    def eos(self):        return self.c2i[self.ss.eos]    @property    def pad(self):        return self.c2i[self.ss.pad]    @property    def unk(self):        return self.c2i[self.ss.unk]    def char2id(self, char):        if char not in self.c2i:            return self.unk        return self.c2i[char]    def id2char(self, id):        if id not in self.i2c:            return self.ss.unk        return self.i2c[id]    def string2ids(self, string, add_bos=False, add_eos=False):        ids = [self.char2id(c) for c in string]        if add_bos:            ids = [self.bos] + ids        if add_eos:            ids = ids + [self.eos]        return ids    def ids2string(self, ids, rem_bos=True, rem_eos=True):        if len(ids) == 0:            return ''        if rem_bos and ids[0] == self.bos:            ids = ids[1:]        if rem_eos and ids[-1] == self.eos:            ids = ids[:-1]        string = ''.join([self.id2char(id) for id in ids])        return stringclass OneHotVocab(CharVocab):    def __init__(self, *args, **kwargs):        super(OneHotVocab, self).__init__(*args, **kwargs)        self.vectors = torch.eye(len(self.c2i))def mapper(n_jobs):    '''    Returns function for map call.    If n_jobs == 1, will use standard map    If n_jobs > 1, will use multiprocessing pool    If n_jobs is a pool object, will return its map function    '''    if n_jobs == 1:        def _mapper(*args, **kwargs):            return list(map(*args, **kwargs))        return _mapper    if isinstance(n_jobs, int):        pool = Pool(n_jobs)        def _mapper(*args, **kwargs):            try:                result = pool.map(*args, **kwargs)            finally:                pool.terminate()            return result        return _mapper    return n_jobs.mapclass Logger(UserList):    def __init__(self, data=None):        super().__init__()        self.sdata = defaultdict(list)        for step in (data or []):            self.append(step)    def __getitem__(self, key):        if isinstance(key, int):            return self.data[key]        if isinstance(key, slice):            return Logger(self.data[key])        ldata = self.sdata[key]        if isinstance(ldata[0], dict):            return Logger(ldata)        return ldata    def append(self, step_dict):        super().append(step_dict)        for k, v in step_dict.items():            self.sdata[k].append(v)    def save(self, path):        df = pd.DataFrame(list(self))        df.to_csv(path, index=None)class LogPlotter:    def __init__(self, log):        self.log = log    def line(self, ax, name):        if isinstance(self.log[0][name], dict):            for k in self.log[0][name]:                ax.plot(self.log[name][k], label=k)            ax.legend()        else:            ax.plot(self.log[name])        ax.set_ylabel('value')        ax.set_xlabel('epoch')        ax.set_title(name)    def grid(self, names, size=7):        _, axs = plt.subplots(nrows=len(names) // 2, ncols=2,                              figsize=(size * 2, size * (len(names) // 2)))        for ax, name in zip(axs.flatten(), names):            self.line(ax, name)class CircularBuffer:    def __init__(self, size):        self.max_size = size        self.data = np.zeros(self.max_size)        self.size = 0        self.pointer = -1    def add(self, element):        self.size = min(self.size + 1, self.max_size)        self.pointer = (self.pointer + 1) % self.max_size        self.data[self.pointer] = element        return element    def last(self):        assert self.pointer != -1, "Can't get an element from an empty buffer!"        return self.data[self.pointer]    def mean(self):        if self.size > 0:            return self.data[:self.size].mean()        return 0.0def disable_rdkit_log():    rdBase.DisableLog('rdApp.*')def enable_rdkit_log():    rdBase.EnableLog('rdApp.*')def get_mol(smiles_or_mol):    '''    Loads SMILES/molecule into RDKit's object    '''    if isinstance(smiles_or_mol, str):        if len(smiles_or_mol) == 0:            return None        mol = Chem.MolFromSmiles(smiles_or_mol)        if mol is None:            return None        try:            Chem.SanitizeMol(mol)        except ValueError:            return None        return mol    return smiles_or_mol@torch.no_grad()def concat_all_gather(tensor):    """    Performs all_gather operation on the provided tensors.    *** Warning ***: torch.distributed.all_gather has no gradient.    """    tensors_gather = [torch.ones_like(tensor)                      for _ in range(torch.distributed.get_world_size())]    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)    output = torch.cat(tensors_gather, dim=0)    return output