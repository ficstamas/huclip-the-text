import os

import transformers
import pickle
# from multi_rake import Rake
import logging
import numpy as np
import torch
import huspacy
import clip
from paddle.utils.network import download_url
from PIL import Image
from typing import Literal, List, Tuple, Dict
import warnings
import pkg_resources
warnings.filterwarnings("ignore")

AVAILABLE_MODELS = {
    'M-BERT-Distil-40': {
        'model_name': 'M-CLIP/M-BERT-Distil-40',
        'tokenizer_name': 'M-CLIP/M-BERT-Distil-40',
        'head_name': 'M-BERT Distil 40 Linear Weights.pkl',
        'weight_link': 'https://github.com/ficstamas/huclip-the-text/releases/download/weights/M-BERT.Distil.40.Linear.Weights.pkl',
        'image_model': 'RN50x4'
    },
    'M-BERT-Base-69': {
        'model_name': 'M-CLIP/M-BERT-Base-69',
        'tokenizer_name': 'M-CLIP/M-BERT-Base-69',
        'head_name': 'M-BERT-Base-69 Linear Weights.pkl',
        'weight_link': 'https://github.com/ficstamas/huclip-the-text/releases/download/weights/M-BERT-Base-69.Linear.Weights.pkl',
        'image_model': 'RN50x4'
    },
    'M-BERT-Base-ViT-B': {
        'model_name': 'M-CLIP/M-BERT-Base-ViT-B',
        'tokenizer_name': 'M-CLIP/M-BERT-Base-ViT-B',
        'head_name': 'M-BERT-Base-69-ViT Linear Weights.pkl',
        'weight_link': 'https://github.com/ficstamas/huclip-the-text/releases/download/weights/M-BERT-Base-69-ViT.Linear.Weights.pkl',
        'image_model': 'ViT-B/32'
    },
}

HUSPACY_MODELS = {
    "cnn": {
        "model_name": "hu_core_news_lg",
        "module_name": "hu-core-news-lg",
        "extractor_function": "extract_keywords_spacy"
    },
    "hubert": {
        "model_name": "hu_core_news_trf",
        "module_name": "hu-core-news-trf",
        "extractor_function": "extract_keywords_spacy_transformers"
    }
}

_IGNORED_POS = ["PUNCT", "CCONJ"]

log = logging.getLogger("huclip")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'))
log.handlers = []
log.addHandler(ch)
log.setLevel(logging.DEBUG)


class MultilingualClip(torch.nn.Module):
    def __init__(self, model_name: str, tokenizer_name: str, head_name: str, weight_link: str, weights_dir='weights/',
                 device='cpu', **kwargs):
        """
        Every parameter is defined in `AVAILABLE_MODELS` except `weights_dir`
        :param model_name: name of the model
        :param tokenizer_name: name of the tokenizer
        :param head_name: name of the projection head
        :param weight_link: download link pointing to the weights
        :param weights_dir: folder to store projection weights
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

        os.makedirs(weights_dir, exist_ok=True)

        self.head_path = weights_dir + head_name

        if not os.path.exists(self.head_path):
            download_url(weight_link, output=os.path.join(weights_dir, head_name),
                         tqdm_params={'desc': 'Downloading Weights:'})

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.transformer = transformers.AutoModel.from_pretrained(model_name).to(self.device)
        self.clip_head = None
        self._load_head()

    def forward(self, txt: str):
        """
        Embedding input text
        :param txt: input text
        :return:
        """
        txt_tok = self.tokenizer(txt, padding=True, return_tensors='pt')
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.clip_head(embs)

    def _load_head(self):
        """
        Loading weights into the projection head
        :return:
        """
        with open(self.head_path, 'rb') as f:
            lin_weights = pickle.loads(f.read())

        head_shape = lin_weights[0].shape
        self.clip_head = torch.nn.Linear(in_features=head_shape[0], out_features=head_shape[1]).to(self.device)
        self.clip_head.weight = torch.nn.Parameter(torch.tensor(lin_weights[0]).float().t()).to(self.device)
        self.clip_head.bias = torch.nn.Parameter(torch.tensor(lin_weights[1]).float()).to(self.device)


def _load_language_model(name: str, device: str):
    """
    Loading Language Model
    :param name: name of the model
    :return:
    """
    config = AVAILABLE_MODELS[name]
    return MultilingualClip(**config, device=device)


class KeywordCLIP:
    def __init__(self, model_name: str = 'M-BERT-Distil-40', spacy_model_name: Literal['cnn', 'hubert'] = 'cnn',
                 device='cpu'):
        """
        Loading everything
        :param model_name: name of the used model
        :param spacy_model_name: name of the utilized spacy model
        :param device: Model placement
        """
        # self.rake = Rake(
        #     min_chars=3, max_words=5, min_freq=1,
        #     language_code='hu', stopwords=None,
        #     lang_detect_threshold=50, max_words_unknown_lang=2,
        #     generated_stopwords_percentile=80, generated_stopwords_max_len=3,
        #     generated_stopwords_min_freq=2
        # )

        log.info(f'Loading language model: HuSpaCy!')
        self.device = device

        spacy_module_name = HUSPACY_MODELS[spacy_model_name]['module_name']
        spacy_extractor_name = HUSPACY_MODELS[spacy_model_name]['extractor_function']
        spacy_model_name = HUSPACY_MODELS[spacy_model_name]['model_name']

        if not self._is_installed(spacy_module_name):
            huspacy.download(model_name=spacy_model_name)

        self.spacy = huspacy.load(name=spacy_model_name)
        self.keyword_extractor_function = getattr(self, spacy_extractor_name)

        language_model_name = AVAILABLE_MODELS[model_name]['model_name']
        log.info(f'Loading language model: {language_model_name}')
        self.language_model = _load_language_model(model_name, device=self.device)
        image_model_name = AVAILABLE_MODELS[model_name]['image_model']
        log.info(f'Loading image model: {image_model_name}')
        self.clip_model, compose = self._load_clip(image_model_name, device=self.device)
        log.info(f'Done!')
        self.compose = lambda img: compose(img).to(self.device)

    @staticmethod
    def available_models():
        return [x for x in AVAILABLE_MODELS]

    @staticmethod
    def _is_installed(pkg_name: str):
        """
        Checks whether a package is installed
        :param pkg_name:
        :return:
        """
        return pkg_name in {pkg.key for pkg in pkg_resources.working_set}

    @staticmethod
    def _load_clip(image_model_name: str, device: str):
        """
        Loading image model
        :param image_model_name: name of the image model
        :return:
        """
        return clip.load(image_model_name, device=device)

    def extract_keywords_spacy(self, sentence: str) -> List[Tuple[str, float]]:
        """
        Extracting potential keywords using SpaCy
        :param sentence: Input text
        :return: List of tuples containing <string, score> pairs
        """
        doc = self.spacy(sentence)
        # find CCONJ's root
        base_ = [(x.head.head, x.head.head.head) for x in doc if x.pos_ == 'CCONJ']
        if len(base_) < 1:
            return [("Sajnos nem ??rtem a k??rd??st. Megpr??b??ln??d ??jra?", 0.0)]

        root, context = base_[0]
        context_ = [c for c in context.children if c.dep_ == 'obj']
        context = context_[0] if len(context_) > 0 else context
        Q = [x for x in context.children if 'amod' in x.dep_]  # amod:att
        if root not in Q:
            Q.append(root)
        keywords = []

        element = None
        while len(Q) > 0:
            element = Q.pop()
            children = element.children
            kw = [element.text, context.lemma_]

            for child in reversed([x for x in children]):
                if child.pos_ not in _IGNORED_POS:
                    if child.idx < element.idx:
                        if child.pos_ != element.pos_:
                            kw.insert(0, child.text)
                        else:
                            Q.append(child)
                    else:
                        Q.append(child)
            keywords.append((" ".join(kw), 0.0))

        return keywords

    def extract_keywords_spacy_transformers(self, sentence: str) -> List[Tuple[str, float]]:
        """
        Extracting potential keywords using SpaCy
        :param sentence: Input text
        :return: List of tuples containing <string, score> pairs
        """
        doc = self.spacy(sentence)
        # find CCONJ's root
        base_ = [(x.head.head, x.head.head.head) for x in doc if x.pos_ == 'CCONJ']
        if len(base_) < 1:
            return [("Sajnos nem ??rtem a k??rd??st. Megpr??b??ln??d ??jra?", 0.0)]

        root, context = base_[0]
        Q = [x for x in context.children if 'amod' in x.dep_]  # amod:att
        if root not in Q:
            Q.append(root)
        keywords = []

        element = None
        while len(Q) > 0:
            element = Q.pop()
            children = element.children
            kw = [element.text, ]

            for child in reversed([x for x in children]):
                if child.pos_ not in _IGNORED_POS:
                    if child.idx < element.idx:
                        if child.pos_ != element.pos_:
                            kw.insert(0, child.text)
                        else:
                            Q.append(child)
                    else:
                        Q.append(child)
            keywords.append((" ".join(kw), 0.0))

        return keywords

    # def extract_keywords_rake(self, sentence: str) -> List[Tuple[str, float]]:
    #     """
    #     Extracting potential keywords using MultiRake
    #     :param sentence: input text
    #     :return: List of tuples containing <string, score> pairs
    #     """
    #     return self.rake.apply(sentence)

    def embedding(self, image: Image.Image, text: str, keyword_extraction_method: Literal["spacy"] = "spacy"
                  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Embedding the input image and text
        :param image: PIL Image
        :param text: sentence
        :param keyword_extraction_method: method to extract keywords
        :return:
        """
        composed_image = torch.stack([self.compose(image), ])

        if keyword_extraction_method == "rake":
            raise NotImplementedError(":(")  # extracted_keywords = self.extract_keywords_rake(text)
        elif keyword_extraction_method == "spacy":
            extracted_keywords = self.keyword_extractor_function(text)
        else:
            raise NotImplementedError(":(")

        log.info(f'Extracted keywords: {[kw[0] for kw in extracted_keywords]}')

        with torch.no_grad():
            image_emb = self.clip_model.encode_image(composed_image).float().to(self.device)
            text_emb = {}
            for kw, _ in extracted_keywords:
                text_emb[kw] = self.language_model(text + " " + kw)
        return image_emb, text_emb

    def compare_embeddings(self, image_emb: torch.Tensor, text_emb: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculating logits
        :param image_emb: image embedding
        :param text_emb: dict of <keyword, embedding> pair
        :return: logits
        """
        logit_scale = self.clip_model.logit_scale.exp().float().to('cpu')

        text_embeddings = []
        for kw in text_emb:
            text_embeddings.append(text_emb[kw])

        text_embeddings = torch.vstack(text_embeddings)

        language_logits = self._cosine_similarity(logit_scale, image_emb.to('cpu'), text_embeddings.to('cpu'))
        return language_logits

    @staticmethod
    def _cosine_similarity(logit_scale: torch.Tensor, img_embs: torch.Tensor, txt_embs: torch.Tensor):
        """
        Calculates cosine similarity between image and text embeddings
        :param logit_scale: temperature
        :param img_embs: image embedding
        :param txt_embs: text embedding
        :return: logits
        """
        # normalized features
        image_features = img_embs / img_embs.norm(dim=-1, keepdim=True)
        text_features = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits = logit_scale * image_features @ text_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits

    def evaluate(self, image: Image.Image, text: str, keyword_extraction_method: Literal["spacy", "rake"] = "spacy"
                 ) -> Tuple[str, float, Dict]:
        """
        Evaluates model on provided Image and text pair
        :param image: PIL Image
        :param text: input text
        :param keyword_extraction_method: method to extract keywords
        :return: Best keyword with its probability
        """
        image_emb, text_emb = self.embedding(image, text, keyword_extraction_method)
        keywords = {i: key for i, key in enumerate(text_emb)}
        language_logits = self.compare_embeddings(image_emb, text_emb)
        probs = self.probabilities(language_logits)

        out = {}
        for i, kw in keywords.items():
            log.info(f"Probability of the answer '{kw}' is {probs[i, 0]}")
            out[kw] = probs[i, 0]
        answer = np.argmax(probs, axis=0)[0]
        return keywords[answer], probs[answer][0], out

    @staticmethod
    def probabilities(language_logits: torch.Tensor) -> np.ndarray:
        """
        Calculates probabilities from logits
        :param language_logits: logits
        :return: probabilities for each keyword
        """
        probs_ = language_logits.softmax(dim=-1).cpu().detach().numpy()
        return probs_.T
