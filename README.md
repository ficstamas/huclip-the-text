# The Project

Object identification and question answering. Main goal of the project was to make decisions about images based on the provided questions. The question MUST contain the possible answers like: 

'_Is this apple **red** or **blue**?_',

Model returns an answer based on what's on the image. We extract the keywords with the **Hungarian Spacy**, so you can just change that part to adapt the model to your language. Keyword extraction part relies on the `CCONJ` part-of-speech tag so you also need to include a word that fulfils that role (like `or`).

# Installation & Requirements

* Python = 3.9.*
```shell
pip install git+https://github.com/ficstamas/huclip-the-text.git
```

# Example Usage

```python
from huclip_the_text.model.clip import KeywordCLIP
from PIL import Image

model = KeywordCLIP(model_name='M-BERT-Base-ViT-B')
img = Image.open('bananas.jpg')
out = model.evaluate(img, 'Sárga, kerek vagy lila banánt látsz?')

# Output:
# Probability of the answer 'Sárga banán' is 0.601322591304779
# Probability of the answer 'kerek banán' is 0.20016320049762726
# Probability of the answer 'lila banán' is 0.19851425290107727
```

# Available Models

Pre-trained models and projection weights are from [MultilingualCLIP](https://github.com/FreddeFrallan/Multilingual-CLIP/)

| Name               |Language Model|Model Base|Vision Model | Pre-trained Languages | Target Languages | #Parameters|
|--------------------|:-----: |:-----: |:-----: |:-----: |:-----: |:-----: |
| M-BERT-Distil-40   | [M-BERT Distil 40](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/M-BERT%20Distil%2040)   | [M-BERT Distil](https://huggingface.co/bert-base-multilingual-uncased)|  RN50x4 | [101 Languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) | [40 Languages](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/Model%20Cards/M-BERT%20Distil%2040/Fine-Tune-Languages.md) | 66 M|
| M-BERT-Base-69     | [M-BERT Base 69](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/M-BERT%20Base%2069)       | [M-BERT Base](https://huggingface.co/bert-base-multilingual-uncased)|RN50x4 | [101 Languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) | 68 Languages | 110 M|
| M-BERT-Base-ViT-B  | [M-BERT Base ViT-B](https://github.com/FreddeFrallan/Multilingual-CLIP/tree/main/Model%20Cards/M-BERT%20Base%20ViT-B) | [M-BERT Base](https://huggingface.co/bert-base-multilingual-uncased)|ViT-B/32 | [101 Languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) | 68 Languages | 110 M|

# Related Works

- MultilingualCLIP model: https://github.com/FreddeFrallan/Multilingual-CLIP/
- HuSpaCy: https://github.com/huspacy/huspacy/
- Multi-Rake: https://github.com/vgrabovets/multi_rake/
