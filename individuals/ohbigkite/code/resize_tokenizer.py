# snunlp/KR-ELECTRA-discriminator 모델의 토크나이저에 <PERSON> 토큰을 추가해주고, vocab에 새로운 단어들을 추가해 resize 한 모델을 저장하여 사용할 수 있게 해주는 코드입니다.

from transformers import ElectraModel
from transformers import ElectraTokenizer


tokenizer = ElectraTokenizer.from_pretrained("snunlp/KR-ELECTRA-discriminator")
model = ElectraModel.from_pretrained("snunlp/KR-ELECTRA-discriminator")


# UNK TOKENS로 인식되었던 단어들 vocab에 추가
new_unk_tokens = ['보샸',
 '믱',
 '대튱령',
 '빂츠',
 '｀',
 '뵛',
 '역시일본것칻더니일본거내',
 '쏘큩',
 '굠',
 '아늼니꽈',
 '사랑이뤄지길바럤는데',
 '바럤',
 '호호홐',
 '☼문재인정부는',
 '쎴',
 '스트휏',
 '로우라이즠',
 '빂',
 '헐춋류훃',
 '오마이가뜨지져스크롸이스트휏',
 '벴',
 '줸쟝',
 '땈',
 '스쾃',
 '귀염뽀쨕',
 '죔죔',
 '좠',
 '뀰잼',
 '뽀쨕','쵯한']

tokenizer.add_tokens(new_unk_tokens)

# <PERSON> special token으로 추가
new_tokens = "<PERSON>"

tokenizer.add_special_tokens({"additional_special_tokens" : [new_tokens]})
model.resize_token_embeddings(len(tokenizer))


# 모델 및 토크나이저 저장
tokenizer.save_pretrained('/Users/yejinchoe/Documents/snunlp_vocab_koelc')
model.save_pretrained('/Users/yejinchoe/Documents/snunlp_vocab_koelc')
