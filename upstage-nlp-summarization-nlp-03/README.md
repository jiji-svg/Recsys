# Dialogue Summarization Competitions
## Team

<table>
<tr>
<td> <div align=center> 팀장 </div> </td>
<td> <div align=center> 팀원</div> </td>
<td> <div align=center> 팀원</div> </td>
<td> <div align=center> 팀원</div> </td>
<td> <div align=center> 팀원</div> </td>
</tr>
<tr>
<td> <div align=center> <b>김지완</b> </div> </td>
<td> <div align=center> <b>김도후</b> </div> </td>
<td> <div align=center> <b>박주혁</b> </div> </td>
<td> <div align=center> <b>정혜윤</b> </div> </td>
<td> <div align=center> <b>최용빈</b> </div> </td>
</tr>
<tr>
<td> <img alt="Github" src="https://avatars.githubusercontent.com/u/121218201?v=4" width="150" height="150"/> </td>
<td> <img alt="Github" src="https://avatars.githubusercontent.com/u/113418415?v=4" width="150" height="150"/> </td>
<td> <img alt="Github" src="https://avatars.githubusercontent.com/u/156395101?v=4" width="150" height="150"/> </td>
<td> <img alt="Github" src="https://avatars.githubusercontent.com/u/118159352?v=4" width="150" height="150"/> </td>
<td> <img alt="Github" src="https://avatars.githubusercontent.com/u/64704608?v=4" width="150" height="150"/> </td>
</tr>
<tr>
<td> <div align=center> <a href="https://github.com/kimjiwan-svg"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a></div> </td>
<td> <div align=center> <a href="https://github.com/kimdohoo1102"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a></div> </td>
<td> <div align=center> <a href="https://github.com/Leodevdd"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a></div> </td>
<td> <div align=center> <a href="https://github.com/Hye-yoonJeong"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a></div> </td>
<td> <div align=center> <a href="https://github.com/whybe-choi"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a></div> </td>
</tr>
</table>

## 0. Overview

<img width="1082" alt="image" src="https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/assets/64704608/2aeacc9a-c9f3-439b-8531-fed303e80e0c">

> Dialogue Summarization 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다. 일상생활에서 대화는 항상 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다. 그러나 하나의 대화에서도 관점, 주제별로 정리하면 수 많은 요약을 만들 수 있습니다. 대화를 하는 도중에 이를 요약하게 되면 대화에 집중할 수 없으며, 대화 이후에 기억에 의존해 요약하게 되면 오해나 누락이 추가되어 주관이 많이 개입되게 됩니다. 이를 돕기 위해, 우리는 이번 대회에서 일상 대화를 바탕으로 요약문을 생성하는 모델을 구축합니다!

### Environment
- Vscode, RTX 3090 server

## 1. Competiton Info
### Overview
- Dialogue Summarization
- Task : Summarization
- Evaluation Metric : _ROUGE(Recall-Oriented Understudy for Gisting Evaluation)_

$$ Score = \frac{\sum\limits_{i}^N ROUGE-1-F1(pred, gold_i)}{N} +\frac{\sum\limits_{i}^N ROUGE-2-F1(pred, gold_i)}{N}+ \frac{\sum\limits_{i}^N ROUGE-L-F1(pred, gold_i)}{N} $$

### Timeline

- Start Date : May 13, 2024
- Final submission deadline : May 27, 2024 (13:00)

## 2. Components

### Directory

```
project
├── LICENSE
├── README.md
├── configs
│   ├── config.yaml
│   ├── logger
│   ├── model
│   └── train
├── data
│   ├── dev.csv
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── data_augmentation.py
├── docs
│   └── [패스트캠퍼스]Upstage_AI_Lab_2기_NLP_경진대회_ 3조 발표.pdf
├── main.py
├── notebooks
│   ├── OPEN-SOLAR-KO-10.7B.ipynb
│   ├── eda.ipynb
│   ├── few_shot.ipynb
│   ├── preprocess.ipynb
│   └── t5-base-korean-summarization.ipynb
├── requirements.txt
└── src
    ├── dataset.py
    ├── inference.py
    ├── model.py
    ├── train.py
    └── utils.py
```

## 3. Data descrption

### Dataset overview

- train : 12457
- dev : 499
- test : 250
- hidden-test : 249

### EDA
- Data distribution of dialogue/topic/summary length.
- Analysis on special tokens in dialogues.
- Summary ratio before and after tokenizing. (train/dev)
    - train

        ![image](https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/assets/118159352/dbfcdf9f-ab9a-4f5b-a20a-b267f6a202e9)
    - dev

        ![image](https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/assets/118159352/5a3ded45-edfe-4898-8de2-4040f0b8b7a1)

### Data Processing
- train : 12457 → 12403
	- dialogue with more than 3 special tokens.
	- summary ratio over 0.5
- dev : 499 → 486
	- dialogue with more than 2 special tokens.
	- summary ratio over 0.5

### Data Augmentation
- Data augmentation using Cohere API
    - original data

        ![image](https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/assets/118159352/a954b61c-941d-433d-a886-29b6ee1d5fd6)


    - augmented data

        ![image](https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/assets/118159352/42870186-c49d-4cf9-ac28-d910cc301c56)


## 4. Modeling

### Model description
- Solar : beomi/OPEN-SOLAR-KO-10.7B + 4-bit quantization + LoRA

### Modeling Process

![upstage_nlp_modeling_process-05](https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/assets/118159352/4a9d1432-537f-4019-a23d-4a07e06e4f76)


## 5. Result

### Leader Board

- Public Score 🥉

<img width="1036" alt="image" src="https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/assets/64704608/1fd440c2-a780-4d1f-8cc9-c2b4a783f8a5">

- Private Score 🥈

<img width="1054" alt="image" src="https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/assets/64704608/b3f7a364-a709-4e12-8810-f3308d40050e">

### Presentation

- [3조 CV 발표자료](https://github.com/UpstageAILab2/upstage-nlp-summarization-nlp-03/blob/main/docs/%5B%ED%8C%A8%EC%8A%A4%ED%8A%B8%EC%BA%A0%ED%8D%BC%EC%8A%A4%5DUpstage_AI_Lab_2%EA%B8%B0_NLP_%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C_%203%EC%A1%B0%20%EB%B0%9C%ED%91%9C.pdf)

## 6. How to use
### train
- default
```
python main.py mode=train
```

- with another config
```
python main.py mode=train train.num_epochs=5 train.learning_rate=2e-4
```

### inference
```
python main.py mode=inference
```


## etc

### Meeting Log

- [<img src="https://img.shields.io/badge/Notion-000000?style=plastic&logo=Notion&logoColor=white"/>](https://www.notion.so/5a2dbdf8b49a4cd4858315aba839ba8a?v=f82ac3c6dd8c426096309a043d88a284&pvs=4)
