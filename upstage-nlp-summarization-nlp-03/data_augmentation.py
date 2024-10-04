import os
import random
import time
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# 현재 스크립트의 디렉토리 경로를 얻기
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data", "dev.csv")

# 데이터 로드
valid_df = pd.read_csv(data_path)

# 데이터 증강 시 사용할 데이터 형태 정의
class DialogueData(BaseModel):
    dialogue: str = Field(description='dialogue between people')
    summary: str = Field(description="summary for dialogue")

# 데이터프레임에서 examples 생성 함수
def create_examples_from_dataframe(df):
    examples = []
    for _, row in df.iterrows():
        example = {
            "dialogue": row["dialogue"],
            "summary": row["summary"],
        }
        examples.append(example)
    return examples

# 데이터프레임에서 examples 생성
valid_data = create_examples_from_dataframe(valid_df)

# 3개의 예시를 임의로 추출
few_shot_examples = random.sample(valid_data, 3)

# 프롬프트 템플릿 설정
prompt_template = """
[CONTEXT]
당신은 대화와 요약을 생성하는 유능한 AI입니다.
당신의 임무는 [EXAMPLE]와 유사한 대화와 요약을 만드는 것입니다.
대화와 요약 스타일을 유지해야 하지만 동일한 대화와 요약을 생성해서는 안됩니다.
반드시 대화와 요약을 모두 생성해야 합니다.

[STEP]
1. {{dialogue}}
대화와 관련된 내용이 들어갑니다. 주어진 대화들과 유사한 형태의 대화를 생성합니다.
2. {{summary}}
요약과 관련된 내용이 들어갑니다. 생성된 대화로부터 예시와 비슷한 요약 스타일을 가지는 요약을 생성합니다.

[EXAMPLE]
{few_shot}

[INSTRUCTION]
주어진 대화와 유사한 스타일로 새로운 대화와 요약을 하나만 만들어주세요.

답변은 아래와 같은 json 포맷을 따라야합니다.
{{ 
    "dialogue" : {{dialogue}},
    "summary" : {{summary}}
}}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Cohere API 설정
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
chat_cohere = ChatCohere(api_key=cohere_api_key, temperature=0.2, max_tokens=512)
chain = prompt | chat_cohere | JsonOutputParser(pydentic_object=DialogueData)

# 데이터 증강 루프
dialogues = []

for i in tqdm(range(1000)):
    try:
        few_shot_examples = random.sample(valid_data, 3)

        few_shot = f"""
{few_shot_examples[0]}

{few_shot_examples[1]}

{few_shot_examples[2]}
"""
        response = chain.invoke({
            "few_shot": few_shot
        })
        dialogues.append(response)

        if i % 100 == 0:
            print("=" * 25, f"[ {i}번째 생성 데이터 ]", "=" * 25)
            print(response['dialogue'])
            print(response['summary'])

        # 요청 간 지연을 추가하여 429 오류 방지
        time.sleep(1)  # 1초 지연 (필요에 따라 조정 가능)

    except Exception as e:
        # 에러 발생 시, 해당 인덱스와 에러 메시지를 출력하고 계속 진행
        print(f"Error at iteration {i}: {e}")
        continue

# 증강한 데이터를 합쳐서 저장
augmented_df = pd.DataFrame(dialogues)
combined_df = pd.concat([valid_df, augmented_df], ignore_index=True)
combined_df.to_csv("./data/valid_augmented.csv", index=False)