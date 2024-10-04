def prompt_formatter(sample):
    return f"""<s>### Instruction:
당신은 대화를 요약해주는 유능한 AI입니다. \
당신의 임무는 다음에 나오는 대화를 요약하는 것입니다. \
당신의 대답은 오직 제공된 대화에만 근거해야 합니다.

### Dialogue:
{sample['dialogue']}

### Summary:
{sample['summary']}</s>"""

def generate_prompt(example):
    output_texts = []
    for i in range(len(example['dialogue'])):
        prompt = f"""<s>### Instruction:
당신은 대화를 요약해주는 유능한 AI입니다. \
당신의 임무는 다음에 나오는 대화를 요약하는 것입니다. \
당신의 대답은 오직 제공된 대화에만 근거해야 합니다.

### Dialogue:
{example['dialogue'][i]}

### Summary:
"""
        output_texts.append(prompt)
    return output_texts