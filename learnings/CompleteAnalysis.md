# ReMUD (Reasoning Multimodal Ultrasound Dataset) — Complete Reverse-Engineering Analysis

> Analysis Date: 2026-03-02
> Repository: HAIBU-ReMUD
> Paper: arxiv.org/abs/2506.07837

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Pipeline Architecture (3-Stage)](#2-pipeline-architecture-3-stage)
3. [CoT Prompt Engineering & Generation Templates](#3-cot-prompt-engineering--generation-templates)
4. [Data Cleaning & Quality Assurance](#4-data-cleaning--quality-assurance)
5. [Organ-Specific Routing](#5-organ-specific-routing)
6. [CoT Generation: The Real Mechanism](#6-cot-generation-the-real-mechanism)
7. [Budget Forcing: Deep Dive](#7-budget-forcing-deep-dive)
8. [Accept/Reject & Quality Filtering: Gap Analysis](#8-acceptreject--quality-filtering-gap-analysis)
9. [Actionable Recommendations for Building a Teacher Agent](#9-actionable-recommendations-for-building-a-teacher-agent)

---

## 1. Repository Structure

```
/home/vito/src/ReMUD/
├── data/
│   ├── pdf2jpg.py              # Stage 1: PDF -> JPG pages
│   ├── QwenVLbbox.py           # Stage 2: Bounding box detection (Qwen2.5-VL-7B)
│   ├── VQA.py                  # Stage 3: VQA generation (GPT-4o)
│   ├── filter.ipynb            # Post-processing: ShareGPT format validator
│   ├── demo.json               # Sample SFT training data (ShareGPT format)
│   ├── Filtered_demo.json      # Filtered copy of demo.json
│   ├── demo_images/            # Cropped ultrasound images
│   ├── pdf_files/              # Source Chinese ultrasound textbook PDF
│   └── save_bbox/              # Bounding box results + GPT outputs per page
│       ├── 123/
│       │   ├── page_123.jpg, page_123.json, gpt_page_123.json
│       │   └── page_1231.jpg ... page_1236.jpg
│       └── 124/
│           ├── page_124.jpg, page_124.json, gpt_page_124.json
│           └── page_1241.jpg ... page_1246.jpg
├── eval/
│   ├── eval.py                 # Evaluation with <think>/<answer> tag parsing
│   ├── generate.py             # Empty stub (1 line, no content)
│   └── generate.ipynb          # CoT evaluation with Budget Forcing + Pass@K
├── demo.sh                     # Pipeline entry point
├── requirements.txt            # Conda environment (Python 3.13, PyTorch 2.6, etc.)
└── imgs/
    ├── flow.svg                # Pipeline diagram
    └── Results.png             # Benchmark results
```

### Key Dependencies (from requirements.txt)

| Category | Libraries |
|---|---|
| Deep Learning | `torch 2.6.0+cu126`, `transformers 4.49.0`, `peft`, `trl`, `deepspeed`, `accelerate` |
| VLM / Qwen | `qwen-vl-utils`, `llamafactory` |
| PDF / Image | `pymupdf`, `pillow`, `opencv-python`, `fitz` |
| OCR | `paddleocr` |
| OpenAI API | `openai 1.63.2` |
| Eval | `rouge-chinese`, `nltk`, `scikit-learn` |

---

## 2. Pipeline Architecture (3-Stage)

Driven by `demo.sh`:
```sh
cd ./data
python pdf2jpg.py                                          # Stage 1
python QwenVLbbox.py                                       # Stage 2
python VQA.py --api_key YOUR_API_KEY --api_url YOUR_API_URL  # Stage 3
```

```
┌─────────────────────────────────────────────────────────────┐
│              ReMUD Full Architecture                         │
├──────────────────────┬──────────────────────────────────────┤
│ STAGE 1: SFT Data    │ pdf2jpg -> QwenVLbbox -> VQA.py     │
│ (No CoT)             │ Generates plain Q&A pairs            │
│                      │ using GPT-4o as teacher               │
├──────────────────────┼──────────────────────────────────────┤
│ STAGE 2: RL Training │ LLaMA-Factory GRPO training          │
│ (Teaches CoT)        │ Model learns <think>/<answer>        │
│                      │ tags through reward signals           │
│                      │ (NOT from pre-generated CoT data)    │
├──────────────────────┼──────────────────────────────────────┤
│ STAGE 3: Evaluation  │ generate.ipynb with Budget Forcing   │
│ (Elicits CoT)        │ Injects "等等，" to extend reasoning  │
│                      │ Min thinking length: 96 chars         │
│                      │ Pass@4 scoring                        │
└──────────────────────┴──────────────────────────────────────┘
```

**Critical Finding**: The VQA.py pipeline generates **plain Q&A pairs** (no chain-of-thought). The reasoning ability comes from **GRPO RL training** (Stage 2), not from pre-generated CoT datasets. The CoT is only **elicited and evaluated** in Stage 3 (`generate.ipynb`).

---

## 3. CoT Prompt Engineering & Generation Templates

### 3A. Bounding Box Detection — Qwen2.5-VL-7B System Prompt

**File**: `data/QwenVLbbox.py:292-306`

```python
parser.add_argument(
   '--system_prompt',
    type=str,
    default="""你会得到一页超声pdf书籍的图像，其中有会有超声图像作为插图，请你框出图中的超声图像，以json格式输出其bbox坐标：
        - 请你提取每张图片的标题图例信息，输出在json的'caption'关键字中，，具体输出格式如下：
        ```json
        [
        {"bbox_2d": [765, 394, 891, 550], "label": "超声图像1", "caption": "图片标题，编号，图注信息等"},
        {"bbox_2d": [132, 234, 234, 350], "label": "超声图像2", "caption": "图片标题，编号，图注信息等"}
        ]
        ```
        - 如果没有超声图像，则输出空列表：
        ```json
        []
        ```
        """)
```

**Translation**: "You will receive an image of a page from an ultrasound PDF textbook. Box out the ultrasound images and output their bbox coordinates in JSON format, extracting title/caption/legend info into the `caption` key."

**User Prompt** (`data/QwenVLbbox.py:308-312`):
```python
parser.add_argument(
   '--prompt',
    type=str,
    default="框出图中的超声图像，以json格式输出其bbox坐标和相关文字")
```

**Message Construction** (`data/QwenVLbbox.py:138-157`):
```python
def inference(img_url, prompt, device, system_prompt="You are a helpful assistant", max_new_tokens=2048, **kwargs):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"image": img_url}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

**Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (line 288)

---

### 3B. VQA Generation — GPT-4o System Prompt

**File**: `data/VQA.py:224-231`

```python
parser.add_argument('--system_prompt', type=str, default="""
    你是一名超声医学人工智能助手。你收到了超声医学书籍某一页的图片和一个json列表，json列表与图中标注的超声图像对应，请你完成以任务：
     - 根据json列表每一个元素代表一张超声图，根据json里的caption关键字和这一页与这张超声图对应的文字（如标题，提及的段落），生成一系列对应这张图的问答对话，如"这是什么超声图片？""图中能看到什么特征？"等，以sharegpt的形式给出，回答中不出现图片编号，序号等信息
     - 将输出标准化为的json格式，其具体为sharegpt格式，以供sft微调，即"```json\n[{"conversations": [{"from": "human", "value": "问题"}, {"from": "gpt", "value": "答案"}]}， {"conversations": [{"from": "human", "value": "问题"}, {"from": "gpt", "value": "答案"}]}]\n```"
     - 输出的sharegpt格式要与json列表对应，即输出列表的第一个对应给你的json列表的第一个，然后可以包含多轮的对话，满足sharegpt格式即可
     - 回答中不出现图片编号，序号等信息，对话尽量多轮且详细，对话围绕这张超声图片展开，的答案能在书上找到，即根据书上内容生成问答，并且你可以根据你知道的内容再补充相关知识的问答，每张图的对话相互独立不要出现互相提及
     - 要求输出的json列表元素与给你的json列表元素个数相同即一一对应，即每张超声图对应一段多轮sharegpt格式的对话
    """)
```

**Translation summary**:
1. You are an ultrasound medical AI assistant.
2. Generate multi-turn Q&A dialogues per ultrasound image based on caption + surrounding text.
3. Output in ShareGPT JSON format (`{"conversations": [{"from": "human/gpt", "value": "..."}]}`).
4. Output list must 1:1 correspond to input JSON list.
5. No image numbers/indices in answers. Detailed multi-turn, independent per image.

**Note**: This prompt generates **plain Q&A** — there is NO instruction for step-by-step reasoning, `<think>` tags, or chain-of-thought.

---

### 3C. Per-Sample SFT System Prompt (Injected Into Training Data)

**File**: `data/VQA.py:223,249-251`

```python
parser.add_argument('--vqa_prompt', type=str, default='是一位医学超声助手，请回答相关问题。')
# ...
prompt = args.vqa_prompt
sys_prompt = {"from": "system", "value": prompt}
chat['conversations'].insert(0, copy.deepcopy(sys_prompt))
```

**Translation**: "Is a medical ultrasound assistant, please answer relevant questions."

---

### 3D. API Call Construction (GPT-4o)

**File**: `data/VQA.py:66-84`

```python
messages = []
messages.append({"role": "system", "content": system_prompt})
messages.append({"role": "user", "content": [
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
    },
    {
        "tuype": "text",   # NOTE: typo in original code ("tuype")
        "text": prompt
    }
]})

completion = client.chat.completions.create(
    model=api_model,  # default: "gpt-4o"
    messages=messages
)
```

---

### 3E. The ACTUAL CoT System Prompt (Evaluation Time)

**File**: `eval/generate.ipynb`, cell `b9f2f3dc`

```python
system_prompt = "你是一名人工智能助手，专门研究超声医学领域。你收到了一个超声选择题，请给出你的思考过程，并放在<think>思考过程</think>标签内，只输出一个选项，把选项答案放在<answer>选项</answer>内。"
```

**Translation**: "You are an AI assistant specializing in ultrasound medicine. You have received an ultrasound multiple-choice question. Please give your thinking process and place it inside `<think>thinking process</think>` tags. Output only one option and place the answer inside `<answer>option</answer>` tags."

This is the **only prompt in the entire repo** that requests CoT reasoning.

---

### 3F. `<think>` / `<answer>` Tag Extraction (Evaluation)

**File**: `eval/eval.py:42-50`

```python
def extract_think_content(text: str):
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def extract_answer_content(text: str):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches
```

Expected model output format:
```
<think>
  [step-by-step clinical reasoning]
</think>
<answer>A</answer>
```

---

### 3G. Final Training Data Schema (ShareGPT)

```json
{
    "conversations": [
        {"from": "system",  "value": "是一位医学超声助手，请回答相关问题。"},
        {"from": "human",   "value": "这是一张什么类型的医学超声图像？\n<image>"},
        {"from": "gpt",     "value": "这是一张肾脏的超声图像，重点关注肾脏的大小方面。"},
        {"from": "human",   "value": "图中显示了什么异常情况？"},
        {"from": "gpt",     "value": "图中A部分展示了肾脏增大的情况..."}
    ],
    "images": ["./demo_images/demo_1241.jpg"]
}
```

---

## 4. Data Cleaning & Quality Assurance

### 4A. Markdown Fence Stripping (`parse_json`)

Present identically in 3 files: `data/VQA.py:23-31`, `data/QwenVLbbox.py:19-28`, `eval/eval.py:31-39`

```python
def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output
```

Used everywhere as `json.loads(parse_json(raw_llm_output))`.

---

### 4B. ShareGPT Format Validation & Cleaning

**File**: `data/VQA.py:131-155`

```python
def check_and_clean_sharegpt(json_list):
    """
    检查 JSON 列表中的每个元素是否符合 ShareGPT 格式，
    若不符合则删除该元素，最后返回清理后的列表
    """
    cleaned_list = []
    for item in json_list:
        if isinstance(item, dict) and 'conversations' in item:
            conversations = item['conversations']
            valid = True
            if conversations and conversations[1].get('from') != 'human':
                valid = False
            for i in range(2, len(conversations)):
                prev_from = conversations[i - 1].get('from')
                current_from = conversations[i].get('from')
                if (prev_from == 'human' and current_from != 'gpt') or \
                   (prev_from == 'gpt' and current_from != 'human'):
                    valid = False
                    break
            if valid:
                cleaned_list.append(item)
    return cleaned_list
```

Called at `VQA.py:252`: `merge_list = check_and_clean_sharegpt(merge_list)`

Validates:
- Item is a dict with `conversations` key
- Conversation starts with a `human` turn (index 1, after system)
- Strict `human`/`gpt` alternation throughout

---

### 4C. Simpler Turn-Order Check

**File**: `data/VQA.py:89-98`

```python
def check(conversation):
    for index in range(len(conversation)):
        if index % 2 == 0 and conversation[index]["from"] != "human":
            return True    # malformed
        if index % 2 == 1 and conversation[index]["from"] != "gpt":
            return True    # malformed
    return False  # valid
```

---

### 4D. Length Consistency Check (GPT output vs. bounding boxes)

**File**: `data/VQA.py:192-193`

```python
if len(gpt_json) != len(bbox_json):
    return []   # Discard entire page if counts don't match
```

---

### 4E. Bounding Box Coordinate Validation

**File**: `data/QwenVLbbox.py:224-238`

```python
for i, item in enumerate(bbox_list):
    bbox = item['bbox_2d']
    if len(bbox) != 4:
        print(f"第 {i + 1} 个 bbox 格式不正确，应为 (left, upper, right, lower)，跳过该 bbox。")
        continue
    left, upper, right, lower = bbox
    if left < 0 or upper < 0 or right > image.width or lower > image.height or left >= right or upper >= lower:
        print(f"第 {i + 1} 个 bbox 坐标不合法，跳过该 bbox。")
        continue
```

---

### 4F. Truncated JSON Recovery (Fallback Parser)

**File**: `data/QwenVLbbox.py:99-104`

```python
try:
    json_output = ast.literal_eval(bounding_boxes)
except Exception as e:
    end_idx = bounding_boxes.rfind('"}') + len('"}')
    truncated_text = bounding_boxes[:end_idx] + "]"
    json_output = ast.literal_eval(truncated_text)
```

Recovers from LLM-truncated JSON arrays by finding the last complete object and closing the array.

---

### 4G. XML Point Decoding

**File**: `data/QwenVLbbox.py:31-49`

```python
def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {"points": points, "alt": alt, "phrase": phrase}
    except Exception as e:
        print(e)
        return None
```

---

### 4H. Error Handling on API Calls

**VQA.py:107-114** (GPT-4o):
```python
try:
    completion = get_instruction(image_path=image_path, prompt=prompt, **kwargs)
    content = completion.choices[0].message.content
    data = json.loads(parse_json(content))
except:
    print(f'page_{page} gpt request error.')
    return []
```

**QwenVLbbox.py:268-272** (Qwen):
```python
try:
    bbox_list = json.loads(parse_json(response))
except:
    print(f"{page} json error")
    return
```

---

### 4I. `data/filter.ipynb` — Full ShareGPT Validator

```python
def validate_conversation(conversation: List[Dict]) -> bool:
    """验证对话格式"""
    has_human = False
    has_gpt = False
    for msg in conversation:
        if not isinstance(msg, dict):
            return False
        if "from" not in msg or "value" not in msg:
            return False
        role = msg["from"]
        value = msg["value"]
        if role not in {"system", "human", "gpt"}:
            return False
        if not isinstance(value, str) or len(value.strip()) == 0:
            return False
        if role == "human":
            has_human = True
        elif role == "gpt":
            has_gpt = True
    return has_human and (has_gpt or has_human)

def validate_sharegpt_item(item: Dict) -> bool:
    if not isinstance(item, dict):
        return False
    required_fields = {"conversations", "images"}
    if not required_fields.issubset(item.keys()):
        return False
    conversations = item["conversations"]
    if not isinstance(conversations, list) or len(conversations) < 1:
        return False
    return validate_conversation(conversations)

def merge_sharegpt_data(input_dir: str, output_file: str) -> None:
    merged_data = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith('.json'):
            continue
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if validate_sharegpt_item(item):
                    merged_data.append(item)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2, sort_keys=True)
```

---

### 4J. Summary: No Semantic QA Exists

| Mechanism | Present? |
|---|---|
| Markdown fence stripping | Yes |
| JSON parse success/failure | Yes |
| ShareGPT turn alternation validation | Yes |
| 1:1 count matching (bboxes vs conversations) | Yes |
| Bbox coordinate bounds checking | Yes |
| Truncated JSON recovery | Yes |
| Hallucination detection | **No** |
| Confidence scoring | **No** |
| JSON schema validation (jsonschema lib) | **No** |
| Semantic content validation | **No** |

All QA is purely **structural/format-based**, not semantic.

---

## 5. Organ-Specific Routing

### Finding: No organ-specific routing exists in the codebase.

After exhaustive search across all Python files, JSON data, and configuration:

- **No conditional branches** (if/elif/else, match/case) dispatching based on organ type
- **No organ-specific prompt templates** — one single generic prompt for all images
- **No organ enums, mappings, or label taxonomies**
- **No routing/dispatch logic** tied to anatomy
- **No fields** for organ type, modality, category, or region in the data schema

The system prompt is completely organ-agnostic: `"你是一名超声医学人工智能助手"` ("You are an ultrasound medical AI assistant"). The pipeline processes any ultrasound image through the same generic prompt regardless of organ.

The "multi-organ" nature of the dataset comes from the **diversity of the source textbook**, not from any code-level differentiation. Demo data covers kidney (肾脏) ultrasound purely because that's the content of the demo pages.

---

## 6. CoT Generation: The Real Mechanism

### Critical Discovery: ReMUD Does NOT Pre-Generate CoT Datasets

The VQA.py pipeline generates **plain Q&A pairs only**. The `<think>` / `<answer>` reasoning structure is:

1. **Learned** by the model through GRPO (Group Relative Policy Optimization) reinforcement learning via LLaMA-Factory
2. **Elicited** at inference time using a system prompt that requests the tags
3. **Extended** via Budget Forcing when the reasoning is too short

The model used for CoT evaluation:
```python
model_name = "../../LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/freeze/train_2025-04-14"
```

This is a **Qwen2.5-VL-7B fine-tuned via LLaMA-Factory** freeze-tuning, which has already learned `<think>`/`<answer>` behavior from GRPO training.

---

## 7. Budget Forcing: Deep Dive

**File**: `eval/generate.ipynb`, cell `b9f2f3dc`

### Configuration

```python
pass_1 = 4                    # Pass@4: generate 4 completions per question
temperature = 0.6
top_p = 0.7
budget_len = 96               # minimum thinking length (characters)
max_new_tokens_tmp = 1024
wait = '等等，'                # "Wait," — injected to force continued reasoning
num_ignore = 2                # max budget forcing retry attempts
```

### Algorithm

```python
for step in range(pass_1):  # 4 attempts per question

    # Step 1: Initial generation
    output_text, text = inference(messages=messages, device=device,
                                  temperature=0.6, top_p=0.7,
                                  max_new_tokens=max_new_tokens_tmp)
    budget_forcing_text = text
    think_trace, answer = split_thinking_answer(output_text)

    # Step 2: Budget Forcing — if thinking is too short, force more
    num_tmp = 0
    while len(think_trace) < budget_len:      # < 96 characters
        num_tmp += 1
        # Inject "Wait," into the thinking to make the model continue
        tmp_text = budget_forcing_text + '<think>' + think_trace + wait
        output_text, _ = inference(messages=messages, text=tmp_text,
                                   device=device, temperature=0.6,
                                   top_p=0.7, max_new_tokens=max_new_tokens_tmp)
        output_text = '<think>' + think_trace + wait + output_text
        think_trace, answer = split_thinking_answer(output_text)
        if num_tmp > num_ignore:    # max 2 retries
            break

    # Step 3: Force close the thinking tag to get final answer
    tmp_text = budget_forcing_text + '<think>' + think_trace + '</'
    output_text, _ = inference(messages=messages, text=tmp_text,
                               device=device, temperature=0.6,
                               top_p=0.7, max_new_tokens=4096)  # 4x more tokens for answer
    output_text = '<think>' + think_trace + '</' + output_text
    think_trace, answer = split_thinking_answer(output_text)
    pass_nlist.append(output_text)
```

### How It Works Step-by-Step

| Step | Action | Text Injected |
|------|--------|---------------|
| 1 | Normal inference | Nothing — model generates `<think>...</think><answer>...</answer>` |
| 2 | If `len(think_trace) < 96`, prefix-inject previous thinking + "Wait," | `<think>{prev_thinking}等等，` |
| 3 | Repeat step 2 up to 2 more times if still too short | Same pattern |
| 4 | Force-close the think tag so model outputs answer | `<think>{final_thinking}</` |

### Key Mechanism: Prefix Injection

The `text=` parameter in `inference()` **bypasses the chat template** and feeds raw prefix text directly:

```python
def inference(messages, device, text=None, temperature=0.6, top_p=0.95, max_new_tokens=512):
    if text is None:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # When text IS provided, it's used AS the raw prompt (prefix injection)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, ...).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_p=top_p)
```

---

## 8. Accept/Reject & Quality Filtering: Gap Analysis

### Finding: No Accept/Reject Mechanism Exists

After tracing every line of the generation loop, there is **no filtering, rejection sampling, or cross-checking of CoT quality**.

### What the Pass@K Loop Actually Does

```python
pass_nlist = []               # stores ALL completions

for step in range(pass_1):    # 4 iterations
    # ... budget forcing + inference ...

    pass_nlist.append(output_text)          # APPEND unconditionally — no filter

    if answer == label_answer:
        score = score + 1                   # just a counter, not a gate

score = score / pass_1                      # fraction correct (e.g., 0.75)
```

### What Gets Written (No Filtering)

```python
tmp["prompt"] = prompt
tmp["predict"] = output_text                # the LAST completion only (not the best)
tmp["label"] = mcq["conversations"][2]["value"]
tmp["score"] = score                        # a metric, not used to filter
tmp["pass_nlist"] = pass_nlist              # all 4 stored, none rejected

with open(output_path, "a") as f:
    f.write(json.dumps(tmp, ensure_ascii=False) + '\n')
```

### Complete Audit of Missing Quality Gates

| Mechanism | Present? | Evidence |
|---|---|---|
| Rejection sampling (keep only correct CoTs) | **No** | All 4 completions appended unconditionally |
| Majority voting (pick most common answer) | **No** | `predict` is just the last sample |
| Best-of-N selection | **No** | No comparison between completions |
| Confidence thresholding | **No** | No probability/logit inspection |
| Hallucination detection | **No** | No cross-checking against ground truth text |
| Consistency filtering | **No** | No comparison between reasoning traces |
| `<think>` quality validation | **No** | Only `len(think_trace) < 96` length check |
| Discarding bad samples | **No** | Every question written regardless of `score` |

### The Only Two Quality Controls That Exist

1. **Budget Forcing** — ensures thinking isn't too short:
   ```python
   while len(think_trace) < budget_len:    # budget_len = 96 chars
   ```

2. **Answer correctness scoring** — but used purely for metrics, not filtering:
   ```python
   score = score / pass_1   # e.g., 3/4 = 0.75
   # Nothing is rejected based on this score
   ```

### Infrastructure Available But Unused

The saved data structure per question supports rejection sampling — it just isn't implemented:

```json
{
    "prompt": "肝脏在这次超声检查中...",
    "predict": "<think>...</think><answer>C</answer>",
    "label": "<answer>C</answer>",
    "score": 0.75,
    "pass_nlist": [
        "<think>reasoning1...</think><answer>C</answer>",
        "<think>reasoning2...</think><answer>B</answer>",
        "<think>reasoning3...</think><answer>C</answer>",
        "<think>reasoning4...</think><answer>C</answer>"
    ]
}
```

`pass_nlist` (all K generations), `label` (ground truth), and `score` (accuracy) are all present — filtering could be trivially added.

---

## 9. Actionable Recommendations for Building a Teacher Agent

### 9.1 What to Reuse from ReMUD

- **Budget Forcing technique** — effective for guaranteeing minimum reasoning depth
- **Prefix injection pattern** — bypass chat template to extend partial generations
- **`<think>`/`<answer>` tag structure** — simple, parseable CoT format
- **System prompt pattern** — explicit instructions for tag usage work well
- **Pass@K infrastructure** — generate multiple completions per question

### 9.2 What You Must Build Yourself

Since ReMUD has no accept/reject mechanism, you need to add your own quality pipeline:

```
For each sample:
  1. Generate K completions with budget forcing (as ReMUD does)
  2. REJECT if: answer != ground_truth_label          <- rejection sampling
  3. REJECT if: len(think_trace) < min_length         <- already in ReMUD
  4. REJECT if: think_trace doesn't reference image   <- anti-hallucination
  5. RANK remaining by: consistency across K samples   <- majority voting
  6. SELECT: the completion whose answer matches
     majority AND has longest/best reasoning           <- best-of-N
```

### 9.3 Additional QA Layers to Consider

- **Semantic validation**: Check that the reasoning mentions the correct organ/anatomy
- **Self-consistency**: Generate N reasoning chains; only keep samples where >50% agree on the answer
- **Teacher verification**: Use a stronger model (GPT-4o) to score/verify the reasoning chain
- **Negative filtering**: Reject if reasoning contains known hallucination patterns (e.g., referencing image features not present)
- **Length-quality correlation**: Track and enforce both minimum AND maximum reasoning lengths

### 9.4 Organ-Specific Routing (Not in ReMUD — Build From Scratch)

If you need organ-specific handling:

```python
# Example routing architecture (not from ReMUD)
ORGAN_PROMPTS = {
    "liver": "Focus on hepatic echogenicity, portal vein...",
    "breast": "Evaluate BI-RADS features, margins, shape...",
    "thyroid": "Assess TI-RADS features, calcifications..."
}

def get_organ_prompt(organ_type: str) -> str:
    return ORGAN_PROMPTS.get(organ_type, DEFAULT_PROMPT)
```

### 9.5 Key Findings Summary

| Component | ReMUD Status | Your Action |
|---|---|---|
| SFT VQA data generation | Complete (GPT-4o) | Reusable |
| CoT prompt template | Minimal (1 system prompt) | Extend with domain-specific instructions |
| Budget Forcing | Implemented | Reuse directly |
| Pass@K generation | Implemented | Reuse, add filtering on top |
| Rejection sampling | **Not implemented** | Build yourself |
| Hallucination detection | **Not implemented** | Build yourself |
| Consistency filtering | **Not implemented** | Build yourself |
| Organ-specific routing | **Not implemented** | Build yourself |
| Confidence scoring | **Not implemented** | Build yourself |

---

*End of analysis.*
