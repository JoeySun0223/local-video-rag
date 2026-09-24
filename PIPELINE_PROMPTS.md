# 视频处理全流程 Prompt 说明

本文档记录当前代码从“生成原始 ASR”到“生成语义片段”过程中，实际提供给模型的全部 Prompt、动态输入格式、输出契约、长度限制和失败重试提示。

> 当前默认配置：本地 `Qwen3-ASR-1.7B-HF` 生成原始 ASR；云端 `GLM-5.3` 清洗句子、一次性规划语义边界并生成标题、摘要和关键词。

## 1. 总体调用顺序

```text
视频/音频
  │
  ├─ 1. 本地 Qwen3-ASR 转写音频片段
  │     Prompt：标题 + 分类 + 热词（如有）
  │
  ├─ 2. 本地 Forced Aligner 生成词级时间戳
  │     不使用自然语言 Prompt
  │
  ├─ 3. 程序合并转写并生成原始句子
  │     不调用大模型
  │
  ├─ 4. GLM-5.3 逐批清洗句子
  │     Prompt：固定清洗规则 + 目标句 + 前后各两句上下文
  │
  ├─ 5. GLM-5.3 一次性规划语义片段边界
  │     Prompt：完整清洗后 ASR + 语义划分规则
  │
  └─ 6. GLM-5.3 为固定片段生成标签
        Prompt：固定片段正文 + 标签规则
```

---

## 2. 原始 ASR：本地 Qwen3-ASR

### 2.1 模型与调用方式

- 模型：`models/qwen3-asr-1.7b-hf`
- 调用：`processor.apply_transcription_request(audio=..., prompt=prompt)`
- 每个音频片段最大生成长度：`2048` tokens
- 采样：关闭，`do_sample=False`
- 音频片段目标长度：约 `180` 秒
- 切分方式：在切点前后约 `5` 秒范围寻找低能量位置

这里不是 GLM/OpenAI 风格的 `system`、`user` 两条消息。代码只向 Qwen3-ASR 的官方 processor 提供一个可选 transcription prompt。

### 2.2 实际 Prompt 拼接规则

Prompt 由以下三个可选部分组成：

```text
Title: {视频标题}
Category: {视频分类}
Vocabulary: {热词1}, {热词2}, {热词3}
```

存在多个部分时使用英文句点和空格连接：

```text
Title: {视频标题}. Category: {视频分类}. Vocabulary: {热词1}, {热词2}
```

具体规则：

1. 有标题时加入 `Title: ...`。
2. 分类非空且不等于“未分类”时加入 `Category: ...`。
3. 存在命令行热词或词表热词时加入 `Vocabulary: ...`。
4. 三者都不存在时，Prompt 为 `None`。

示例：

```text
Title: 执行案件办理系统操作培训. Category: 法院业务培训. Vocabulary: 管案, 执行立案, 结案审批
```

### 2.3 热词来源

热词按照以下顺序合并并去重：

1. 命令行 `--hotword`；
2. `resources/glossary.json` 的 `global` 条目；
3. 当前视频 ID 对应的 `sources` 条目。

### 2.4 ASR 后续无 Prompt 步骤

Qwen3-ASR 输出后，以下操作不调用语言模型：

- Forced Aligner 对转写文本和音频做强制对齐；
- 各音频片段的文字直接连接；
- 时间戳加上各片段的时间偏移；
- 对超过 `240` 个词法字符的句子补充分句边界；
- 根据全局文本和时间戳生成原始句子数组。

相关源码：

- `video_pipeline/asr/engine.py`：`_prompt()`、`_run()`
- `video_pipeline/asr/cli.py`：标题、分类和热词输入
- `video_pipeline/config.json`：ASR 模型和长度配置

---

## 3. 清洗原始 ASR：GLM-5.3

### 3.1 System Prompt（原文）

```text
你是资深中文编辑，负责把法院业务培训视频的逐句 ASR 整理成准确、简洁、自然的书面语。

1. 删除迟疑音、口癖、口吃、紧邻重复、自我纠正前的错误表达，以及不影响理解系统功能和必要操作的授课回顾、转场、总结和演示旁白。讲解行为本身、对前后内容的评价、演示者无业务约束的临时选择不属于有效信息，必须删除；“这/以上就是……的介绍、说明、展示”等只复述前文且没有新增操作、条件或结果的总结句，即使含有业务术语也应整句删除。无明确指代的“我们、这个、这边”和不表示真实顺序的“然后”也应删除。输出应是可直接进入知识库的书面语；可在同一 sentence_id 内重排语序、整理逻辑或写成多个书面句。
2. 不得改变、遗漏或虚构有效信息。主体、动作、对象、按钮和路径、条件、分支、否定、数字、角色、步骤、结果及“一、二、三”等结构编号均须保留；长句中的多个阶段和动作必须全部保留。上下文只用于消歧，不得跨 sentence_id 搬移、合并或去重。
3. 结合视频标题、上下文、法院业务流程和固定搭配主动纠正错字、同音字、近音字及病句，不能因原词在普通中文中成立就保留。confirmed_business_terms 是已确认字形的术语证据：语境匹配时采用，不匹配时不得强行替换。只有存在多个合理候选，或必须查看画面、回听音频才能确定时才需审核。
4. 审核只取决于确定性和语义风险，不取决于修改幅度。能够可靠处理时 needs_review=false；术语、数字、界面字段、条件、否定、步骤顺序或句意仍有歧义时 needs_review=true 并说明原因。无法可靠修正时 text 保持原文；有较可信的建议时可写入 text 等待审核。

只输出 JSON：
{"results":[{"sentence_id":1,"text":"处理后的完整句子","needs_review":false,"review_reason":""}]}
每个目标句必须按输入顺序返回且只返回一次。每项只能包含 sentence_id、text、needs_review、review_reason。needs_review 必须是布尔值。needs_review 为 false 时 review_reason 必须为空字符串；needs_review 为 true 时 review_reason 必须简要说明需要审核或不能确定的原因。
```

### 3.2 User Prompt（动态 JSON）

User Prompt 没有额外的自然语言前缀，直接发送 JSON：

```json
{
  "video_title": "当前视频标题",
  "confirmed_business_terms": [
    "经过确认的当前视频业务术语"
  ],
  "context_before": [
    "目标批次前第2句",
    "目标批次前第1句"
  ],
  "target_sentences": [
    {
      "sentence_id": 21,
      "text": "需要清洗的原始 ASR 句子"
    },
    {
      "sentence_id": 22,
      "text": "需要清洗的下一句话"
    }
  ],
  "context_after": [
    "目标批次后第1句",
    "目标批次后第2句"
  ]
}
```

字段用途：

- `video_title`：辅助理解视频领域，但 Prompt 禁止从标题补写句子内容。
- `confirmed_business_terms`：来自词表的正确术语字形，仅为模型提供判断证据，不是字符串替换表。
- `context_before`：最多提供目标批次前两句，仅作判断依据。
- `target_sentences`：模型必须逐句清洗并逐条返回的内容。
- `context_after`：最多提供目标批次后两句，仅作判断依据。

### 3.3 期望输出

```json
{
  "results": [
    {
      "sentence_id": 21,
      "text": "处理后的完整句子",
      "needs_review": false,
      "review_reason": ""
    },
    {
      "sentence_id": 22,
      "text": "保持原文或给出待审核建议",
      "needs_review": true,
      "review_reason": "疑似界面字段识别错误，需要核对画面"
    }
  ]
}
```

### 3.4 分批与输出限制

- 每批最多 `40` 个目标句；
- 每批目标句正文合计最多 `3800` 个字符；
- 单个句子即使超过字符上限，也会单独成为一批；
- 前后文不计入上述目标句字符累计；
- 每次最大输出 `4096` tokens；
- 默认最多尝试 `3` 次；
- 当前 GLM-5.3 配置启用 thinking，并使用 `high` 推理强度；GLM-5.3 不支持直接关闭 thinking；
- 当前有效请求温度为 `0`；
- 要求返回 JSON object。

### 3.5 字段校验失败后的追加 Prompt

如果返回字段、句子 ID、顺序或覆盖范围不合法，下一次请求会在原 User Prompt 后追加：

```text


上一次字段校验失败：{具体错误}。请重新返回完整结果。
```

典型错误包括：

- 顶层缺少 `results`；
- 每项字段不是且不只包含 `sentence_id`、`text`、`needs_review`、`review_reason`；
- `sentence_id` 重复；
- 句子漏返、多返或顺序错误；
- `text` 或 `review_reason` 不是字符串；
- `needs_review` 不是布尔值；
- `needs_review` 与 `review_reason` 的空值规则不一致。

### 3.6 模型结果的程序处理

- `needs_review=false` 且候选文本与原文相同：`unchanged`；
- `needs_review=false` 且候选文本发生变化：直接接受，`auto_accepted`；
- `needs_review=true`：无论文本是否变化都记为 `pending_review`，最终文本暂时保持原文；
- 存在 `pending_review` 时，该视频暂停生成 `cleaned_asr`，等待人工批准或拒绝。

相关源码：

- `video_pipeline/cleanup/prompts.py`：System Prompt
- `video_pipeline/cleanup/service.py`：User Prompt、批处理与重试
- `video_pipeline/cleanup/core.py`：输出校验和自动接受规则
- `video_pipeline/cleanup/cli.py`：默认批次和 token 参数

---

## 4. 一次性生成语义片段边界：GLM-5.3

### 4.1 System Prompt（原文）

```text
你是视频章节编辑。通读全部清洗后句子，按独立主题、完整问答或完整操作划分连续章节。语义完整优先，字数只用于防止章节过长或过碎。

- 全文不超过1300字，保持一个章节；不要因其中包含多个相关子功能而拆分。
- 1301至2200字通常划分为2章；只有确实存在3个独立内容单元且每章均不少于500字时才分3章，不得切成4章以上。
- 更长文本的单章以700至1100字为宜；超过1200字时，应优先在附近的自然语义边界拆分。
- 不要为了凑字数切断完整问题、完整回答、操作步骤、校验修改或连续业务场景。
- 原则上不得形成不足500字的章节；应移动到附近更合适的自然边界或并入相邻章节。引入、过渡语和结束语不得单独成章。
- 一个功能或操作从开始到完成应保持在同一章；只有转入另一个功能、问题或业务主题时才开始新章。

输出前根据 start_char、end_char 自检各章字数；若不符合上述区间，先移动边界或合并，再输出最终结果。

你只返回每个章节的起始句 ID。程序会自动把每段延伸到下一段起始句的前一句，最后一段自动覆盖到全文末句。第一个起始 ID 必须是输入第一条 sentence_id，后续 ID 严格递增。

输入包含 total_chars；sentences 中每项为 [sentence_id, start_char, end_char, text]。start_char 和 end_char 是去除空白和标点后的累计字符位置，不需要自行数数。
只输出 JSON，不要解释：
{"start_sentence_ids":[1,18]}
```

### 4.2 User Prompt（动态文本 + JSON）

```text
请划分以下视频的语义片段：
{"video_title":"当前视频标题","sentence_count":3,"total_chars":30,"sentences":[[1,0,10,"第一句清洗后的文本"],[2,10,20,"第二句清洗后的文本"],[3,20,30,"第三句清洗后的文本"]]}
```

正式输入使用紧凑 JSON，不添加缩进和多余空格。每个句子表示为：

```text
[sentence_id, start_char, end_char, text]
```

### 4.3 期望输出

```json
{"start_sentence_ids":[1,18]}
```

模型不返回结束句。程序按照下一个片段的起始句推导结束句：

```text
片段 1：第一个 start_sentence_id → 第二个 start_sentence_id 的前一句
片段 2：第二个 start_sentence_id → 第三个 start_sentence_id 的前一句
最后片段：最后一个 start_sentence_id → 全文最后一句
```

### 4.4 输入、输出和校验限制

- 一个视频的完整清洗后 ASR 一次提交；
- 紧凑 User Prompt 最大允许 `120000` 个字符，超出直接报错，不自动切批；
- 每次最大输出 `8192` tokens；
- 默认最多尝试 `3` 次；
- 默认关闭 thinking；
- 第一个起始句必须等于输入的第一条 `sentence_id`；
- 所有起始 ID 必须在输入中存在；
- 起始 ID 必须严格递增，不能重复；
- 必须返回 `start_sentence_ids` 数组；
- 模型误加的其他顶层字段会被程序忽略；
- 最终片段对象及其严格 JSON 结构由程序组装。

### 4.5 输出校验失败后的追加 Prompt

```text


上一次输出校验失败，请重新输出完整结果并修正此问题：{具体错误}
```

相关源码：

- `video_pipeline/segments/core.py`：`SEGMENT_BOUNDARY_SYSTEM_PROMPT`、`segment_prompt_for()`
- `video_pipeline/segments/cli.py`：完整输入字符限制和默认 token 参数

---

## 5. 为固定语义片段生成标签：GLM-5.3

一次性边界规划完成后，程序固定所有片段的起止句，再调用 GLM 生成标题、摘要和关键词。模型在此阶段不得改变边界。

### 5.1 System Prompt（原文）

```text
你是视频知识库的语义片段标签编辑。片段边界已经固定，不得修改、合并或拆分。

请为输入的每个片段生成：
- 具体、可检索的中文标题，准确表达该片段的问题、操作或主题，避免“功能介绍”“第一部分”等空泛表述；
- 忠于原文的一至两句摘要，不添加原文没有的事实；
- 按内容需要生成关键词，优先覆盖业务术语、模块名、操作对象、关键动作和核心概念，不为凑数添加泛词。

必须按照输入 position 逐一返回，不得遗漏、重复或增加片段。标题不超过40个字符，摘要不超过160个字符，每个关键词不超过24个字符。
只输出一个 JSON 对象，不要输出 Markdown 或解释：
{"labels":[{"position":1,"title":"具体标题","summary":"忠实摘要。","keywords":["关键词"]}]}
每个标签只能包含 position、title、summary、keywords。
```

### 5.2 User Prompt（动态文本 + JSON）

```text
请为以下固定片段生成标签：
{"video_title":"当前视频标题","fixed_segments":[{"position":1,"start_sentence_id":1,"end_sentence_id":17,"text":"该片段的全部清洗后正文\n第二句正文"},{"position":2,"start_sentence_id":18,"end_sentence_id":42,"text":"下一片段的全部清洗后正文"}]}
```

### 5.3 期望输出

```json
{
  "labels": [
    {
      "position": 1,
      "title": "执行案件概览与办理入口",
      "summary": "介绍执行案件的业务范围，并说明进入办理模块的方式。",
      "keywords": ["执行案件", "办理入口"]
    },
    {
      "position": 2,
      "title": "执行立案操作流程",
      "summary": "演示执行立案所需的信息填写、校验和提交操作。",
      "keywords": ["执行立案", "信息填写", "提交"]
    }
  ]
}
```

### 5.4 标签批次与校验限制

- 每批最多 `6` 个片段；
- 每批片段正文合计目标上限 `12000` 字符；
- 单个片段超过 `12000` 字符时仍会单独提交；
- 每次最大输出 `8192` tokens；
- 标题非空且最多 `40` 个字符；
- 摘要非空且最多 `160` 个字符；
- 关键词按内容需要生成，不设固定数量上限；
- 每个关键词非空且最多 `24` 个字符；
- 重复关键词由程序去重；
- 必须覆盖本批所有 `position`，不能遗漏、重复或增加；
- 顶层只能包含 `labels`；
- 每项只能包含 `position`、`title`、`summary`、`keywords`。

输出校验失败时，使用与边界规划相同的追加 Prompt：

```text


上一次输出校验失败，请重新输出完整结果并修正此问题：{具体错误}
```

相关源码：

- `video_pipeline/segments/core.py`：`SEGMENT_LABEL_SYSTEM_PROMPT`、`label_prompt_for()`、`make_label_batches()`

---

## 6. GLM API 消息封装

清洗、一次性边界规划和标签生成最终都使用以下消息结构：

```json
{
  "model": "glm-5.3",
  "messages": [
    {
      "role": "system",
      "content": "对应阶段的固定 System Prompt"
    },
    {
      "role": "user",
      "content": "对应阶段的动态 User Prompt，以及失败时可能追加的纠错要求"
    }
  ],
  "stream": false,
  "do_sample": false,
  "temperature": 0,
  "thinking": {
    "type": "enabled"
  },
  "reasoning_effort": "high",
  "max_tokens": 4096,
  "response_format": {
    "type": "json_object"
  },
  "request_id": "每次请求生成的新 UUID"
}
```

其中 `max_tokens` 按阶段变化：

| 阶段 | 默认 `max_tokens` |
|---|---:|
| ASR 清洗 | 4096 |
| 一次性语义边界 | 8192 |
| 固定片段标签 | 8192 |
| Web 人工变化片段重新生成标签 | 4096 |

> `max_tokens` 是单次回答的最大生成长度，不是输入上下文长度，也不是整个视频所有调用的总输出长度。

相关源码：

- `video_pipeline/shared/glm.py`：统一 API 请求结构
- `video_pipeline/config.json`：API 地址、模型、thinking 和超时设置

---

## 7. 从模型输出到最终语义片段文件

最终每个语义片段由程序合成，模型不直接输出完整最终文件：

```json
{
  "segment_no": 1,
  "start_sentence_id": 1,
  "end_sentence_id": 17,
  "start_ms": 0,
  "end_ms": 95000,
  "title": "执行案件概览与办理入口",
  "summary": "介绍执行案件的业务范围和办理入口。",
  "keywords": ["执行案件", "办理入口"],
  "content": "片段内所有清洗后句子按原顺序连接"
}
```

字段来源：

| 字段 | 来源 |
|---|---|
| `start_sentence_id` | GLM 边界规划输出 |
| `end_sentence_id` | 程序根据下一片段起点推导 |
| `start_ms`、`end_ms` | 清洗句子继承的 ASR 时间戳 |
| `title`、`summary`、`keywords` | GLM 标签输出 |
| `content` | 程序按固定边界连接清洗后句子 |

这保证模型只负责语义判断和标签撰写，句子覆盖范围、正文和时间戳均由程序确定及校验。程序还会将同一份结果渲染为 `data/semantic_markdown/<分区>/<video_id>.md`；保存清洗句子或片段修改时，Markdown 与 JSON 同步更新，不再调用模型。

---

## 8. 当前 Prompt 设计的关键特征

1. **原始 ASR 使用局部音频 Prompt**：标题、分类和热词只辅助识别。
2. **清洗逐句可审计**：每个目标句必须按 ID 原位返回。
3. **按确定性分流**：删除、替换或较大书面化改写只要有明确依据且信息完整即可自动接受；存在术语、数字、条件、否定、步骤顺序或业务含义歧义时才进入人工审核。
4. **边界与标签分两次生成**：先固定覆盖范围，再写标题和摘要，避免标签生成改变边界。
5. **字数只提供尺度**：模型在一次性边界规划中综合有效字数和语义完整性，不按固定字数硬切。
6. **模型只返回起始点**：结束点、正文和播放时间均由程序计算，降低漏句和重叠风险。
