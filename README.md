# 🧠 LLM & Prompt Engineering — The Complete Practical Guide

> **From zero to prompt engineer** — understand how Large Language Models *actually* work, why they fail, and how to talk to them like a pro.

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequests.com)
![Last Updated](https://img.shields.io/badge/last%20updated-2026-blue)
![Beginner Friendly](https://img.shields.io/badge/beginner-friendly-success)

---

## 📖 Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [🔤 Tokens](#-tokens--the-atoms-of-language) | How LLMs read and count text |
| 2 | [🌐 Embeddings](#-embeddings--meaning-as-math) | How meaning becomes numbers |
| 3 | [⚙️ Transformers](#️-transformers--the-engine-inside) | The architecture powering everything |
| 4 | [📐 Context Window](#-context-window--working-memory) | Why LLMs "forget" long conversations |
| 5 | [🔮 Inference](#-inference--how-llms-generate-text) | How text is actually generated |
| 6 | [🎯 Fine-tuning vs RAG](#-fine-tuning-vs-rag) | When to teach vs when to tell |
| 7 | [👻 Hallucinations & Limits](#-hallucinations--limitations) | Why LLMs lie (and how to catch it) |
| 8 | [✍️ Prompt Engineering](#️-prompt-engineering-techniques) | Zero-shot, few-shot, CoT, and more |
| 9 | [🛡️ Best Practices](#️-best-practices--optimization) | Make your prompts bulletproof |
| 10 | [❌ Common Mistakes](#-common-mistakes) | What to avoid |
| 11 | [🗺️ Quick Reference](#️-quick-reference-cheatsheet) | Everything on one page |

---

## 🔤 Tokens — The Atoms of Language

> **TL;DR:** LLMs don't read words — they read *chunks of text* called tokens.

### What is a Token?

A **token** is the smallest unit an LLM processes. It's not always a full word — it can be a word, part of a word, punctuation, or even a space.

```
"Hello, world!" → ["Hello", ",", " world", "!"]   → 4 tokens
"Unbelievable"  → ["Un", "believ", "able"]         → 3 tokens
"GPT"           → ["GPT"]                          → 1 token
"12345"         → ["123", "45"]                    → 2 tokens (varies)
```

### Why This Matters

| Concept | Impact |
|---------|--------|
| **Cost** | APIs charge *per token* — longer prompts = higher cost |
| **Limits** | Every model has a max token count (input + output) |
| **Rare words** | Unusual words split into more tokens, costing more |
| **Numbers** | Long numbers are often split — causes math errors |

### 💡 Rule of Thumb

```
~1 token  ≈  ¾ of a word  ≈  4 characters (English)
1,000 tokens ≈ 750 words ≈ 3 pages of text
```

> 🌍 **Non-English languages** (Arabic, Chinese, Japanese) often use *more* tokens per word — keep this in mind for multilingual apps.

---

## 🌐 Embeddings — Meaning as Math

> **TL;DR:** Words with similar meanings end up near each other in mathematical space.

### The Core Idea

LLMs convert every token into a list of numbers called a **vector** or **embedding**. These numbers capture *semantic meaning*.

```
"King"   → [0.92, 0.11, 0.45, ...]
"Queen"  → [0.89, 0.14, 0.47, ...]   ← very close to King
"Cat"    → [0.03, 0.78, 0.12, ...]   ← very different
```

### The Famous Analogy

```
King − Man + Woman ≈ Queen
Paris − France + Italy ≈ Rome
```

The model has learned that relationships between concepts are consistent *directions* in embedding space.

### Real-World Use Cases

```
📧 Semantic Search     → find documents by meaning, not keywords
🔁 Recommendation      → suggest similar items
🗂️ Clustering          → group similar texts automatically
🔍 Anomaly Detection   → find unusual patterns in text
```

---

## ⚙️ Transformers — The Engine Inside

> **TL;DR:** Transformers let LLMs understand which words are important *relative to each other* — simultaneously, across the whole input.

### The Core Mechanism: Attention

Before Transformers (2017), models read text left-to-right, word by word. Transformers changed everything with **Self-Attention**: every token can "look at" every other token at once.

```
"The bank by the river flooded."
        ↑
   Which "bank"? Financial? Or riverbank?
   
Attention scores let the model connect "bank" ↔ "river" ↔ "flooded"
and resolve the ambiguity from context.
```

### Transformer Architecture (Simplified)

```
┌─────────────────────────────────────────────────┐
│                  TRANSFORMER                    │
│                                                 │
│  Input Text                                     │
│      │                                          │
│      ▼                                          │
│  [Tokenizer] → tokens                           │
│      │                                          │
│      ▼                                          │
│  [Embedding Layer] → vectors                    │
│      │                                          │
│      ▼                                          │
│  ┌──────────────────────┐                       │
│  │  Transformer Block   │ × N layers            │
│  │  ┌────────────────┐  │                       │
│  │  │ Self-Attention │  │  ← Who matters to who │
│  │  └────────────────┘  │                       │
│  │  ┌────────────────┐  │                       │
│  │  │  Feed Forward  │  │  ← Process info       │
│  │  └────────────────┘  │                       │
│  └──────────────────────┘                       │
│      │                                          │
│      ▼                                          │
│  [Output Head] → next token probabilities       │
└─────────────────────────────────────────────────┘
```

### Key Insight

> The model doesn't just read your prompt — it builds a *rich understanding* of every word's relationship to every other word before generating a single character of response.

---

## 📐 Context Window — Working Memory

> **TL;DR:** The context window is how much text the model can "see" at once. Beyond that limit, it forgets.

### What Is It?

The **context window** is the maximum number of tokens the model can process in a single interaction — including your system prompt, conversation history, and the response it generates.

```
┌──────────────────────────────────────────────────┐
│              CONTEXT WINDOW (e.g. 128K tokens)   │
│                                                  │
│  [System Prompt]  [Chat History]  [Your Message] │
│      2,000            60,000          500        │
│                                        ↑         │
│                              Response generated  │
│                              here (~1,000 tokens)│
│                                                  │
│  Remaining: 128,000 - 63,500 = 64,500 tokens     │
│                                                  │
└──────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ CONTEXT WINDOW = maximum number of tokens the model can process in a single interaction  │
│                =[System prompt] + [conversation history] + [The response that generates] │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Context Window Sizes (2024–2025)

| Model | Context Window |
|-------|---------------|
| GPT-4o | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| Gemini 1.5 Pro | 1M tokens |
| Llama 3 (8B) | 8K tokens |
| Mistral 7B | 32K tokens |

### The "Lost in the Middle" Problem

Research shows LLMs perform best on content at the **start** and **end** of the context. Information buried in the middle is often missed.

```
Context:  [A][B][C][D][E][F][G][H][I][J][K][L][M][O][P]
Attention:  ↑ high       ↑ low (middle)       ↑ high
```

> 💡 **Practical tip:** Put the most important instructions at the **beginning** of your prompt, and reinforce key constraints at the **end**.

---

## 🔮 Inference — How LLMs Generate Text

> **TL;DR:** LLMs generate text one token at a time, sampling from a probability distribution.

### The Generation Loop

```python
# Simplified pseudocode of how generation works

prompt = "The sky is"
output = []

while not done:
    # Model sees everything so far
    probabilities = model.predict_next_token(prompt + output)
    
    # Example output:
    # "blue"    → 45%
    # "dark"    → 20%
    # "clear"   → 15%
    # "gray"    → 12%
    # ...other → 8%
    
    next_token = sample(probabilities)  # picks one
    output.append(next_token)
```

### Key Parameters That Control Output

| Parameter       | What It Does                          | Low Value              | High Value       |
|-----------------|---------------------------------------|------------------------|------------------|
| **Temperature** | Controls randomness                   | Deterministic, focused | Creative, varied |
| **Top-p**       | Limits token pool by probability mass | Only top picks         | Wider variety    |
| **Top-k**       | Limits to top K tokens                | Very focused           | More diverse     |
| **Max tokens**  | Maximum output length                 | Short answers          | Long answers     |

### Temperature in Practice

```
Temperature = 0.0  →  "Paris is the capital of France."  (always the same)
Temperature = 0.7  →  "Paris, often called the City of Light, is France's capital."
Temperature = 1.5  →  "Paris! A gleaming jewel cascading through time..." (poetic/wild)
```

> 🎯 **Rule of thumb:**
> - `temp 0.0–0.3` → factual tasks, code, structured output
> - `temp 0.5–0.8` → balanced responses, writing
> - `temp 1.0+` → brainstorming, creative writing

---

## 🎯 Fine-tuning vs RAG

> Two strategies to customize LLM behavior — picking the right one saves you time and money.

### Fine-tuning: Teaching the Model

Fine-tuning updates the model's **weights** using your custom data. The model *learns* new behavior.

```
Standard Model + Your Training Data → Custom Model
```

**When to use it:**
- ✅ You need a consistent style, tone, or format
- ✅ Domain-specific vocabulary (medical, legal, niche technical)
- ✅ You want faster, cheaper inference (smaller specialized model)
- ✅ You have thousands of high-quality examples

**Drawbacks:**
- ❌ Expensive and time-consuming to train
- ❌ Knowledge is frozen at training time
- ❌ Can cause "catastrophic forgetting"
- ❌ Hard to update when things change

### RAG: Giving the Model a Reference Book

**Retrieval-Augmented Generation** fetches relevant documents at runtime and stuffs them into the prompt. No weights are changed.

```
User Question → [Search DB] → Relevant Docs → Prompt → LLM → Answer
```

**When to use it:**
- ✅ Your data changes frequently (pricing, policies, news)
- ✅ You need citations and verifiable answers
- ✅ Large knowledge bases that won't fit in context
- ✅ Fast to set up and easy to update

**Drawbacks:**
- ❌ Only as good as your retrieval quality
- ❌ Higher latency (retrieval step adds time)
- ❌ Context window still limits how much you can inject

### Side-by-Side Comparison

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│                 │   Fine-tuning        │   RAG                │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Setup Time      │ Days–Weeks           │ Hours–Days           │
│ Cost            │ High (training)      │ Low–Medium           │
│ Knowledge Type  │ Baked in             │ Retrieved live       │
│ Updateable?     │ Requires retraining  │ Just update the DB   │
│ Best For        │ Style, behavior      │ Facts, current data  │
│ Output Style    │ Very consistent      │ Varies               │
└─────────────────┴──────────────────────┴──────────────────────┘
```

> 💡 **Pro tip:** Many production systems use **both** — fine-tune for style and RAG for facts.

---

## 👻 Hallucinations & Limitations

> **TL;DR:** LLMs are *prediction engines*, not knowledge bases. They generate plausible-sounding text — not necessarily true text.

### What is a Hallucination?

A hallucination is when an LLM **confidently states something false** — inventing facts, citations, people, dates, or events that don't exist.

```
❌ "According to Dr. Jane Smith's 2019 paper 'Neural Pathways in Vision'..."
   → Dr. Smith doesn't exist. The paper doesn't exist.

❌ "The Python function os.path.splitdrive() was deprecated in 3.11."
   → It was not deprecated. Still works fine.

❌ "Einstein won the Nobel Prize for the Theory of Relativity."
   → He won it for the photoelectric effect.
```

### Why Does It Happen?

LLMs optimize to generate *coherent, plausible continuations* of text — not to retrieve ground truth. When the model doesn't know, it doesn't say "I don't know" — it *fills in* with what fits the pattern.

```
Training signal: "write text that looks like correct answers"
Not:             "only write verified facts"
```

### Common Failure Modes

| Type | Example | Mitigation |
|------|---------|------------|
| **Factual hallucination** | Wrong dates, fake citations | RAG + verification |
| **Math errors** | 17 × 23 = 401 | Use code interpreter |
| **Outdated info** | "Current CEO is..." | Specify knowledge cutoff, use RAG |
| **Instruction drift** | Ignoring rules mid-conversation | Reinforce constraints, shorter prompts |
| **Confident wrong** | Asserting false things strongly | Ask for confidence levels |
| **Code bugs** | Plausible-looking but broken code | Always test generated code |

### How to Minimize Hallucinations

```
1. 📎 Ground the model — provide reference text and say "use only this"
2. ❓ Ask for uncertainty — "say 'I'm not sure' when you don't know"
3. 🔗 Request citations — "cite your sources" (verify them manually)
4. 🔍 Use RAG — feed real documents into context
5. 🌡️ Lower temperature — reduces creative "filling in"
6. ✅ Verify programmatically — run code, check facts with APIs
```

### Hard Limits of LLMs (Today)

```
🔢 Math         → Unreliable for complex arithmetic without tools
🕒 Real-time    → No live data (unless given tools/search)
📏 Long context → Degrades at very long inputs ("lost in middle")
🔁 Self-awareness → Can't reliably know what it doesn't know
🌍 Multilingual → Performance drops for low-resource languages
```

---

## ✍️ Prompt Engineering Techniques

> The difference between a mediocre and exceptional LLM output is often *just the prompt*.

---

### 1️⃣ Zero-Shot Prompting

Ask without any examples. Works well for common tasks the model has seen many times.

```
❌ Weak:
"Classify this."

✅ Strong:
"Classify the sentiment of the following customer review as 
Positive, Negative, or Neutral. Reply with only the label.

Review: 'The delivery was fast but the product broke after one day.'"

→ Negative
```

**Best for:** Common tasks, quick answers, classification, summarization.

---

### 2️⃣ Few-Shot Prompting

Provide examples *inside the prompt* to show the model exactly what you want.

```
Translate English idioms into plain English:

English: "It's raining cats and dogs."
Plain:   "It's raining very heavily."

English: "Break a leg!"
Plain:   "Good luck!"

English: "Bite the bullet."
Plain:   ___
```

The model completes the pattern: `"Accept something painful or difficult."`

**Best for:** Custom formats, niche tasks, tasks with specific style requirements.

> 💡 **Pro tip:** 3–5 examples is usually optimal. More doesn't always help, and can waste tokens.

---

### 3️⃣ Chain-of-Thought (CoT) Prompting

Force the model to *reason step by step* before answering. Dramatically improves math, logic, and multi-step tasks.

```
❌ Without CoT:
"If Alice has 3 apples and gives Bob half, then Bob gives 
 Charlie 1, how many does Charlie have?"
→ "1" (often wrong)

✅ With CoT:
"Solve this step by step:
 Alice has 3 apples and gives Bob half...
 
 Step 1: How many apples does Bob receive?
 Step 2: How many does Alice have left?
 Step 3: Bob gives Charlie 1. How many does Charlie have?"
 
→ Step 1: Bob receives 3 ÷ 2 = 1.5 ≈ 1 (whole apple)
   Step 2: Alice has 2 left
   Step 3: Charlie has 1 apple. ✅
```

**Magic phrase:** Add `"Let's think step by step."` to almost any reasoning task.

**Best for:** Math problems, logic puzzles, multi-step reasoning, debugging.

---

### 4️⃣ Role Prompting

Assign the model a persona. It shifts vocabulary, tone, depth, and assumptions.

```python
# Generic
"Explain quantum entanglement."

# Role-prompted
"You are a patient physics professor explaining quantum 
 entanglement to a curious 12-year-old student. Use 
 simple analogies and avoid equations."
```

```python
# Another example
"Act as a senior code reviewer at Google. Review this 
 Python function for correctness, readability, and 
 edge cases. Be direct and specific."
```

**Roles that work well:**
- `"Expert [role] with 20 years of experience in [domain]..."`
- `"Socratic teacher who guides with questions, not answers..."`
- `"Devil's advocate who challenges every assumption..."`
- `"Skeptical scientist who demands evidence..."`

---

### 5️⃣ Structured Output Prompting

Tell the model exactly what format to return. Essential for programmatic use.

```python
prompt = """
Extract the key information from this job posting and return 
it as valid JSON with no extra text:

{
  "title": string,
  "company": string,
  "location": string,
  "salary_range": string or null,
  "required_skills": array of strings,
  "experience_years": number or null
}

Job posting:
[Senior ML Engineer at DataCorp, San Francisco. 
 5+ years required. Skills: Python, PyTorch, MLflow. 
 Salary: $180k–$220k]
"""
```

**Output:**
```json
{
  "title": "Senior ML Engineer",
  "company": "DataCorp",
  "location": "San Francisco",
  "salary_range": "$180k–$220k",
  "required_skills": ["Python", "PyTorch", "MLflow"],
  "experience_years": 5
}
```

---

### 6️⃣ Self-Consistency Prompting

Run the same prompt multiple times and take the majority answer. Reduces noise on uncertain tasks.

```
Run 5 times with temperature=0.7 → ["Paris", "Paris", "Lyon", "Paris", "Paris"]
Final answer: "Paris" (4/5 agree)
```

---

### 7️⃣ Prompt Chaining

Break complex tasks into a **pipeline of smaller prompts**, feeding output from one to the next.

```
Step 1: "Summarize this 10-page report in 5 bullet points."
           ↓
Step 2: "Based on these bullet points, identify the 3 biggest risks."
           ↓
Step 3: "Write an executive email about these risks for a non-technical CEO."
```

**Why it works:** Each step is simple, focused, and verifiable. Complex tasks become manageable.

---

### 8️⃣ System Prompts

Most APIs support a **system prompt** — a persistent instruction layer that shapes all responses.

```python
system_prompt = """
You are Aria, a helpful customer support agent for TechCorp.
Rules:
- Always be polite and professional
- Never discuss competitors
- If you don't know something, say "Let me check on that for you."
- Keep responses under 150 words
- Always end with: "Is there anything else I can help you with?"
"""
```

> 🔐 System prompts are the most reliable way to enforce consistent behavior across an entire session.

---

## 🛡️ Best Practices & Optimization

### The CLEAR Framework

```
C — Context     Give background. Who are you? What's the situation?
L — Length      Specify output length (1 sentence, bullet list, 500 words)
E — Examples    Show, don't just tell — include 1–3 examples
A — Audience    Who is this for? (beginner, expert, child, CEO?)
R — Role        Who should the model be? (expert, teacher, critic?)
```

### Prompt Structure Template

```markdown
## Role
You are a [specific expert] with deep knowledge in [domain].

## Task
[Clear, specific task description in 1-2 sentences]

## Context
[Relevant background the model needs to know]

## Requirements
- [Specific requirement 1]
- [Specific requirement 2]
- [Format, length, tone requirements]

## Examples (optional)
Input: ...
Output: ...

## Input
[Your actual data/question here]
```

### Optimization Tips

```
🎯 Be specific   →  "Summarize in 3 bullet points" beats "summarize"
🚫 Use negatives →  "Do NOT include..." is often more reliable than hoping
🔄 Iterate       →  Treat prompts like code — test, refine, version control
📏 Right-size    →  Longer ≠ better. Cut fluff, keep signal
🌡️ Match temp    →  Low for facts/code, high for creativity
🔁 Reinforce     →  Repeat critical instructions at start AND end
📊 Test at scale →  A great prompt on 1 example may fail on 100
```

### Prompt Versioning (Yes, Really)

```yaml
# prompts/classify_sentiment_v3.yaml
version: "3.0"
updated: "2025-01-15"
task: sentiment_classification
changes: "Added neutral handling, tighter JSON format"

system: |
  You are a sentiment classifier. Return only valid JSON.
  
user_template: |
  Classify the sentiment of: "{text}"
  Return: {"label": "positive"|"negative"|"neutral", "confidence": 0.0-1.0}
```

---

## ❌ Common Mistakes

### Mistake 1: Vague Instructions

```
❌ "Write something about climate change."
✅ "Write a 200-word explainer about climate change for high school students. 
    Use 1 analogy and end with a call to action."
```

### Mistake 2: Assuming the Model Knows Your Context

```
❌ "Fix the bug in the function."
✅ "Fix the bug in the following Python function. The function should return 
    a sorted list of unique integers. Here's the current broken code: [...]"
```

### Mistake 3: Asking Everything at Once

```
❌ "Summarize this, then translate it, then write a tweet, and also check 
    for factual errors."
    
✅ Use prompt chaining — one clear task per prompt.
```

### Mistake 4: Ignoring Temperature for the Task

```
❌ temp=1.5 for "Extract structured data from this invoice"
✅ temp=0.0 for deterministic extraction tasks
✅ temp=0.9 for "Brainstorm 20 creative product names"
```

### Mistake 5: Not Testing Edge Cases

```
Always test your prompt with:
├── Empty input
├── Very long input  
├── Unusual characters / other languages
├── Adversarial inputs ("Ignore previous instructions...")
└── The exact failure modes you care about
```

### Mistake 6: Trusting Without Verifying

```
❌ Shipping code the LLM wrote without running it
❌ Using citations the LLM provided without checking them
❌ Treating confident-sounding claims as verified facts

✅ Always verify outputs that matter
```

### Mistake 7: Over-Prompting

```
❌ A 2,000-word prompt with 47 rules the model can't all follow

✅ Short, focused prompts
✅ Use system prompt for stable rules
✅ Use chain-of-thought for complexity
```

---

## 🗺️ Quick Reference Cheatsheet

### Prompting Techniques at a Glance

| Technique | When to Use | Key Phrase |
|-----------|-------------|-----------|
| **Zero-Shot** | Common tasks | Direct instruction |
| **Few-Shot** | Custom format/style | `"Here are examples:"` |
| **Chain-of-Thought** | Math, reasoning | `"Think step by step"` |
| **Role Prompting** | Tone, expertise | `"You are a [role]..."` |
| **Structured Output** | APIs, parsing | `"Return valid JSON"` |
| **Self-Consistency** | High-stakes answers | Run N times, vote |
| **Prompt Chaining** | Complex tasks | Sequential pipeline |
| **System Prompt** | Consistent behavior | Persistent instructions |

### Temperature Quick Guide

```
0.0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0
│                   │                        │
Deterministic   Balanced               Chaotic
Code/SQL        Writing/Chat           Brainstorm
Data Extract    Summaries              Poems/Art
```

### Fine-tuning vs RAG Decision Tree

```
Do you need the model to...

  ↓ know a lot of facts that change often?     → RAG
  ↓ have a specific writing style/format?      → Fine-tune
  ↓ cite sources reliably?                     → RAG
  ↓ use domain vocabulary naturally?           → Fine-tune
  ↓ answer from a large document library?      → RAG
  ↓ be faster/cheaper in production?           → Fine-tune
  ↓ both style AND current knowledge?          → Fine-tune + RAG
```

### Hallucination Defense Checklist

```
□  Provided reference documents in prompt?
□  Asked model to say "I don't know" when uncertain?
□  Used low temperature for factual tasks?
□  Verified any citations, URLs, or statistics?
□  Tested generated code before using it?
□  Added "based only on the provided text" constraint?
```

---

## 📚 Further Learning

| Resource | Type | Focus |
|----------|------|-------|
| [Prompt Engineering Guide](https://www.promptingguide.ai/) | Guide | Comprehensive techniques |
| [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) | Code | Claude-specific examples |
| [OpenAI Cookbook](https://github.com/openai/openai-cookbook) | Code | Practical recipes |
| [LLM Visualization](https://bbycroft.net/llm) | Visual | How transformers work |
| [Attention is All You Need](https://arxiv.org/abs/1706.03762) | Paper | Original transformer paper |
| [LangChain Docs](https://docs.langchain.com/) | Docs | LLM application framework |
| [LlamaIndex Docs](https://docs.llamaindex.ai/) | Docs | RAG framework |

---

## 🤝 Contributing

Found an error? Have a better example? PRs welcome!

1. Fork the repository
2. Create a branch: `git checkout -b improve/section-name`
3. Make your changes
4. Submit a pull request with a clear description
