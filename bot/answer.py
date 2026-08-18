#!/usr/bin/env python3
"""ma-deal-design Issue 问答机器人。

流程: 每日限额检查 -> 私有知识库检索(纯stdlib分块+二元组IDF) -> GitHub Models推理 -> 回帖。
仅用标准库,无第三方依赖。环境变量由 workflow 注入。
"""
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

ISSUE_NUMBER = int(os.environ["ISSUE_NUMBER"])
ISSUE_TITLE = os.environ.get("ISSUE_TITLE", "")
ISSUE_BODY = os.environ.get("ISSUE_BODY") or ""
ISSUE_USER = os.environ.get("ISSUE_USER", "")
REPO = os.environ["REPO"]
GH_TOKEN = os.environ["GH_TOKEN"]
MODELS_PAT = os.environ["MODELS_PAT"]
MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-4o-mini")
REFS_DIR = Path(os.environ.get("REFS_DIR", "refs-private/references"))
SYSTEM_PROMPT_FILE = Path(os.environ.get("SYSTEM_PROMPT_FILE", "bot/system_prompt.md"))
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "3"))
MODELS_URL = "https://models.github.ai/inference/chat/completions"
CHUNK_CHARS = 1400
TOP_K = 6
MAX_QUESTION_CHARS = 4000
MAX_ANSWER_TOKENS = 1800


def gh(*args: str) -> None:
    subprocess.run(["gh", *args], check=True)


def comment(body: str) -> None:
    p = Path("/tmp/ma_answer.md")
    p.write_text(body, encoding="utf-8")
    gh("issue", "comment", str(ISSUE_NUMBER), "--body-file", str(p))


def check_daily_quota() -> None:
    """限制: 每位用户每日最多 DAILY_LIMIT 个新提问(含当日更早的)。"""
    today = date.today().isoformat()
    q = urllib.request.quote(f"repo:{REPO} is:issue author:{ISSUE_USER} created:>={today}")
    req = urllib.request.Request(
        f"https://api.github.com/search/issues?q={q}",
        headers={"Authorization": f"Bearer {GH_TOKEN}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            n = json.load(r).get("total_count", 0)
    except Exception:
        return  # 查询失败不拦截提问
    if n > DAILY_LIMIT:
        comment(
            f"@{ISSUE_USER} 你今天的提问数已达上限({DAILY_LIMIT}个/天,含本条)。\n\n"
            "为保证服务质量,请明天再来;如需更高额度,请在 issue 标题注明【加急】并说明理由,维护者会人工处理。"
        )
        sys.exit(0)


# ---------- 检索: 分块 + 二元组IDF ----------

def tokenize(text: str) -> set:
    """英文按词、中文按字符二元组,统一小写。"""
    out = set()
    for m in re.finditer(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        t = m.group()
        if re.match(r"[\u4e00-\u9fff]", t):
            out |= {t[i:i + 2] for i in range(len(t) - 1)}
        else:
            out.add(t)
    return out


def chunk_file(path: Path):
    """按标题行分块,单块约 CHUNK_CHARS 字符。产出 (来源, 文本)。"""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    heading, buf, size = path.stem, [], 0
    for ln in lines:
        if re.match(r"^#{1,3} ", ln) and size > CHUNK_CHARS * 0.6:
            yield (heading, "\n".join(buf).strip())
            heading = f"{path.stem} :: {ln.lstrip('# ').strip()[:40]}"
            buf, size = [], 0
        buf.append(ln)
        size += len(ln)
    if buf and "\n".join(buf).strip():
        yield (heading, "\n".join(buf).strip())


def load_chunks() -> list:
    chunks = []
    for f in sorted(REFS_DIR.rglob("*.md")):
        chunks.extend(chunk_file(f))
    return chunks


def retrieve(chunks: list, query: str) -> list:
    q = tokenize(query)
    if not q or not chunks:
        return []
    df = {}
    sets = []
    for _, text in chunks:
        s = tokenize(text)
        sets.append(s)
        for t in s:
            df[t] = df.get(t, 0) + 1
    n = len(chunks)
    scored = []
    for i, s in enumerate(sets):
        score = sum(math.log(n / df[t]) for t in q if t in s and df[t] < n)
        if score > 0:
            scored.append((score, i))
    scored.sort(reverse=True)
    return [chunks[i] for _, i in scored[:TOP_K]]


# ---------- 推理: GitHub Models ----------

def generate(system: str, user_msg: str) -> str:
    payload = json.dumps({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": MAX_ANSWER_TOKENS,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        MODELS_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {MODELS_PAT}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    check_daily_quota()

    question = (ISSUE_TITLE.strip() + "\n\n" + ISSUE_BODY.strip()).strip()[:MAX_QUESTION_CHARS]

    system = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    if not system:
        system = "你是并购交易专家助手,基于参考资料用中文回答,输出结构化分析,不构成法律/税务/投资意见。"

    chunks = load_chunks()
    hits = retrieve(chunks, question)
    ctx = "\n\n".join(f"【资料{i}】{src}\n{text}" for i, (src, text) in enumerate(hits, 1))
    user_msg = (
        f"用户提问:\n{question}\n\n"
        + (f"内部参考资料(供你组织答案,禁止逐字复述原文):\n{ctx}" if ctx else "知识库无直接命中,请基于通用专业知识回答并明确说明。")
    )

    try:
        answer = generate(system, user_msg)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:300]
        comment(
            f"@{ISSUE_USER} 回答生成失败(HTTP {e.code}),通常是模型服务限流或密钥问题。\n"
            f"维护者会看到此 issue 并人工补答。错误摘要: `{body}`"
        )
        print(f"LLM error {e.code}: {body}", file=sys.stderr)
        sys.exit(0)  # 不打失败标签,留待人工

    sources = sorted({src for src, _ in hits})
    footer = (
        "\n\n---\n"
        f"**参考来源**: {', '.join(f'`{s}`' for s in sources) if sources else '(知识库无直接命中,通用专业回答)'}\n"
        f"**模型**: `{MODEL_ID}` · 自动生成于 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        "以上内容仅供学习参考,不构成法律、税务或投资意见;引用法规请自行核实现行有效性。\n"
        f"每日提问上限 {DAILY_LIMIT} 个。"
    )
    comment(answer + footer)


if __name__ == "__main__":
    main()
