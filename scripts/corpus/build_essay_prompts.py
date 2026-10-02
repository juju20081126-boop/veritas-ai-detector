#!/usr/bin/env python
"""
Append student-essay prompts (IELTS/TOEFL-style topics, written by us: they are instructions, not model output or
human text) to data/corpus/prompts.jsonl. Each topic gets three styles:

  plain        "Write an essay of about N words on the following topic: ..."
  student      "I'm a high school student ... please write ..."
  esl_persona  "... as if you were an intermediate English learner (CEFR B1) ... let a few small grammar mistakes slip in"

The ESL-persona style is a realistic evasion route (an LLM imitating learner English) and it keeps "learner style" from
becoming a shortcut for the human label (W&I+LOCNESS essays are human-only). All three style variants of a topic share a
split (assignment is per topic). Idempotent: rows with task_type == "essay" are replaced.

  python scripts/corpus/build_essay_prompts.py
"""

import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils  # noqa: E402

TOPICS = [
    "Some people think universities should teach practical job skills, while others think theoretical knowledge matters more. Discuss both views and give your opinion.",
    "Do you agree or disagree: social media does more harm than good for teenagers?",
    "Should governments spend money on space exploration, or should that money be used to solve problems on Earth?",
    "Many people now work from home. What are the advantages and disadvantages of this trend?",
    "Is it better to live in a big city or in a small town? Give reasons for your answer.",
    "Some people believe that children should start learning a foreign language in primary school. To what extent do you agree?",
    "Technology has made our lives easier but has also created new problems. Discuss.",
    "Should public transport be free for everyone? Explain your view.",
    "Do you think it is important to preserve traditional customs in a modern world?",
    "Some people prefer to spend their holidays at home; others like to travel abroad. Which do you prefer and why?",
    "Is competition among students a good or a bad thing for their development?",
    "Should schools require students to wear uniforms? Give reasons.",
    "Many young people spend a lot of time playing video games. Is this a positive or negative development?",
    "How important is it for a country to protect its own language and culture?",
    "Some people think that the best way to reduce crime is to give longer prison sentences. Do you agree?",
    "Should university education be free for all citizens?",
    "What are the main causes of stress among students, and what can be done about it?",
    "Is it ethical to use animals in scientific research?",
    "The Internet has changed how people read the news. Is this change mainly positive or negative?",
    "Some people say money is the key to happiness; others disagree. What is your opinion?",
    "Should companies be allowed to track what their employees do online at work?",
    "Do the benefits of tourism outweigh the problems it creates for local communities?",
    "Is it better to have a few close friends or many casual friends?",
    "Online shopping is replacing traditional stores. Discuss the advantages and disadvantages.",
    "Should governments ban single-use plastic products? Explain your position.",
    "Some people think young people today have less respect for older generations. Do you agree?",
    "Is a gap year before university a good idea for students?",
    "Should everyone learn to cook at school? Give reasons for your view.",
    "Does reading books still matter in the age of television and the Internet?",
    "Artificial intelligence will take many jobs in the next twenty years. Is this a threat or an opportunity?",
    "Should professional athletes earn much more money than teachers and nurses?",
    "What role should parents play in choosing their children's career?",
    "Is volunteering a good way for teenagers to learn about the world?",
    "Some people think zoos should be closed. Others say they protect endangered species. Discuss both views.",
    "Is it a good idea to take part in school or university exchange programmes?",
    "Should people be required to retire at a fixed age?",
    "Group projects are common in school. Are they a good way to learn?",
    "Is online learning as effective as learning in a classroom?",
    "How can cities reduce traffic congestion and air pollution?",
    "Should the government control the prices of basic foods such as rice and bread?",
]
STYLES = {
    "plain": "Write an essay of about {w} words on the following topic: {t}",
    "student": "I'm a high school student and I have to write an essay. Please write a {w}-word essay on this topic: {t}",
    "esl_persona": ("Write a {w}-word essay on this topic as if you were an intermediate English learner (CEFR B1): use simple "
                    "vocabulary and sentence structures, and let a few small grammar mistakes slip in naturally. Topic: {t}"),
}
WORDS = [150, 200, 250, 300, 400]
GEN_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5"]


def main():
    path = os.path.join(io_utils.DATA, "corpus", "prompts.jsonl")
    prompts = [p for p in io_utils.read_jsonl(path) if p.get("task_type") != "essay"]
    rng = random.Random(20261006)
    order = list(range(len(TOPICS)))
    rng.shuffle(order)
    split_of = {}
    for rank, ti in enumerate(order):
        split_of[ti] = "locked" if rank < 8 else ("dev" if rank < 14 else "train")
    n = 0
    for ti, topic in enumerate(TOPICS):
        for style, tpl in STYLES.items():
            n += 1
            w = rng.choice(WORDS)
            prompts.append({
                "prompt_id": f"essa-{ti:02d}-{style[:3]}", "task_type": "essay", "domain": "student_essay", "split": split_of[ti],
                "instruction": tpl.format(w=w, t=topic), "target_words": w, "style": style, "source": "authored:essay-topics",
                "source_doc": f"topic-{ti:02d}", "matched_human": False, "gen_models": list(GEN_MODELS)})
    io_utils.write_jsonl(path, prompts)
    from collections import Counter
    ess = [p for p in prompts if p["task_type"] == "essay"]
    print("essay prompts:", len(ess), dict(Counter(p["split"] for p in ess)), dict(Counter(p["style"] for p in ess)))
    print("total prompts:", len(prompts))


if __name__ == "__main__":
    main()
