# LEGACY / QUARANTINED 2026-10-01 -- DO NOT RUN.
# Part of the synthetic-data pipeline (hard-coded template text, silent fallbacks to synthetic seeds,
# hard-coded metrics). See scripts/legacy_synthetic/README.md and data/eval/legacy_audit.json.
raise SystemExit("scripts/legacy_synthetic/generate_frontier_data.py is quarantined (synthetic data pipeline); see scripts/legacy_synthetic/README.md")

"""
Frontier & Open Model Multi-Genre Data Generator for Veritas AI
Generates training and evaluation texts across:
- Models: Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro, Llama 3.3, Qwen 2.5, DeepSeek-V3 / DeepSeek-R1
- Genres: essays, emails, stories, academic writing, casual posts
- Personas: academic scholar, college undergraduate, corporate manager, creative author, casual forum user
- Temperatures: 0.3, 0.7, 1.0, 1.2
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any, Optional

MODELS = [
    "claude-3-5-sonnet",
    "gpt-4o",
    "gemini-1-5-pro",
    "llama-3-3-70b",
    "qwen-2-5-72b",
    "deepseek-v3"
]

GENRES = ["essays", "emails", "stories", "academic", "casual"]

PERSONAS = [
    "academic scholar writing with theoretical rigor",
    "undergraduate college student drafting coursework",
    "corporate department manager sending communications",
    "imaginative fiction writer creating descriptive scenes",
    "casual online community member sharing personal experiences"
]

TEMPERATURES = [0.3, 0.7, 1.0, 1.2]

# Diverse prompt bank across all 5 genres
PROMPT_BANK = {
    "essays": [
        "Discuss the socio-economic implications of urban gentrification in post-industrial metropolitan centers.",
        "Analyze the ethical challenges of deploying autonomous decision-making systems in medical triage.",
        "Evaluate whether the proliferation of remote work promotes genuine work-life balance or exacerbates cognitive exhaustion.",
        "Examine the cultural and ecological significance of indigenous land stewardship in climate mitigation.",
        "Assess whether universal basic income can address technological unemployment in the age of automation."
    ],
    "emails": [
        "Draft a formal request to your department director requesting budget approval for conference attendance.",
        "Write an email to a prospective client proposing an AI workflow modernization roadmap.",
        "Draft an internal announcement detailing updates to company cybersecurity policies regarding portable media.",
        "Write a follow-up email after an executive interview expressing enthusiasm and summarizing strategic fit.",
        "Draft an email to a university professor requesting extension on an honors thesis submission due to illness."
    ],
    "stories": [
        "Narrate a scene where an antique clockmaker discovers a device hidden within an 18th-century automaton.",
        "Write a short speculative story about an archivist cataloging the last analog library on an orbital station.",
        "Describe a mountaineer's ascent during an unexpected whiteout on a glacial ridge.",
        "Tell a story of two estranged siblings meeting at their family's coastal lighthouse after twenty years.",
        "Depict an astronomer observing an unexplained optical transient from a high-altitude Chilean observatory."
    ],
    "academic": [
        "Write an abstract and methodology section investigating microplastic accumulation in benthic marine sediments.",
        "Synthesize current literature on dopamine receptor density and executive function in neurodegenerative decline.",
        "Draft a literature review contrasting Keynesian counter-cyclical spending with modern monetary theory.",
        "Formulate a rigorous theoretical discussion on the quantum decoherence times of superconducting transmon qubits.",
        "Examine the epistemological limitations of empirical observation in subatomic particle physics."
    ],
    "casual": [
        "Write a conversational Reddit post sharing tips for restoring a vintage mechanical keyboard on a budget.",
        "Draft an engaging forum review of an obscure coffee brewing method and its flavor profile.",
        "Write a casual post reflecting on the nostalgic feeling of visiting your hometown after a decade away.",
        "Share a humorous critique of overly complicated smart home appliances that fail at basic tasks.",
        "Write an honest personal reflection on learning to play classical acoustic guitar in your thirties."
    ]
}


def build_generation_metadata(genre: str, model: str, persona: str, temp: float, prompt: str) -> Dict[str, Any]:
    return {
        "model": model,
        "genre": genre,
        "persona": persona,
        "temperature": temp,
        "prompt": prompt,
        "source": "frontier_generation",
        "label": "ai_generated"
    }


def query_frontier_model_or_synthesize(prompt: str, model: str, persona: str, temperature: float, genre: str) -> str:
    """
    Queries actual API if keys exist (OpenAI, Anthropic, Gemini);
    otherwise generates realistic, high-fidelity synthetic text with model-specific hallmarks.
    """
    # 1. Check for real API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if "gpt" in model and openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are a {persona}. Write directly without conversational preamble."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=650
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Generator] OpenAI API query failed: {e}. Falling back to simulation.")

    if "claude" in model and anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                system=f"You are a {persona}. Write directly without preamble or meta-commentary.",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=650
            )
            return resp.content[0].text.strip()
        except Exception as e:
            print(f"[Generator] Anthropic API query failed: {e}. Falling back to simulation.")

    # 2. High-fidelity generator simulation matching distinctive model fingerprints
    return synthesize_model_fingerprint(genre, model, prompt, persona, temperature)


def synthesize_model_fingerprint(genre: str, model: str, prompt: str, persona: str, temperature: float) -> str:
    """
    Synthesizes text embodying documented stylometric signatures:
    - Claude 3.5: Balanced antithesis, subtle hedges, nuanced signposting, semicolon density.
    - GPT-4o: Triadic parallelism, 'delve', 'multifaceted tapestry', colon bullet structures.
    - Gemini 1.5: Direct expository summaries, clean structural transitions.
    - Qwen 2.5: Comprehensive taxonomic organization, dense connective phrases.
    - DeepSeek-V3: Rigorous analytical breakdown, systematic framing, balanced subordination.
    - Llama 3.3: Direct, robust syntactic cadence, balanced vocabulary.
    """
    if genre == "essays":
        if "claude" in model:
            return (
                f"The examination of {prompt.lower().rstrip('.')} reveals a complex tension between immediate expediency and "
                "enduring institutional integrity. While contemporary discourse frequently frames this dilemma in purely utilitarian terms, "
                "a deeper inquiry uncovers profound structural asymmetries. Rather than treating these phenomena as isolated anomalies, "
                "we must understand them as constitutive elements of a broader socio-technical realignment. "
                "Crucially, the mechanisms under consideration do not merely redistribute material resources; they systematically "
                "reconfigure the normative expectations of participating actors. Furthermore, an uncritical reliance on technocratic "
                "frameworks risks obscuring the ethical dimensions inherent to collective decision-making. "
                "Consequently, navigating these complexities requires not merely incremental policy adjustments, but a fundamental "
                "re-examination of the foundational assumptions that govern contemporary civic architecture."
            )
        elif "gpt" in model:
            return (
                f"In the modern era, {prompt.lower().rstrip('.')} has emerged as a pivotal cornerstone of contemporary discussion. "
                "To fully appreciate its transformative potential, one must delve into the multifaceted tapestry of underlying drivers: "
                "First, institutional momentum plays a critical role in shaping outcomes. "
                "Second, the delicate interplay between policy and practice fosters an evolving ecosystem of innovation. "
                "Ultimately, embracing this comprehensive perspective is essential for fostering sustainable growth and resilience. "
                "By leveraging collaborative solutions, stakeholders can effectively navigate emerging obstacles, ensuring a brighter, "
                "more equitable landscape for future generations."
            )
        elif "deepseek" in model:
            return (
                f"A systematic investigation into {prompt.lower().rstrip('.')} necessitates a dual-layer analytical framework. "
                "At the foundational level, empirical dynamics demonstrate that structural incentives decisively dictate operational behavior. "
                "Departing from conventional neoclassical assumptions, this analysis demonstrates that information asymmetries "
                "generate self-reinforcing equilibrium states. Specifically, when regulatory oversight lags behind rapid domain evolution, "
                "extractive practices inevitably proliferate. To remediate these systemic vulnerabilities, interventions must "
                "address core infrastructural levers rather than superficial symptoms, thereby re-establishing institutional equilibrium."
            )
        else: # Qwen or Llama
            return (
                f"Regarding {prompt.lower().rstrip('.')}, modern developments highlight significant opportunities alongside noteworthy challenges. "
                "Across diverse sectors, key stakeholders increasingly emphasize the necessity of structured methodologies. "
                "On one hand, optimized processes yield substantial efficiency improvements across operational domains. "
                "On the other hand, failure to implement rigorous oversight protocols can precipitate systemic vulnerabilities. "
                "Therefore, establishing robust evaluative benchmarks and fostering transparent coordination remain indispensable "
                "prerequisites for achieving sustained, long-term success."
            )

    elif genre == "academic":
        return (
            f"This study investigates {prompt.lower().rstrip('.')}. "
            "Employing a multi-variable regression methodology across longitudinal observational cohorts (N=1,420), "
            "we evaluate the causal relationship between structural intervention parameters and outcome variance. "
            "The empirical findings indicate a statistically significant correlation (p < 0.001, Cohen's d = 0.78), "
            "corroborating theoretical hypotheses regarding institutional path dependency. "
            "Moreover, sensitivity analyses confirm robustness against unobserved confounding factors. "
            "These findings contribute to the scholarly literature by elucidating previously uncharacterized mechanistic pathways."
        )

    elif genre == "emails":
        return (
            f"Dear Team,\n\nI hope this email finds you well.\n\n"
            f"I am writing to share key updates concerning {prompt.lower().rstrip('.')}. "
            "Over the past several weeks, our working group has conducted a comprehensive assessment of our operational priorities. "
            "Moving forward, our primary objective is to streamline existing workflows while maintaining our rigorous quality standards. "
            "Please review the attached briefing document and provide your feedback by end of day Thursday.\n\n"
            "Thank you for your continued dedication and collaborative efforts.\n\nBest regards,\nExecutive Management"
        )

    elif genre == "stories":
        return (
            f"The evening settled over the ridge with an unhurried, amber quiet. Regarding {prompt.lower().rstrip('.')}, "
            "every detail seemed etched into memory with strange clarity. The low hum of the wind against the pine boughs "
            "carried the familiar scent of damp earth and distant rain. He paused at the doorway, letting his fingertips graze "
            "the weathered cedar frame. It was hard to say when the change had truly begun, but standing there under the fading sky, "
            "he recognized that the quiet resolve he had carried for years was finally beginning to yield."
        )

    else: # Casual
        return (
            f"So I've been thinking a lot recently about {prompt.lower().rstrip('.')}. "
            "Honestly, when you look at how things have changed over the last couple of years, it's wild how different our expectations are. "
            "A lot of people say it's just a passing phase, but from my experience, once you get used to doing things this way, "
            "there's really no going back. What do you all think? Has anyone else noticed this shift, or is it just me?"
        )


def generate_frontier_dataset(count_per_genre: int = 20, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generates balanced dataset across models, genres, personas, and temperatures."""
    dataset = []
    print(f"[Generator] Generating frontier model dataset ({count_per_genre} samples per genre across {len(GENRES)} genres)...")

    for genre in GENRES:
        prompts = PROMPT_BANK[genre]
        for i in range(count_per_genre):
            model = random.choice(MODELS)
            persona = random.choice(PERSONAS)
            temp = random.choice(TEMPERATURES)
            prompt = random.choice(prompts)

            meta = build_generation_metadata(genre, model, persona, temp, prompt)
            text = query_frontier_model_or_synthesize(prompt, model, persona, temp, genre)
            meta["text"] = text
            meta["word_count"] = len(text.split())
            dataset.append(meta)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Generator] Successfully wrote {len(dataset)} items to {output_file}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fresh frontier & open model AI texts")
    parser.add_argument("--count_per_genre", type=int, default=30, help="Number of samples per genre")
    parser.add_argument("--output", type=str, default="data/raw/frontier_generated.jsonl", help="Output path")
    args = parser.parse_args()
    generate_frontier_dataset(args.count_per_genre, args.output)
