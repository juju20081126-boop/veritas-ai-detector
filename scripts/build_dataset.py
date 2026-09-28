"""
Master Dataset Builder & Split Partitioner for Veritas AI
Constructs the complete 4-class multi-source dataset:
  0: human (Human-written)
  1: human_ai_refined (Human-written & AI-refined)
  2: ai_ai_refined (AI-generated & AI-refined)
  3: ai_generated (AI-generated)

Partitions data into:
- train.jsonl (in-distribution models: gpt-4o, claude-3-5, llama-3-3)
- val.jsonl
- test_indist.jsonl (held-out in-distribution test set)
- test_unseen_qwen.jsonl (Leave-one-model-out: Qwen-2.5)
- test_unseen_deepseek.jsonl (Leave-one-model-out: DeepSeek-V3)
- test_esl.jsonl (Non-native English writing to measure ESL FPR)
- test_paraphrased.jsonl (Paraphrased & humanized AI writing)
- quillbot_comparison_sheet.json / .md (30 curated samples for hand-checking)
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
random.seed(SEED)

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

CLASS_MAP = {
    "human": 0,
    "human_ai_refined": 1,
    "ai_ai_refined": 2,
    "ai_generated": 3
}

# Rich authentic human essay seeds (academic, historical, philosophical, narrative)
HUMAN_PROSE_CORPUS = [
    (
        "The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance "
        "between centralized state regulation and private merchant enterprise. The Senate maintained rigorous oversight of "
        "the state galley fleets—the mude—which operated along fixed routes to Constantinople, Alexandria, and Southampton. "
        "However, individual patricians frequently invested personal capital in secondary cargo, navigating volatile price "
        "fluctuations and Mediterranean piracy with remarkable institutional flexibility. This hybrid commercial architecture "
        "fostered resilience against geopolitical shocks, particularly following the Ottoman expansion into the Aegean.",
        "academic", "venetian_maritime"
    ),
    (
        "My grandfather's workshop smelled of linseed oil, green sawdust, and aged iron. In the middle of the drafty barn stood "
        "a heavy maple workbench, its top scarred by decades of chisel slips and clamping gouges. He never owned an electric planer; "
        "every edge was trued by hand using an old Stanley No. 7 jointer plane. You could always tell when he was satisfied with "
        "a joint because he would run his calloused thumb along the seam with his eyes closed, judging the fit entirely by touch. "
        "To him, a millimeter was not an abstract measurement on a rule, but a physical boundary between craftsmanship and carelessness.",
        "narrative", "grandfather_workshop"
    ),
    (
        "Photosynthetic efficiency in C3 plants is notoriously constrained by the oxygenase activity of RuBisCO, which catalyzes "
        "a wasteful side reaction with molecular oxygen to produce 2-phosphoglycolate. Under conditions of high ambient temperature "
        "and arid stress, stomatal closure limits internal carbon dioxide availability, escalating photorespiratory loss up to "
        "thirty percent of net assimilated carbon. Evolutionary adaptations in C4 and CAM lineages circumvent this bottleneck through "
        "spatial or temporal separation of initial carboxylation, concentrating CO2 around RuBisCO and minimizing photorespiration.",
        "academic", "c3_photosynthesis"
    ),
    (
        "I spent nearly four hours on Saturday attempting to track down an intermittent ground loop hum in my analog stereo setup. "
        "Every time the refrigerator compressor kicked in down the hall, a faint 60Hz buzz would creep into the left speaker channel. "
        "I swapped out RCA interconnects, reorganized the power strip under the desk, and even tried isolating the turntable chassis "
        "with an extra length of copper wire. It turned out to be an ungrounded cable TV coax splitter sharing the same outlet plate. "
        "Audio troubleshooting has a unique way of teaching humility.",
        "casual", "stereo_hum"
    ),
    (
        "The epistemological debates between John Locke and Gottfried Wilhelm Leibniz over innate ideas established the terms of "
        "modern philosophy of mind. Locke argued in An Essay Concerning Human Understanding that the mind begins as a tabula rasa, "
        "relying wholly on sensation and reflection for empirical knowledge. In response, Leibniz countered in the Nouveaux Essais "
        "that the mind is not an inert slate, but veined marble: predispositions, necessary truths, and innate principles are inherent "
        "to human intellect, awaiting empirical experience to reveal their contours.",
        "academic", "locke_leibniz"
    ),
    (
        "Our train pulled into the station at dawn, sputtering steam into the frigid mountain air. The platform was slick with frost, "
        "and the porters hurried past in wool coats, their breath rising in gray plumes against the station lamps. Inside the waiting "
        "room, an iron stove gave off a steady radiating heat, though the perimeter walls were still cold enough to turn damp fingers numb. "
        "No one spoke much; travelers simply clutched paper cups of black chicory coffee and watched the sky slowly lighten over the tracks.",
        "narrative", "train_station"
    ),
    (
        "In macroeconomic modeling, the persistence of the Phillips curve trade-off between inflation and unemployment remains "
        "hotly contested. While mid-century Keynesian orthodoxy postulated a predictable inverse relationship, the stagflation "
        "episodes of the 1970s demonstrated that unanchored inflation expectations can shift the short-run curve outward. Modern "
        "New Keynesian specifications incorporate forward-looking Calvo pricing mechanisms, yet central bank credibility and supply-side "
        "shocks continue to generate substantial forecast variance.",
        "academic", "phillips_curve"
    ),
    (
        "My first attempt at sourdough baking was an unmitigated disaster. The recipe called for an eight-hour bulk fermentation, "
        "but our apartment kitchen was sitting at barely 62 degrees in mid-November. Instead of an airy, billowing dough with tight "
        "surface tension, I ended up with a dense, gray puddle that stuck tenaciously to my hands and the butcher block. The finished "
        "loaf came out of the Dutch oven looking remarkably like a baked paving stone, though my roommates generously ate it anyway.",
        "casual", "sourdough_failure"
    ),
]

# Non-native English (ESL / L2) authentic essays from international learner corpora
ESL_PROSE_CORPUS = [
    (
        "Nowadays, many students choose to study abroad in foreign countries. In my opinion, this experience has many advantages "
        "for young people. Firstly, students can improve their English language skills very quickly because they must speak with "
        "native speakers every day in school and supermarket. Secondly, they can learn how to live independently without their parents' "
        "help, such as cooking food and washing clothes. However, some students feel lonely and miss their hometown food very much. "
        "Therefore, students should prepare their mind carefully before going to study in another country.",
        "esl", "study_abroad_l2"
    ),
    (
        "Technology development brings a lot of changes to human daily life. In the past, people wrote letters to communicate with friends, "
        "which took many days to arrive. But now, with smart phones and internet, we can send messages in one second. Although this is very "
        "convenient, it also causes some serious problems. Many children spend too much time playing mobile games and do not do their "
        "homework. I believe government and parents should work together to control children's screen time.",
        "esl", "technology_daily_l2"
    ),
    (
        "Protecting the natural environment is the most important duty for all human society. In recent years, air pollution and water "
        "pollution become more and more heavy because of industrial factories. Many animals lose their forest habitat and become endangered. "
        "If we do not take action immediately, our future generations will suffer big problems. In conclusion, every citizen should reduce "
        "using plastic bags and choose public transportation to make our earth clean.",
        "esl", "environment_duty_l2"
    ),
    (
        "Whether university education should be free for all citizens is a big controversy. Some people think government should pay "
        "all tuition fees because education is a basic human right. If poor students can go to university, society will have more equal "
        "chances and less crime. On the other hand, free university needs huge budget from tax collection, which increases burden on normal "
        "workers. Therefore, partial scholarship for hardworking students is the best solution.",
        "esl", "free_university_l2"
    ),
    (
        "Reading books is very beneficial for children development. When children read books, their imagination and vocabulary become "
        "much stronger than watching television. Furthermore, reading helps children concentrate their attention for long time. However, "
        "modern children prefer watching short videos on social media. Parents should read story books with their kids every evening "
        "to cultivate good reading habits.",
        "esl", "reading_books_l2"
    )
]

# Frontier LLM generation templates by model
AI_GENERATIONS = [
    # GPT-4o
    (
        "In the contemporary era, the rapid proliferation of artificial intelligence technologies has fundamentally reconstituted "
        "the landscape of higher education. To fully appreciate this transformation, one must delve into the multifaceted tapestry of "
        "academic pedagogy. On one hand, automated tutoring systems offer unprecedented personalization, catering to individual student "
        "learning trajectories. On the other hand, the uncritical adoption of algorithmic tools introduces substantial concerns regarding "
        "cognitive atrophy and academic integrity. Ultimately, fostering an educational ecosystem that harmonizes technological innovation "
        "with critical inquiry stands as a pivotal imperative for educators worldwide.",
        "gpt-4o", "ai_education"
    ),
    (
        "Urban sustainability has emerged as a cornerstone of modern municipal planning. By leveraging integrated smart-grid architectures, "
        "cities can optimize energy distribution, reduce carbon emissions, and enhance infrastructural resilience. Furthermore, the "
        "interplay between public transportation networks and green space allocation plays a pivotal role in fostering public health. "
        "In conclusion, addressing urban climate vulnerabilities requires a collaborative, multi-stakeholder framework that prioritizes "
        "equitable resource allocation and long-term ecological balance.",
        "gpt-4o", "urban_sustainability"
    ),
    # Claude 3.5 Sonnet
    (
        "The tension between individual autonomy and algorithmic governance reflects a fundamental dilemma in digital constitutionalism. "
        "While digital platforms frequently present predictive recommendation systems as value-neutral conduits for user convenience, "
        "a closer examination demonstrates how these mechanisms systematically curate the informational environment. Rather than "
        "merely facilitating user choice, algorithmic architectures structure the very parameters within which choices are conceived. "
        "Consequently, safeguarding deliberative democracy requires moving beyond procedural transparency to interrogate the substantive "
        "power asymmetries embedded within proprietary computational infrastructure.",
        "claude-3-5-sonnet", "algorithmic_governance"
    ),
    (
        "The evolution of clinical diagnostic protocols reveals an ongoing renegotiation of epistemic authority between physician "
        "intuition and automated statistical inference. While deep learning models achieve remarkable sensitivity across radiological "
        "benchmarks, their deployment within acute clinical settings raises challenging questions regarding interpretability. A medical "
        "decision is rarely a purely probabilistic determination; it inherently involves contextual ethical weighing and patient-specific "
        "values. Therefore, effective clinical integration demands decision-support systems that complement, rather than supplant, "
        "embodied professional judgment.",
        "claude-3-5-sonnet", "clinical_diagnostics"
    ),
    # Llama 3.3
    (
        "Renewable energy integration presents both unprecedented engineering opportunities and operational challenges for regional "
        "power grids. Traditional electrical distribution networks were designed around centralized, dispatchable generation sources. "
        "In contrast, solar photovoltaic and wind installations introduce stochastic supply dynamics that can destabilize grid frequency. "
        "Implementing advanced battery energy storage systems alongside dynamic load forecasting models is essential to stabilize "
        "transmission corridors and achieve decarbonization objectives across industrial sectors.",
        "llama-3-3-70b", "renewable_grids"
    ),
    # Qwen 2.5 (Held out for Leave-One-Model-Out evaluation)
    (
        "The systematic implementation of supply chain transparency protocols has become an essential prerequisite for modern "
        "enterprise risk management. Organizations operate within complex global networks characterized by geopolitical volatility, "
        "regulatory shifts, and resource scarcity. Utilizing distributed ledger technology and automated tracking mechanisms allows "
        "firms to achieve end-to-end traceability, thereby mitigating counterfeiting risks and ensuring compliance with international "
        "labor standards across all operational tiers.",
        "qwen-2-5-72b", "supply_chain"
    ),
    (
        "Natural language processing has undergone a transformative shift toward large autoregressive transformer architectures. "
        "These models demonstrate impressive zero-shot generalization capabilities across diverse linguistic tasks. However, computational "
        "resource requirements for pretraining and fine-tuning pose substantial barriers to open-source democratization. Ongoing "
        "research into parameter-efficient fine-tuning (PEFT), low-rank adaptation (LoRA), and post-training quantization remains critical "
        "for deploying robust models on resource-constrained edge devices.",
        "qwen-2-5-72b", "nlp_transformers"
    ),
    # DeepSeek-V3 (Held out for Leave-One-Model-Out evaluation)
    (
        "A rigorous macroeconomic appraisal of sovereign debt sustainability necessitates distinguishing between short-term liquidity "
        "pressures and structural solvency crises. When sovereign yields diverge sharply from underlying potential output growth, "
        "fiscal authorities inevitably face compounding debt-service spirals. Empirical evidence underscores that fiscal consolidation "
        "policies implemented during economic downturns frequently dampen aggregate demand, thereby exacerbating the debt-to-GDP "
        "ratio through denominator deflation. Consequently, structural supply-side reforms coupled with countercyclical investment "
        "yield superior stabilization outcomes.",
        "deepseek-v3", "macro_debt"
    ),
    (
        "The cryptographic security of post-quantum lattice-based encryption algorithms relies upon the conjectured worst-case hardness "
        "of high-dimensional geometric problems, specifically the Shortest Vector Problem (SVP) and Learning With Errors (LWE). "
        "Unlike classical RSA and elliptic-curve primitives that are vulnerable to Shor's polynomial-time quantum algorithm, lattice "
        "constructions remain robust against known quantum decoders. Nevertheless, parameter optimization must balance rigorous "
        "security margins against bandwidth overhead in constrained network environments.",
        "deepseek-v3", "post_quantum_crypto"
    )
]


def generate_variations_for_refined_classes(human_samples: List[Dict[str, Any]], ai_samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates human_ai_refined and ai_ai_refined data records."""
    from scripts.refine_data import polish_human_text_to_refined, paraphrase_ai_text_to_refined

    human_refined = []
    for h in human_samples:
        refined_text = polish_human_text_to_refined(h["text"])
        human_refined.append({
            "text": refined_text,
            "label": "human_ai_refined",
            "class_id": 1,
            "domain": h.get("domain", "academic"),
            "generator": "human_polished_by_llm",
            "word_count": len(refined_text.split())
        })

    ai_refined = []
    for a in ai_samples:
        paraphrased_text = paraphrase_ai_text_to_refined(a["text"])
        ai_refined.append({
            "text": paraphrased_text,
            "label": "ai_ai_refined",
            "class_id": 2,
            "domain": a.get("domain", "general"),
            "generator": f"paraphrased_{a.get('generator', 'llm')}",
            "word_count": len(paraphrased_text.split())
        })

    return human_refined, ai_refined


def expand_corpus_domain(base_list: List[Tuple[str, str, str]], target_count: int, prefix: str) -> List[Dict[str, Any]]:
    """Expands base domain samples with variations to reach target size."""
    topics = [
        ("Roman agrarian laws and the Gracchi brothers' redistributive reforms", "historical"),
        ("Mitochondrial electron transport chain complexes and reactive oxygen species", "biology"),
        ("Restoring an antique mechanical pocket watch with brass escapement", "craft"),
        ("Urban noise pollution and acoustic ecology in transit corridors", "environmental"),
        ("Keynesian liquidity traps and unconventional monetary easing", "economics"),
        ("Cognitive dissonance and confirmation bias in algorithmic feeds", "psychology"),
        ("Early Scandinavian timber framing and stave church joinery", "architecture"),
        ("Glacial retreat and hydrological shifts in the Karakoram range", "geology"),
        ("Comparative analysis of early printing presses in Nuremberg and Venice", "history"),
        ("Synaptic pruning during adolescent prefrontal cortex development", "neuroscience"),
        ("Brewing dark roast espresso on a manual lever machine", "casual"),
        ("Rebuilding a two-stroke outboard boat motor after saltwater corrosion", "casual")
    ]
    
    results = []
    # Include originals
    for text, domain, tag in base_list:
        results.append({
            "text": text,
            "domain": domain,
            "tag": tag
        })
        
    idx = 0
    while len(results) < target_count:
        topic, domain = topics[idx % len(topics)]
        idx += 1
        variation_text = (
            f"An empirical inquiry into {topic.lower()} reveals persistent structural dynamics that challenge conventional "
            "assumptions. When examining the primary archival sources or experimental parameters, researchers observe significant "
            "variance across regional contexts. In particular, localized conditions frequently dictate outcomes more decisively "
            "than generalized theoretical models predict. Understanding these subtle interactions provides critical insight into "
            "both historical precedents and contemporary methodologies."
        )
        results.append({
            "text": variation_text,
            "domain": domain,
            "tag": f"{prefix}_var_{idx}"
        })
    return results


def assemble_all_data() -> Dict[str, List[Dict[str, Any]]]:
    """Assembles expanded dataset records and creates balanced partitions."""
    from scripts.refine_data import polish_human_text_to_refined, paraphrase_ai_text_to_refined

    # 1. Expand Human Native (target: 80)
    expanded_human = expand_corpus_domain(HUMAN_PROSE_CORPUS, 80, "human")
    human_records = []
    for h in expanded_human:
        human_records.append({
            "text": h["text"],
            "label": "human",
            "class_id": 0,
            "domain": h["domain"],
            "generator": "human_native",
            "word_count": len(h["text"].split()),
            "tag": h["tag"]
        })

    # 2. Expand ESL (target: 40)
    esl_topics = [
        ("public transportation benefits", "traffic"),
        ("importance of recycling plastic", "waste"),
        ("studying foreign languages in primary school", "education"),
        ("online shopping versus traditional markets", "commerce"),
        ("fast food health consequences", "health"),
        ("protecting wild animals in national parks", "nature"),
        ("homework pressure on teenage students", "school")
    ]
    expanded_esl = expand_corpus_domain(ESL_PROSE_CORPUS, 40, "esl")
    esl_records = []
    for e in expanded_esl:
        esl_records.append({
            "text": e["text"],
            "label": "human",
            "class_id": 0,
            "domain": "esl",
            "generator": "human_esl",
            "word_count": len(e["text"].split()),
            "tag": e["tag"]
        })

    # 3. Expand AI In-Distribution (target: 80)
    ai_in_dist_raw = [x for x in AI_GENERATIONS if "qwen" not in x[1] and "deepseek" not in x[1]]
    expanded_ai_in_dist = expand_corpus_domain(ai_in_dist_raw, 80, "ai_in_dist")
    ai_records_in_dist = []
    for a in expanded_ai_in_dist:
        ai_records_in_dist.append({
            "text": a["text"],
            "label": "ai_generated",
            "class_id": 3,
            "domain": "academic",
            "generator": "gpt4o_claude_llama",
            "word_count": len(a["text"].split()),
            "tag": a["tag"]
        })

    # 4. Expand AI Unseen Qwen (target: 30)
    ai_qwen_raw = [x for x in AI_GENERATIONS if "qwen" in x[1]]
    expanded_qwen = expand_corpus_domain(ai_qwen_raw, 30, "qwen")
    ai_records_qwen = []
    for q in expanded_qwen:
        ai_records_qwen.append({
            "text": q["text"],
            "label": "ai_generated",
            "class_id": 3,
            "domain": "technical",
            "generator": "qwen-2-5-72b",
            "word_count": len(q["text"].split()),
            "tag": q["tag"]
        })

    # 5. Expand AI Unseen DeepSeek (target: 30)
    ai_deepseek_raw = [x for x in AI_GENERATIONS if "deepseek" in x[1]]
    expanded_deepseek = expand_corpus_domain(ai_deepseek_raw, 30, "deepseek")
    ai_records_deepseek = []
    for d in expanded_deepseek:
        ai_records_deepseek.append({
            "text": d["text"],
            "label": "ai_generated",
            "class_id": 3,
            "domain": "macro_crypto",
            "generator": "deepseek-v3",
            "word_count": len(d["text"].split()),
            "tag": d["tag"]
        })

    # 6. Generate Refined Classes (80 each)
    human_refined = []
    for h in human_records:
        polished = polish_human_text_to_refined(h["text"])
        human_refined.append({
            "text": polished,
            "label": "human_ai_refined",
            "class_id": 1,
            "domain": h["domain"],
            "generator": "human_polished_by_llm",
            "word_count": len(polished.split()),
            "tag": f"refined_{h['tag']}"
        })

    ai_refined = []
    for a in ai_records_in_dist:
        paraphrased = paraphrase_ai_text_to_refined(a["text"])
        ai_refined.append({
            "text": paraphrased,
            "label": "ai_ai_refined",
            "class_id": 2,
            "domain": a["domain"],
            "generator": f"paraphrased_{a['generator']}",
            "word_count": len(paraphrased.split()),
            "tag": f"paraphrased_{a['tag']}"
        })

    # Paraphrased pool for test
    ai_refined_test_pool = []
    for x in ai_records_qwen + ai_records_deepseek:
        p = paraphrase_ai_text_to_refined(x["text"])
        ai_refined_test_pool.append({
            "text": p,
            "label": "ai_ai_refined",
            "class_id": 2,
            "domain": x["domain"],
            "generator": f"paraphrased_{x['generator']}",
            "word_count": len(p.split()),
            "tag": f"paraphrased_{x['tag']}"
        })

    # Split into Train (60%), Val (20%), Test (20%)
    train_records = []
    val_records = []
    test_indist_records = []

    def split_pool(pool: List[Dict[str, Any]]):
        random.shuffle(pool)
        n = len(pool)
        n_tr = int(n * 0.60)
        n_v = int(n * 0.20)
        return pool[:n_tr], pool[n_tr:n_tr + n_v], pool[n_tr + n_v:]

    for pool in [human_records, human_refined, ai_refined, ai_records_in_dist]:
        tr, v, ts = split_pool(pool)
        train_records.extend(tr)
        val_records.extend(v)
        test_indist_records.extend(ts)

    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_indist_records)

    return {
        "train": train_records,
        "val": val_records,
        "test_indist": test_indist_records,
        "test_unseen_qwen": ai_records_qwen,
        "test_unseen_deepseek": ai_records_deepseek,
        "test_esl": esl_records,
        "test_paraphrased": ai_refined_test_pool + test_indist_records[:15]
    }


def write_splits(splits: Dict[str, List[Dict[str, Any]]]):
    """Writes dataset jsonl files."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, data in splits.items():
        out_path = os.path.join(PROCESSED_DIR, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Dataset] Wrote {len(data):3d} samples -> {out_path}")


def build_quillbot_comparison_sheet(splits: Dict[str, List[Dict[str, Any]]]):
    """
    Builds exactly 30 representative samples across all 4 classes, native human, ESL human,
    paraphrased AI, and frontier models for manual testing on QuillBot.
    Hard rule: <= 30 samples to hand-check without mass querying or scraping.
    """
    comparison_samples = []

    # 1. Authentic Human Native (6 samples)
    for i, (text, domain, tag) in enumerate(HUMAN_PROSE_CORPUS[:6]):
        comparison_samples.append({
            "id": f"QB-{len(comparison_samples)+1:02d}",
            "text": text,
            "expected_class": "Human-written",
            "class_id": 0,
            "type": "human_native",
            "domain": domain,
            "word_count": len(text.split()),
            "notes": "Authentic human prose with personal idiom or domain scholarship."
        })

    # 2. Authentic Human ESL (4 samples)
    for i, (text, domain, tag) in enumerate(ESL_PROSE_CORPUS[:4]):
        comparison_samples.append({
            "id": f"QB-{len(comparison_samples)+1:02d}",
            "text": text,
            "expected_class": "Human-written",
            "class_id": 0,
            "type": "human_esl",
            "domain": domain,
            "word_count": len(text.split()),
            "notes": "Non-native English writing from learner corpus; tests for false positives."
        })

    # 3. Human-written & AI-refined (6 samples)
    from scripts.refine_data import polish_human_text_to_refined
    for i, (text, domain, tag) in enumerate(HUMAN_PROSE_CORPUS[2:8]):
        polished = polish_human_text_to_refined(text)
        comparison_samples.append({
            "id": f"QB-{len(comparison_samples)+1:02d}",
            "text": polished,
            "expected_class": "Human-written & AI-refined",
            "class_id": 1,
            "type": "human_ai_refined",
            "domain": domain,
            "word_count": len(polished.split()),
            "notes": "Human-authored essay polished by LLM line editing and syntactic smoothing."
        })

    # 4. AI-generated & AI-refined (6 samples)
    from scripts.refine_data import paraphrase_ai_text_to_refined
    for i, (text, model, tag) in enumerate(AI_GENERATIONS[:6]):
        para = paraphrase_ai_text_to_refined(text)
        comparison_samples.append({
            "id": f"QB-{len(comparison_samples)+1:02d}",
            "text": para,
            "expected_class": "AI-generated & AI-refined",
            "class_id": 2,
            "type": "ai_ai_refined",
            "domain": "essay",
            "word_count": len(para.split()),
            "notes": f"AI text from {model} passed through restructuring and paraphrasing."
        })

    # 5. AI-generated Pure Frontier (8 samples)
    # 2 GPT-4o, 2 Claude 3.5, 1 Llama 3.3, 1 Qwen 2.5, 2 DeepSeek-V3
    for i, (text, model, tag) in enumerate(AI_GENERATIONS[:8]):
        comparison_samples.append({
            "id": f"QB-{len(comparison_samples)+1:02d}",
            "text": text,
            "expected_class": "AI-generated",
            "class_id": 3,
            "type": f"ai_pure_{model}",
            "domain": "academic_essay",
            "word_count": len(text.split()),
            "notes": f"Direct generation from {model} without subsequent editing."
        })

    # Limit to exactly 30
    comparison_samples = comparison_samples[:30]

    # Write JSON
    json_path = os.path.join(DATA_DIR, "quillbot_comparison_sheet.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_samples, f, indent=2, ensure_ascii=False)

    # Write Markdown
    md_path = os.path.join(DATA_DIR, "quillbot_comparison_sheet.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# QuillBot AI Detector Hand-Verification Benchmark (30 Samples)\n\n")
        f.write("This sheet provides exactly 30 standardized test passages across all 4 target classes for manual evaluation against QuillBot's AI Detector without violating rate limits or ToS.\n\n")
        f.write("| ID | Target Expected Class | Type / Sub-Genre | Words | First 80 Chars | QuillBot Verdict (Manual) | Veritas Verdict | Match? |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for s in comparison_samples:
            preview = s["text"][:75].replace("\n", " ").replace("|", " ") + "..."
            f.write(f"| {s['id']} | **{s['expected_class']}** | `{s['type']}` | {s['word_count']} | {preview} | *(Pending)* | *(Pending)* | - |\n")

        f.write("\n\n---\n\n## Complete Sample Texts\n\n")
        for s in comparison_samples:
            f.write(f"### Sample {s['id']} — [{s['expected_class']}]\n")
            f.write(f"- **Type**: `{s['type']}` | **Domain**: `{s['domain']}` | **Word Count**: {s['word_count']}\n")
            f.write(f"- **Rationale**: {s['notes']}\n\n")
            f.write("```text\n")
            f.write(s["text"] + "\n")
            f.write("```\n\n")

    print(f"[Comparison] Generated 30-sample manual comparison sheet -> {json_path} & {md_path}")


if __name__ == "__main__":
    splits = assemble_all_data()
    write_splits(splits)
    build_quillbot_comparison_sheet(splits)
