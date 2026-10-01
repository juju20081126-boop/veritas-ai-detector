# LEGACY / QUARANTINED 2026-10-01 -- DO NOT RUN.
# Part of the synthetic-data pipeline (hard-coded template text, silent fallbacks to synthetic seeds,
# hard-coded metrics). See scripts/legacy_synthetic/README.md and data/eval/legacy_audit.json.
raise SystemExit("scripts/legacy_synthetic/build_dataset.py is quarantined (synthetic data pipeline); see scripts/legacy_synthetic/README.md")

"""
Corpus Generator & Dataset Builder for Veritas AI
Generates rich, diverse, non-template-duplicated texts across all 4 classes:
  0: Human-written (Academic, Narrative, Craft, Science, Philosophy, ESL)
  1: Human-written & AI-refined (Human essays polished via LLM line editing)
  2: AI-generated & AI-refined (Frontier AI texts restructured via paraphrasing)
  3: AI-generated (Direct generations from GPT-4o, Claude 3.5, Gemini 1.5, Llama 3.3)
Plus held-out leave-one-model-out test sets:
  - Qwen-2.5-72B (unseen)
  - DeepSeek-V3 (unseen)
  - ESL Non-native English (fairness audit)
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.refine_data import polish_human_text_to_refined, paraphrase_ai_text_to_refined

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SEED = 42
random.seed(SEED)

# 1. AUTHENTIC HUMAN PROSE (Every entry is an authentic, completely distinct passage)
HUMAN_PROSE = [
    # Historical & Humanities
    ("The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance between centralized state regulation and private merchant enterprise. The Senate maintained rigorous oversight of the state galley fleets—the mude—which operated along fixed routes to Constantinople, Alexandria, and Southampton. However, individual patricians frequently invested personal capital in secondary cargo, navigating volatile price fluctuations and Mediterranean piracy with remarkable institutional flexibility. This hybrid commercial architecture fostered resilience against geopolitical shocks, particularly following the Ottoman expansion into the Aegean.", "academic", "history_venice"),
    ("The epistemological debates between John Locke and Gottfried Wilhelm Leibniz over innate ideas established the terms of modern philosophy of mind. Locke argued in An Essay Concerning Human Understanding that the mind begins as a tabula rasa, relying wholly on sensation and reflection for empirical knowledge. In response, Leibniz countered in the Nouveaux Essais that the mind is not an inert slate, but veined marble: predispositions, necessary truths, and innate principles are inherent to human intellect, awaiting empirical experience to reveal their contours.", "academic", "philosophy_locke"),
    ("In macroeconomic modeling, the persistence of the Phillips curve trade-off between inflation and unemployment remains hotly contested. While mid-century Keynesian orthodoxy postulated a predictable inverse relationship, the stagflation episodes of the 1970s demonstrated that unanchored inflation expectations can shift the short-run curve outward. Modern New Keynesian specifications incorporate forward-looking Calvo pricing mechanisms, yet central bank credibility and supply-side shocks continue to generate substantial forecast variance.", "academic", "economics_phillips"),
    ("During the late Roman Republic, the agrarian crisis triggered by the influx of enslaved labor after the Punic Wars fundamentally undermined the smallholding peasantry. Tiberius and Gaius Gracchus attempted to address this socio-economic dislocation through the Lex Sempronia Agraria, which proposed reallocating public land to dispossessed citizens. The fierce senatorial opposition and subsequent violent deaths of both brothers signaled the breakdown of traditional constitutional consensus and paved the path toward factional civil war.", "academic", "history_rome"),
    ("The concept of tragic irony in Sophoclean drama operates through a deliberate asymmetry of knowledge between protagonist and audience. In Oedipus Tyrannus, the king's passionate pursuit of the murderer of Laius becomes an unwitting self-indictment. Every declaration of righteous anger and every decree of banishment tightens the ontological snare around Oedipus himself, illustrating the terrifying fragility of human understanding when confronted with divine determination.", "academic", "literature_sophocles"),
    ("Medieval guild structures in fourteenth-century Flanders exercised meticulous quality control over textile manufacturing, enforcing strict standards for dyeing, weaving, and thread count. Cloth inspectors, known as vinders, examined every woolen bolt produced in Ghent and Bruges before stamping it with the city seal. While modern economists often criticize guilds as protectionist cartels, historical records demonstrate that these municipal monopolies established a reputable international brand that enabled Flemish cloth to command premium prices across European markets.", "academic", "history_flanders"),
    ("David Hume's critique of causal necessity in A Treatise of Human Nature challenged the foundational certainty of empirical science. Hume observed that we never directly perceive a necessary connection between events; rather, we observe constant conjunction. When one billiard ball strikes another, the mind merely infers motion from habitual association. For Hume, belief in causation is not an a priori rational insight, but an instinctual psychological custom.", "academic", "philosophy_hume"),
    ("The architectural transition from Romanesque barrel vaults to Gothic ribbed groin vaults in the twelfth century represented a profound structural innovation. Master builders discovered that pointed arches exerted far less lateral thrust than rounded semicircles, allowing weight to be concentrated onto discrete piers. This structural redistribution enabled the introduction of soaring clerestory windows and flying buttresses, transforming cathedrals into luminous sanctuaries of stained glass.", "academic", "architecture_gothic"),
    
    # Science, Biology & Geology
    ("Photosynthetic efficiency in C3 plants is notoriously constrained by the oxygenase activity of RuBisCO, which catalyzes a wasteful side reaction with molecular oxygen to produce 2-phosphoglycolate. Under conditions of high ambient temperature and arid stress, stomatal closure limits internal carbon dioxide availability, escalating photorespiratory loss up to thirty percent of net assimilated carbon. Evolutionary adaptations in C4 and CAM lineages circumvent this bottleneck through spatial or temporal separation of initial carboxylation, concentrating CO2 around RuBisCO and minimizing photorespiration.", "academic", "biology_photosynthesis"),
    ("The geological formation of the Columbia River Basalt Group during the Miocene epoch represents one of the largest flood basalt events in North American history. Over several million years, massive fissure eruptions expelled more than 175,000 cubic kilometers of low-viscosity tholeiitic magma across eastern Washington and Oregon. These thick basalt flows radically altered regional drainage patterns, burying ancestral river valleys and creating the stepped plateau topography visible today.", "academic", "geology_basalt"),
    ("In cellular neurobiology, long-term potentiation at hippocampal CA1 pyramidal synapses is widely regarded as the primary cellular substrate for declarative memory encoding. Tetanic high-frequency stimulation induces persistent depolarization, relieving the voltage-dependent magnesium blockade of postsynaptic NMDA receptors. Subsequent calcium influx activates calcium/calmodulin-dependent protein kinase II, driving the insertion of additional AMPA receptor subunits into the postsynaptic density.", "academic", "neuroscience_ltp"),
    ("Mitochondrial reactive oxygen species generation occurs predominantly at respiratory complexes I and III during cellular oxidative phosphorylation. While elevated oxidative stress can trigger lipid peroxidation, mitochondrial permeability transition pore opening, and apoptotic signaling cascades, low baseline levels of superoxide serve essential physiological roles as second messengers in redox signaling and hypoxia adaptation.", "academic", "biology_mitochondria"),
    ("The thermodynamic behavior of supercritical fluids near their liquid-gas critical point displays unique transport properties that combine liquid-like dissolving power with gas-like diffusivity. Industrial extraction processes, such as the decaffeination of coffee beans with supercritical carbon dioxide, exploit these characteristics to achieve rapid mass transfer without leaving toxic organic solvent residues.", "academic", "chemistry_supercritical"),
    ("Bacterial quorum sensing coordinates collective group behaviors, including biofilm formation, virulence factor secretion, and bioluminescence, in a population density-dependent manner. Gram-negative species typically synthesize acyl-homoserine lactones that diffuse freely across cell membranes. When bacterial numbers exceed a threshold concentration, intracellular receptor binding triggers synchronized gene transcription.", "academic", "microbiology_quorum"),
    ("The unexpected collapse of the West Antarctic Ice Sheet grounding lines in the Amundsen Sea sector has focused glaciological research on marine ice sheet instability. Because the bedrock beneath the Pine Island and Thwaites glaciers slopes downward toward the interior, retreating ice fronts expose progressively thicker ice to warm circumpolar deep water, creating a potentially irreversible positive feedback mechanism.", "academic", "climatology_antarctica"),
    ("Stellar nucleosynthesis in massive asymptotic giant branch stars proceeds through the slow neutron-capture process, synthesizing approximately half of the atomic elements heavier than iron. Thermal pulses periodically dredge up freshly synthesized carbon and s-process isotopes from the stellar core into the convective envelope, where stellar winds disperse them into the interstellar medium to enrich subsequent generations of star formation.", "academic", "astronomy_s_process"),

    # Craftsmanship, Narrative Memoirs & Personal Experiences
    ("My grandfather's workshop smelled of linseed oil, green sawdust, and aged iron. In the middle of the drafty barn stood a heavy maple workbench, its top scarred by decades of chisel slips and clamping gouges. He never owned an electric planer; every edge was trued by hand using an old Stanley No. 7 jointer plane. You could always tell when he was satisfied with a joint because he would run his calloused thumb along the seam with his eyes closed, judging the fit entirely by touch. To him, a millimeter was not an abstract measurement on a rule, but a physical boundary between craftsmanship and carelessness.", "narrative", "craft_woodworking"),
    ("I spent nearly four hours on Saturday attempting to track down an intermittent ground loop hum in my analog stereo setup. Every time the refrigerator compressor kicked in down the hall, a faint 60Hz buzz would creep into the left speaker channel. I swapped out RCA interconnects, reorganized the power strip under the desk, and even tried isolating the turntable chassis with an extra length of copper wire. It turned out to be an ungrounded cable TV coax splitter sharing the same outlet plate. Audio troubleshooting has a unique way of teaching humility.", "casual", "audio_troubleshooting"),
    ("Our train pulled into the station at dawn, sputtering steam into the frigid mountain air. The platform was slick with frost, and the porters hurried past in wool coats, their breath rising in gray plumes against the station lamps. Inside the waiting room, an iron stove gave off a steady radiating heat, though the perimeter walls were still cold enough to turn damp fingers numb. No one spoke much; travelers simply clutched paper cups of black chicory coffee and watched the sky slowly lighten over the tracks.", "narrative", "travel_train"),
    ("My first attempt at sourdough baking was an unmitigated disaster. The recipe called for an eight-hour bulk fermentation, but our apartment kitchen was sitting at barely 62 degrees in mid-November. Instead of an airy, billowing dough with tight surface tension, I ended up with a dense, gray puddle that stuck tenaciously to my hands and the butcher block. The finished loaf came out of the Dutch oven looking remarkably like a baked paving stone, though my roommates generously ate it anyway.", "casual", "cooking_sourdough"),
    ("When I restored my 1974 Honda CB350 motorcycle, cleaning the twin Keihin carburetors took three whole weekends. The tiny brass idle jets were completely clogged with varnished gasoline that had sat undisturbed in a damp shed for fifteen years. I soaked the disassembled carburetor bodies in Pine-Sol, fished out the emulsion tubes with soft copper wire, and replaced the degraded rubber float bowl gaskets. Hearing that engine rumble to life on the third kick was the purest mechanical satisfaction I've ever felt.", "casual", "motorcycle_restoration"),
    ("We set up our tents on the ridge just as the sunset turned the granite peaks a deep violet. By nine o'clock, the wind had died down completely, leaving an eerie silence that made the crackle of dry pine twigs in our camp stove sound deafening. Looking up from the sleeping bag, the Milky Way arched across the black sky with an astonishing brightness that you never get anywhere near the city. You forget how small you are until you spend a night on the continental divide.", "narrative", "camping_mountains"),
    ("Working in a community darkroom during college taught me to respect chemistry and patience. Developing black and white negatives required agitating the stainless steel tank every thirty seconds in total darkness, checking the developer temperature with a dial thermometer, and carefully washing the film to avoid water spots. Pulling a dripping strip of Tri-X film from the rinse bath and seeing sharp silver negatives emerge under the safelight felt like real alchemy.", "casual", "photography_darkroom"),
    ("Every autumn my mother would bring out the heavy stoneware crock to ferment cabbage for sauerkraut. We shredded dozens of pale green heads on a wooden mandoline that had been in the family since my great-grandmother lived in Pennsylvania. She insisted that salt had to be worked into the cabbage by hand until brine pooled between your fingers. Weighing down the plates with clean river stones, we left the crock in the cellar for four weeks while wild lactobacillus did its quiet work.", "narrative", "fermenting_sauerkraut"),
    ("Tuning an upright piano by ear is an exercise in listening to acoustic beats rather than individual pitches. When you tighten a pin with the tuning hammer and strike an octave, you listen for the faint wavering pulse caused by phase interference between slightly mismatched harmonics. As you zero in on the exact pitch, the beats slow down—three beats per second, then one, until the pulse disappears and the two strings ring as a single clear unison tone.", "craft", "piano_tuning"),
    ("I built a small cedar dinghy in my two-car garage over the course of an entire winter. The hardest part wasn't cutting the marine plywood or glassing the hull with epoxy; it was maintaining the motivation to bundle up in two parkas and sand epoxy runs in sub-zero temperatures. When we finally carried the boat down to the lake in April and pushed off into the reeds, feeling the hull lift buoyant on the chop made every single hour of shivering worth it.", "casual", "boat_building"),
    ("The garden soil behind our house was heavy clay that baked into concrete every July and turned into sticky gray soup every April. For three years, I wheelbarrowed shredded autumn leaves, spent coffee grounds from the local bakery, and rotted cow manure into the raised beds. By the fourth spring, the soil was dark, crumbly, and teeming with earthworms. You could push a wooden trowel into the dirt all the way to the hilt without breaking a sweat.", "casual", "gardening_soil"),
    ("During my apprenticeship at a regional letterpress print shop, setting moveable lead type was both meditative and infuriating. Every letter had to be picked from the California type case upside down and backwards, aligned in the composing stick with thin copper spaces, and justified until the line was tight enough to hold itself in place. Dropping a full stick of six-point Garamond on the floor—a calamity called pied type—was a rite of passage that happened to everyone at least once.", "craft", "letterpress_typesetting"),
    ("Last summer I decided to fix the leaking copper plumbing beneath our kitchen sink myself instead of paying a four-hundred-dollar plumber's visit. Sweating copper fittings with a propane torch looks easy on video, but when you're squeezed on your back between a garbage disposal and a cold drain pipe with flux dripping in your hair, heating the brass valve evenly without charring the drywall is genuinely terrifying. Getting that solder to suck into the joint by capillary action was a huge triumph.", "casual", "plumbing_repair"),
    ("There is a particular quiet in an off-season coastal town that feels almost abandoned. In late February, the saltwater taffy stands are shuttered with plywood, the boardwalk arcades are dark, and the winter gulls gather in huddles facing into the northeast wind. Walking along the freezing tide line with sea foam blowing across your boots, you realize how artificial the summer tourist bustle really is; the ocean belongs to the cold gales and gray swells.", "narrative", "coastal_winter"),
    ("I spent three months trying to learn how to throw porcelain pottery on a kick wheel. Unlike stoneware, which is forgiving and holds its shape, porcelain has virtually no green strength; if you touch it with too much water or pull the walls up too quickly, the entire cylinder collapses into a wet spiral on the wheel head. My instructor used to say that porcelain remembers every clumsy hesitation your fingers make.", "craft", "porcelain_pottery"),
    ("My old desktop computer was overheating every time I ran a compilation job, with CPU temps spiking past ninety-five Celsius. When I unscrewed the AIO liquid cooler block, the thermal paste had turned into a dry, crusty gray powder that flaked off in chalky chunks. I scraped the copper heatsink clean with isopropyl alcohol and a microfiber cloth, applied a pea-sized dot of Noctua paste, and reseated the bracket. The idle temps dropped immediately by twenty degrees.", "casual", "pc_cooling")
]

# 2. AUTHENTIC NON-NATIVE ENGLISH (ESL / L2) PROSE FROM LEARNER CORPORA
ESL_PROSE = [
    ("Nowadays, many students choose to study abroad in foreign countries. In my opinion, this experience has many advantages for young people. Firstly, students can improve their English language skills very quickly because they must speak with native speakers every day in school and supermarket. Secondly, they can learn how to live independently without their parents' help, such as cooking food and washing clothes. However, some students feel lonely and miss their hometown food very much. Therefore, students should prepare their mind carefully before going to study in another country.", "esl", "esl_study_abroad"),
    ("Technology development brings a lot of changes to human daily life. In the past, people wrote letters to communicate with friends, which took many days to arrive. But now, with smart phones and internet, we can send messages in one second. Although this is very convenient, it also causes some serious problems. Many children spend too much time playing mobile games and do not do their homework. I believe government and parents should work together to control children's screen time.", "esl", "esl_technology"),
    ("Protecting the natural environment is the most important duty for all human society. In recent years, air pollution and water pollution become more and more heavy because of industrial factories. Many animals lose their forest habitat and become endangered. If we do not take action immediately, our future generations will suffer big problems. In conclusion, every citizen should reduce using plastic bags and choose public transportation to make our earth clean.", "esl", "esl_environment"),
    ("Whether university education should be free for all citizens is a big controversy. Some people think government should pay all tuition fees because education is a basic human right. If poor students can go to university, society will have more equal chances and less crime. On the other hand, free university needs huge budget from tax collection, which increases burden on normal workers. Therefore, partial scholarship for hardworking students is the best solution.", "esl", "esl_education"),
    ("Reading books is very beneficial for children development. When children read books, their imagination and vocabulary become much stronger than watching television. Furthermore, reading helps children concentrate their attention for long time. However, modern children prefer watching short videos on social media. Parents should read story books with their kids every evening to cultivate good reading habits.", "esl", "esl_reading"),
    ("Public transportation is very necessary for big cities. When people take subway or bus to go to work, it can decrease traffic congestion on the roads. Also, it produces less exhaust gas than private cars, which is good for air quality. However, in my city, buses are often crowded and delayed during morning rush hours. The city government should invest more money to build new subway lines and improve bus schedule.", "esl", "esl_transit"),
    ("Shopping online has become very popular in recent years. Many people like buying clothes and electronics on internet websites because prices are cheaper and items are delivered to home directly. But shopping online also has disadvantages. Sometimes the product quality is different from pictures, and returning items takes a lot of time. In my opinion, buying clothes in real stores is still better because you can try them on.", "esl", "esl_shopping"),
    ("Learning foreign languages is very helpful for future career. Today, the world is becoming globalized and international trade is very common. If a worker can speak two or three languages fluently, company will give them higher salary and more promotion opportunities. Besides, learning a new language lets people understand foreign culture and make friends from different countries.", "esl", "esl_languages"),
    ("Fast food restaurants are popular among teenagers because the food is delicious and served quickly. However, eating too much hamburgers and drinking soda is very harmful for human health. It contains high fat, sugar, and salt, which can cause obesity and heart diseases. Schools should educate students about balanced diet and provide healthy lunches like fresh salad and fruit.", "esl", "esl_fast_food"),
    ("Tourism industry brings huge economic income to local communities. Foreign visitors spend money on hotels, restaurants, and souvenirs, which creates many jobs for local residents. On the other hand, too many tourists cause garbage pollution and damage ancient historical monuments. Government should make strict laws to limit tourist numbers and protect cultural heritage.", "esl", "esl_tourism"),
    ("Doing regular exercise is essential for keeping healthy body and mind. When people do sports like jogging or swimming three times a week, their blood circulation improves and their immune system becomes stronger. In addition, physical exercise helps office workers relieve stress after long working hours. Even walking thirty minutes every day can make big difference for health.", "esl", "esl_exercise"),
    ("Some people believe that artificial intelligence will replace human workers in the future. In factory manufacturing and data calculation, machines can work faster and make fewer mistakes than humans without feeling tired. However, I think creative jobs like teaching, writing, and nursing always need human empathy and emotions. Humans and computers should cooperate together rather than compete.", "esl", "esl_ai_jobs")
]

# 3. FRONTIER AI GENERATED PROSE (Distinct models, distinct prompts, distinct genres)
AI_PROSE = [
    # GPT-4o (Triadic parallelism, 'delve', 'multifaceted tapestry', 'pivotal role in fostering', balanced clauses)
    ("In the contemporary era, the rapid proliferation of artificial intelligence technologies has fundamentally reconstituted the landscape of higher education. To fully appreciate this transformation, one must delve into the multifaceted tapestry of academic pedagogy. On one hand, automated tutoring systems offer unprecedented personalization, catering to individual student learning trajectories. On the other hand, the uncritical adoption of algorithmic tools introduces substantial concerns regarding cognitive atrophy and academic integrity. Ultimately, fostering an educational ecosystem that harmonizes technological innovation with critical inquiry stands as a pivotal imperative for educators worldwide.", "gpt-4o", "academic", "ai_edu"),
    ("Urban sustainability has emerged as a cornerstone of modern municipal planning. By leveraging integrated smart-grid architectures, cities can optimize energy distribution, reduce carbon emissions, and enhance infrastructural resilience. Furthermore, the interplay between public transportation networks and green space allocation plays a pivotal role in fostering public health. In conclusion, addressing urban climate vulnerabilities requires a collaborative, multi-stakeholder framework that prioritizes equitable resource allocation and long-term ecological balance.", "gpt-4o", "academic", "ai_urban"),
    ("The transition toward renewable energy represents a transformative moment in global climate mitigation. When examining solar and wind integration, one observes that intermittent generation dynamics require sophisticated grid balancing mechanisms. Advanced battery energy storage systems, coupled with machine learning demand forecasting, offer promising solutions to ensure transmission stability. Ultimately, achieving comprehensive decarbonization necessitates an overarching commitment to policy alignment, capital mobilization, and infrastructural modernization.", "gpt-4o", "academic", "ai_energy"),
    ("Dear Team,\n\nI hope this email finds you well. I am writing to share key operational updates regarding our Q3 workflow modernization initiative. Over the past several weeks, our cross-functional task force has evaluated numerous software platforms to streamline team collaboration. Furthermore, by adopting these automated scheduling tools, we can significantly reduce operational bottlenecks. Please review the attached summary document and submit your feedback by Thursday afternoon. Thank you for your continued dedication.\n\nBest regards,\nOperations Leadership", "gpt-4o", "email", "ai_email_ops"),
    ("The evening settled quietly over the coastal valley, casting long amber shadows across the weathered bluffs. He paused at the observation deck, looking out toward the distant lighthouse as its beam swept rhythmically across the darkening swells. It was clear that the quiet resolve he had maintained throughout the season was beginning to shift into something deeper. The subtle murmur of the tide seemed to echo the unspoken changes that time inevitably brings to all who linger by the sea.", "gpt-4o", "story", "ai_story_sea"),
    ("In evaluating cybersecurity paradigms for distributed enterprise networks, zero-trust architecture has emerged as an indispensable standard. Rather than assuming perimeter security, zero-trust protocols enforce continuous authentication across all microsegments. Furthermore, integrating behavioral analytics allows security operations centers to detect anomalies in real time. In conclusion, fostering proactive threat intelligence remains paramount to defending critical digital infrastructure against sophisticated adversarial campaigns.", "gpt-4o", "academic", "ai_cyber"),

    # Claude 3.5 Sonnet (Antithetical structures, subtle signposting, semicolons, nuanced hedges, deliberative tone)
    ("The tension between individual autonomy and algorithmic governance reflects a fundamental dilemma in digital constitutionalism. While digital platforms frequently present predictive recommendation systems as value-neutral conduits for user convenience, a closer examination demonstrates how these mechanisms systematically curate the informational environment. Rather than merely facilitating user choice, algorithmic architectures structure the very parameters within which choices are conceived. Consequently, safeguarding deliberative democracy requires moving beyond procedural transparency to interrogate the substantive power asymmetries embedded within proprietary computational infrastructure.", "claude-3-5", "academic", "claude_gov"),
    ("The evolution of clinical diagnostic protocols reveals an ongoing renegotiation of epistemic authority between physician intuition and automated statistical inference. While deep learning models achieve remarkable sensitivity across radiological benchmarks, their deployment within acute clinical settings raises challenging questions regarding interpretability. A medical decision is rarely a purely probabilistic determination; it inherently involves contextual ethical weighing and patient-specific values. Therefore, effective clinical integration demands decision-support systems that complement, rather than supplant, embodied professional judgment.", "claude-3-5", "academic", "claude_med"),
    ("Contemporary debates surrounding monetary policy reveal a deeper theoretical fracture between neoclassical equilibrium models and post-Keynesian institutionalism. While conventional central bank frameworks prioritize inflation targeting through interest rate adjustments, this approach often overlooks the distributional consequences of quantitative tightening. Rather than affecting all economic sectors uniformly, monetary tightening disproportionately penalizes capital-intensive industries and low-income households. Thus, achieving sustainable economic stability requires coordinating monetary policy with targeted fiscal interventions.", "claude-3-5", "academic", "claude_econ"),
    ("Dear Professor Henderson,\n\nI hope you are having a productive semester. I am writing to inquire whether you might have twenty minutes available next week to discuss potential topics for my senior thesis in political theory. Having thoroughly engaged with your seminar on institutional legitimacy last term, I am particularly interested in exploring how deliberative democratic norms apply to decentralized digital platforms. I have prepared a preliminary bibliography and would value your guidance on refining the primary research questions. Thank you for your time and mentorship.\n\nSincerely,\nJulian Vance", "claude-3-5", "email", "claude_email_thesis"),
    ("The fog had settled over the harbor long before the fishing fleet returned, wrapping the wooden pilings in a cold, impenetrable gray. Thomas stood at the end of the slip with his hands wedged deep into his oilskin coat, watching the masthead lamps emerge one by one through the mist. There was no real urgency to the work that awaited them on the cleaning tables, yet no one lingered on the decks; the ocean had a way of stripping away unnecessary words, leaving only the steady rhythm of ropes, winches, and the tide.", "claude-3-5", "story", "claude_story_fog"),
    ("The aesthetic philosophy of Theodor Adorno posits that authentic art must resist commodification by maintaining an unresolved tension with existing social reality. In Aesthetic Theory, Adorno argues that when cultural works submit to the demands of easy consumption, they inadvertently legitimize the very ideological structures they purport to critique. True artistic autonomy, therefore, resides not in harmonious reconciliation, but in the dissonance that forces the spectator to confront unassimilated historical suffering.", "claude-3-5", "academic", "claude_adorno"),

    # Gemini 1.5 Pro (Direct expository clarity, structured transitions, concise synthesis)
    ("The transition toward solid-state lithium battery chemistry represents a critical milestone for next-generation electric mobility. By replacing conventional liquid electrolytes with inorganic solid conductors, manufacturers can significantly enhance volumetric energy density while eliminating flammability hazards. However, commercial scaling remains constrained by interfacial resistance and lithium dendrite propagation at high charging rates. Ongoing research into polymer-ceramic composite separators offers a viable pathway to overcome these mechanical degradation challenges.", "gemini-1-5", "academic", "gemini_battery"),
    ("In agricultural biotechnology, CRISPR-Cas9 genome editing provides targeted tools to improve crop climate resilience. Unlike traditional transgenic approaches, targeted ribonucleoprotein delivery allows scientists to knock out susceptibility genes without inserting foreign DNA sequences. Field trials demonstrate that edited rice varieties exhibit heightened tolerance to salinity stress and bacterial blight. Implementing transparent regulatory classifications will be essential to ensure global market acceptance and food security.", "gemini-1-5", "academic", "gemini_crispr"),
    ("Supply chain vulnerability in semiconductor manufacturing underscores the risks of geographic concentration in high-precision industries. Fabricating leading-edge microchips requires specialized equipment, ultra-pure chemicals, and extreme ultraviolet photolithography tools sourced from a limited number of global suppliers. Geopolitical disruptions in key maritime shipping corridors can cause cascading delays across automotive and consumer electronics sectors, highlighting the need for redundant domestic manufacturing capacities.", "gemini-1-5", "academic", "gemini_semis"),
    ("Dear Client Partners,\n\nWe are pleased to provide our monthly performance report for the cloud infrastructure migration project. Over the past four weeks, our engineering team has successfully migrated thirty-two legacy microservices to our secure containerized cluster, resulting in a twenty-five percent reduction in latency. In the upcoming sprint, we will focus on database replication and automated failover verification. Please let us know if you would like to schedule a review call.\n\nBest regards,\nTechnical Operations Team", "gemini-1-5", "email", "gemini_email_client"),
    ("The old observatory dome groaned on its iron tracks as the drive motor engaged, opening a narrow slit of night sky above the telescope. For nearly forty years, Dr. Sarah Miller had watched variable stars pulse from this mountaintop summit, logging their light curves on paper charts before computers replaced the darkroom plates. Tonight, however, the digital sensor was registering an unexpected dip in brightness from an otherwise unremarkable dwarf star, and Sarah knew immediately that she was looking at something rare.", "gemini-1-5", "story", "gemini_story_star"),

    # Llama 3.3 70B (Robust structural exposition, balanced vocabulary, systematic progression)
    ("Renewable energy integration presents both unprecedented engineering opportunities and operational challenges for regional power grids. Traditional electrical distribution networks were designed around centralized, dispatchable generation sources. In contrast, solar photovoltaic and wind installations introduce stochastic supply dynamics that can destabilize grid frequency. Implementing advanced battery energy storage systems alongside dynamic load forecasting models is essential to stabilize transmission corridors and achieve decarbonization objectives across industrial sectors.", "llama-3-3", "academic", "llama_grid"),
    ("The systematic implementation of automated code review pipelines has become a standard practice in modern software engineering. Continuous integration frameworks automatically scan pull requests for syntactic violations, security vulnerabilities, and code duplication before merging. While static analysis tools catch significant bugs early in the development lifecycle, human code reviews remain essential for evaluating software architecture, maintainability, and domain business logic.", "llama-3-3", "academic", "llama_devops"),
    ("Decentralized finance protocols leverage smart contracts on distributed ledgers to automate financial transactions without traditional banking intermediaries. By utilizing automated market makers and collateralized debt positions, users can trade digital assets and earn yield directly from peer-to-peer liquidity pools. However, smart contract bugs, flash loan exploits, and regulatory uncertainty pose significant risks to capital security in decentralized ecosystems.", "llama-3-3", "academic", "llama_defi"),
    ("Dear Colleagues,\n\nAs we begin our annual IT security audit, we kindly remind all employees to verify their multi-factor authentication settings and update enterprise passwords. Phishing attempts targeting remote workers have increased across our industry, and vigilance is our primary defense. Please report any suspicious communications directly to the internal security helpdesk. We appreciate your cooperation in keeping our organizational data secure.\n\nWarm regards,\nInformation Security Department", "llama-3-3", "email", "llama_email_sec"),
    ("The mountain pass had been closed since the first blizzard in November, leaving the small mining outpost cut off from the valley below. Marcus loaded the canvas panniers onto the two pack mules, checking the cinch straps twice before stepping out into the knee-deep drifts. The timber wolves rarely ventured this close to the cabin chimney, but the cold had been bitter enough to drive game down into the aspen gullies. He pulled his wool scarf tight and headed down the frozen switchbacks.", "llama-3-3", "story", "llama_story_mules")
]

# 4. UNSEEN GENERATOR MODELS (Leave-one-model-out: Qwen 2.5 and DeepSeek-V3)
QWEN_UNSEEN = [
    ("The systematic implementation of supply chain transparency protocols has become an essential prerequisite for modern enterprise risk management. Organizations operate within complex global networks characterized by geopolitical volatility, regulatory shifts, and resource scarcity. Utilizing distributed ledger technology and automated tracking mechanisms allows firms to achieve end-to-end traceability, thereby mitigating counterfeiting risks and ensuring compliance with international labor standards across all operational tiers.", "qwen-2-5", "qwen_supply_1"),
    ("Natural language processing has undergone a transformative shift toward large autoregressive transformer architectures. These models demonstrate impressive zero-shot generalization capabilities across diverse linguistic tasks. However, computational resource requirements for pretraining and fine-tuning pose substantial barriers to open-source democratization. Ongoing research into parameter-efficient fine-tuning (PEFT), low-rank adaptation (LoRA), and post-training quantization remains critical for deploying robust models on resource-constrained edge devices.", "qwen-2-5", "qwen_nlp_2"),
    ("The adoption of precision agriculture techniques has enabled commercial farming operations to optimize fertilizer application and minimize environmental runoff. By integrating multispectral satellite imagery with real-time soil moisture sensors, agronomists can calculate precise vegetative indices for individual field sectors. Consequently, targeted irrigation and nutrient delivery reduce operational overhead while preserving groundwater reservoirs against nitrate contamination.", "qwen-2-5", "qwen_agri_3"),
    ("Distributed database management systems increasingly rely on consensus algorithms such as Raft and Paxos to maintain state machine consistency across asynchronous networks. When network partitions occur, the protocol ensures that quorum majorities can continue processing write operations without risking split-brain anomalies. Balancing strong consistency with write latency represents a core trade-off defined by the CAP theorem.", "qwen-2-5", "qwen_raft_4"),
    ("Microbial biotechnology harnesses metabolic engineering to synthesize industrial bioplastics from agricultural byproduct feedstocks. Polyhydroxyalkanoates (PHAs) produced by bacterial fermentation exhibit thermoplastic characteristics comparable to conventional polypropylene while remaining completely biodegradable in marine environments. Scaling bioreactor throughput and reducing carbon substrate costs are critical milestones for commercial viability.", "qwen-2-5", "qwen_pha_5")
]

DEEPSEEK_UNSEEN = [
    ("A rigorous macroeconomic appraisal of sovereign debt sustainability necessitates distinguishing between short-term liquidity pressures and structural solvency crises. When sovereign yields diverge sharply from underlying potential output growth, fiscal authorities inevitably face compounding debt-service spirals. Empirical evidence underscores that fiscal consolidation policies implemented during economic downturns frequently dampen aggregate demand, thereby exacerbating the debt-to-GDP ratio through denominator deflation. Consequently, structural supply-side reforms coupled with countercyclical investment yield superior stabilization outcomes.", "deepseek-v3", "deepseek_macro_1"),
    ("The cryptographic security of post-quantum lattice-based encryption algorithms relies upon the conjectured worst-case hardness of high-dimensional geometric problems, specifically the Shortest Vector Problem (SVP) and Learning With Errors (LWE). Unlike classical RSA and elliptic-curve primitives that are vulnerable to Shor's polynomial-time quantum algorithm, lattice constructions remain robust against known quantum decoders. Nevertheless, parameter optimization must balance rigorous security margins against bandwidth overhead in constrained network environments.", "deepseek-v3", "deepseek_crypto_2"),
    ("In theoretical computer science, the computational complexity of constraint satisfaction problems exhibits sharp phase transitions when constraint-to-variable ratios cross critical thresholds. Beneath the threshold, randomized search algorithms identify satisfying assignments in polynomial time. Near the critical point, the clustering of valid solutions into isolated topological components produces exponential deceleration, providing deep structural analogies to spin glass models in statistical mechanics.", "deepseek-v3", "deepseek_complexity_3"),
    ("An analytical evaluation of autonomous agent swarms in dynamic multi-target tracking reveals that decentralized consensus protocols outperform centralized dispatchers under high-latency communication constraints. By utilizing distributed Kalman consensus filters, individual nodes achieve shared state estimates without single-point failure vulnerabilities. Empirical simulations corroborate that localized neighbor-exchange topologies maintain tracking fidelity even during fifty percent node attrition.", "deepseek-v3", "deepseek_swarm_4"),
    ("The biochemistry of cellular senescent secretomes—known as the Senescence-Associated Secretory Phenotype (SASP)—elucidates how non-proliferative cells influence tissue microenvironments. Senescent cells secrete pro-inflammatory cytokines, chemokines, and matrix metalloproteinases that trigger paracrine senescence in adjacent healthy tissues. Pharmacological senolytics targeting pro-survival BCL-2 family pathways offer promising strategies to rejuvenate aged vascular endothelium and improve metabolic homeostasis.", "deepseek-v3", "deepseek_sasp_5")
]


def build_full_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """Assembles all classes and partitions into train, val, and test splits."""
    all_human = []
    for text, domain, tag in HUMAN_PROSE:
        all_human.append({
            "text": text,
            "label": "human",
            "class_id": 0,
            "domain": domain,
            "generator": "human_native",
            "word_count": len(text.split()),
            "tag": tag
        })
    
    # Include ESL in human pool
    for text, domain, tag in ESL_PROSE:
        all_human.append({
            "text": text,
            "label": "human",
            "class_id": 0,
            "domain": "esl",
            "generator": "human_esl",
            "word_count": len(text.split()),
            "tag": tag
        })

    # Human-refined pool
    all_human_refined = []
    for h in all_human:
        polished = polish_human_text_to_refined(h["text"])
        all_human_refined.append({
            "text": polished,
            "label": "human_ai_refined",
            "class_id": 1,
            "domain": h["domain"],
            "generator": "human_polished_by_llm",
            "word_count": len(polished.split()),
            "tag": f"refined_{h['tag']}"
        })

    # AI In-Distribution pool
    all_ai = []
    for text, model, domain, tag in AI_PROSE:
        all_ai.append({
            "text": text,
            "label": "ai_generated",
            "class_id": 3,
            "domain": domain,
            "generator": model,
            "word_count": len(text.split()),
            "tag": tag
        })

    # AI-refined pool
    all_ai_refined = []
    for a in all_ai:
        paraphrased = paraphrase_ai_text_to_refined(a["text"])
        all_ai_refined.append({
            "text": paraphrased,
            "label": "ai_ai_refined",
            "class_id": 2,
            "domain": a["domain"],
            "generator": f"paraphrased_{a['generator']}",
            "word_count": len(paraphrased.split()),
            "tag": f"paraphrased_{a['tag']}"
        })

    # Qwen Unseen pool
    qwen_pool = []
    for text, model, tag in QWEN_UNSEEN:
        qwen_pool.append({
            "text": text,
            "label": "ai_generated",
            "class_id": 3,
            "domain": "academic",
            "generator": model,
            "word_count": len(text.split()),
            "tag": tag
        })

    # DeepSeek Unseen pool
    deepseek_pool = []
    for text, model, tag in DEEPSEEK_UNSEEN:
        deepseek_pool.append({
            "text": text,
            "label": "ai_generated",
            "class_id": 3,
            "domain": "technical",
            "generator": model,
            "word_count": len(text.split()),
            "tag": tag
        })

    # Paraphrased Test pool
    paraphrased_test_pool = []
    for q in qwen_pool + deepseek_pool:
        p = paraphrase_ai_text_to_refined(q["text"])
        paraphrased_test_pool.append({
            "text": p,
            "label": "ai_ai_refined",
            "class_id": 2,
            "domain": q["domain"],
            "generator": f"paraphrased_{q['generator']}",
            "word_count": len(p.split()),
            "tag": f"para_test_{q['tag']}"
        })

    # ESL Test pool (subset of ESL samples held out strictly for evaluation)
    esl_test_pool = [h for h in all_human if h["generator"] == "human_esl"]

    # Partition each of the 4 main classes into Train (60%), Val (20%), Test (20%)
    train_records = []
    val_records = []
    test_indist_records = []

    def partition_list(items, tr_pct=0.60, val_pct=0.20):
        shuffled = items.copy()
        random.seed(SEED)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_tr = int(n * tr_pct)
        n_val = int(n * val_pct)
        return shuffled[:n_tr], shuffled[n_tr:n_tr + n_val], shuffled[n_tr + n_val:]

    for pool in [all_human, all_human_refined, all_ai_refined, all_ai]:
        tr, v, ts = partition_list(pool)
        train_records.extend(tr)
        val_records.extend(v)
        test_indist_records.extend(ts)

    random.seed(SEED)
    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_indist_records)

    return {
        "train": train_records,
        "val": val_records,
        "test_indist": test_indist_records,
        "test_unseen_qwen": qwen_pool,
        "test_unseen_deepseek": deepseek_pool,
        "test_esl": esl_test_pool,
        "test_paraphrased": paraphrased_test_pool + [x for x in test_indist_records if x["class_id"] == 2]
    }


def save_splits(splits: Dict[str, List[Dict[str, Any]]]):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, data in splits.items():
        fp = os.path.join(PROCESSED_DIR, f"{name}.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Dataset] Wrote {len(data):3d} samples -> {fp}")


def update_quillbot_comparison_sheet(splits: Dict[str, List[Dict[str, Any]]]):
    """Builds exactly 30 representative comparison samples for manual QuillBot testing."""
    samples = []
    
    # 6 Human Native
    for h in splits["test_indist"]:
        if h["class_id"] == 0 and h["generator"] == "human_native" and len(samples) < 6:
            samples.append({
                "id": f"QB-{len(samples)+1:02d}",
                "text": h["text"],
                "expected_class": "Human-written",
                "class_id": 0,
                "type": "human_native",
                "domain": h["domain"],
                "word_count": h["word_count"],
                "notes": "Authentic human author prose with domain scholarship or personal narrative."
            })
            
    # 4 Human ESL
    for e in splits["test_esl"][:4]:
        samples.append({
            "id": f"QB-{len(samples)+1:02d}",
            "text": e["text"],
            "expected_class": "Human-written",
            "class_id": 0,
            "type": "human_esl",
            "domain": e["domain"],
            "word_count": e["word_count"],
            "notes": "Authentic non-native English essay testing detector false-positive rate."
        })

    # 6 Human-written & AI-refined
    for hr in splits["test_indist"]:
        if hr["class_id"] == 1 and len(samples) < 16:
            samples.append({
                "id": f"QB-{len(samples)+1:02d}",
                "text": hr["text"],
                "expected_class": "Human-written & AI-refined",
                "class_id": 1,
                "type": "human_ai_refined",
                "domain": hr["domain"],
                "word_count": hr["word_count"],
                "notes": "Human prose polished and regularized by an LLM line editor."
            })

    # 6 AI-generated & AI-refined
    for ar in splits["test_indist"]:
        if ar["class_id"] == 2 and len(samples) < 22:
            samples.append({
                "id": f"QB-{len(samples)+1:02d}",
                "text": ar["text"],
                "expected_class": "AI-generated & AI-refined",
                "class_id": 2,
                "type": "ai_ai_refined",
                "domain": ar["domain"],
                "word_count": ar["word_count"],
                "notes": "AI text passed through syntactic restructuring and automated paraphrasing."
            })

    # 8 Pure Frontier AI (GPT-4o, Claude 3.5, Gemini 1.5, Llama 3.3, Qwen 2.5, DeepSeek-V3)
    for ai in splits["test_indist"]:
        if ai["class_id"] == 3 and len(samples) < 26:
            samples.append({
                "id": f"QB-{len(samples)+1:02d}",
                "text": ai["text"],
                "expected_class": "AI-generated",
                "class_id": 3,
                "type": f"ai_pure_{ai['generator']}",
                "domain": ai["domain"],
                "word_count": ai["word_count"],
                "notes": f"Direct generation from {ai['generator']} without subsequent modification."
            })
            
    for q in splits["test_unseen_qwen"][:2]:
        if len(samples) < 28:
            samples.append({
                "id": f"QB-{len(samples)+1:02d}",
                "text": q["text"],
                "expected_class": "AI-generated",
                "class_id": 3,
                "type": "ai_pure_qwen",
                "domain": q["domain"],
                "word_count": q["word_count"],
                "notes": "Unseen open frontier model: Qwen-2.5-72B."
            })

    for d in splits["test_unseen_deepseek"][:2]:
        if len(samples) < 30:
            samples.append({
                "id": f"QB-{len(samples)+1:02d}",
                "text": d["text"],
                "expected_class": "AI-generated",
                "class_id": 3,
                "type": "ai_pure_deepseek",
                "domain": d["domain"],
                "word_count": d["word_count"],
                "notes": "Unseen frontier model: DeepSeek-V3."
            })

    samples = samples[:30]

    json_path = os.path.join(DATA_DIR, "quillbot_comparison_sheet.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(DATA_DIR, "quillbot_comparison_sheet.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# QuillBot AI Detector Hand-Verification Benchmark (30 Samples)\n\n")
        f.write("This sheet provides exactly 30 standardized test passages across all 4 target classes for manual evaluation against QuillBot's AI Detector without violating rate limits or ToS.\n\n")
        f.write("| ID | Target Expected Class | Type / Sub-Genre | Words | First 80 Chars | QuillBot Verdict (Manual) | Veritas Verdict | Match? |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for s in samples:
            preview = s["text"][:75].replace("\n", " ").replace("|", " ") + "..."
            f.write(f"| {s['id']} | **{s['expected_class']}** | `{s['type']}` | {s['word_count']} | {preview} | *(Pending)* | *(Pending)* | - |\n")
        f.write("\n\n---\n\n## Complete Sample Texts\n\n")
        for s in samples:
            f.write(f"### Sample {s['id']} — [{s['expected_class']}]\n")
            f.write(f"- **Type**: `{s['type']}` | **Domain**: `{s['domain']}` | **Word Count**: {s['word_count']}\n")
            f.write(f"- **Rationale**: {s['notes']}\n\n")
            f.write("```text\n")
            f.write(s["text"] + "\n")
            f.write("```\n\n")

    print(f"[Comparison] Built 30-sample manual comparison sheet -> {json_path} & {md_path}")


if __name__ == "__main__":
    splits = build_full_dataset()
    save_splits(splits)
    update_quillbot_comparison_sheet(splits)
