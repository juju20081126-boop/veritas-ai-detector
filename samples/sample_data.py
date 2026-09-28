"""
Preloaded Realistic Sample Datasets for Instant Demonstration & Testing
Includes:
1. Pure ChatGPT 4o Generated Academic Essay
2. Pure Claude 3.5 Sonnet Technical Analysis
3. Genuine Human Academic History Essay
4. Genuine Human Narrative & Personal Reflection
5. Mixed Submission (Human Student Intro + AI Generated Body)
"""

SAMPLES = {
    "chatgpt_academic": {
        "title": "ChatGPT 4o: Impact of AI on Modern Education",
        "description": "Standard generative AI essay exhibiting classic LLM signposts ('multifaceted', 'pivotal role', 'delve', 'rich tapestry', 'in conclusion').",
        "text": """In the contemporary era of rapid technological proliferation, artificial intelligence has emerged as a transformative force reshaping the landscape of modern education. At its core, AI offers multifaceted tools that play a pivotal role in personalizing learning experiences for students across diverse academic domains. Furthermore, intelligent tutoring systems are capable of analyzing student performance metrics in real time, thereby fostering an adaptive pedagogical environment that caters to individual cognitive paces. 

Moreover, educators can harness these sophisticated algorithms to automate routine administrative burdens, including grading and curriculum scheduling, which in turn allows them to dedicate more time to mentorship and holistic student development. However, it is essential to remember that this technological integration is not without substantial challenges. Questions surrounding data privacy, algorithmic bias, and the potential erosion of critical thinking skills underscore the imperative need for robust regulatory frameworks. 

Ultimately, navigating the complexities of this evolving educational realm requires a balanced approach. By understanding both the promising opportunities and inherent ethical dilemmas, educational institutions can pave the way for a more equitable future. In conclusion, artificial intelligence stands as a testament to human ingenuity, offering a vibrant catalyst for intellectual advancement when wielded with responsibility and foresight."""
    },
    "claude_technical": {
        "title": "Claude 3.5: Quantum Computing Architectures",
        "description": "Sophisticated LLM technical exposition featuring balanced transitions, high structural uniformity, and characteristic analytical phrasing.",
        "text": """Quantum computing represents a paradigm shift in computational complexity theory, fundamentally departing from classical von Neumann architectures through the principles of superposition and entanglement. By leveraging qubits capable of existing in continuous linear combinations of orthogonal basis states, quantum algorithms such as Shor's and Grover's achieve provable superpolynomial speedups for specialized mathematical problem classes.

Delving into physical hardware implementations, superconducting circuit topologies and trapped-ion systems presently occupy the vanguard of experimental development. Superconducting qubits utilize Josephson junctions operating at millikelvin dilution refrigerator temperatures to mitigate environmental thermal decoherence. Conversely, trapped-ion systems achieve exceptionally high gate fidelities and coherence lifetimes by confining ionized atomic species within radio-frequency Paul traps, manipulated via precisely calibrated laser pulses.

Nevertheless, achieving fault-tolerant quantum computation remains fundamentally constrained by quantum error correction overhead. The implementation of surface codes requires thousands of physical qubits to synthesize a single fault-tolerant logical qubit. As researchers navigate these intricate engineering hurdles, the synergy between hardware scalability and noise-resilient quantum algorithms will dictate the timeline for realizing practical quantum utility."""
    },
    "human_academic": {
        "title": "Genuine Human: Mediterranean Maritime Trade (Historical Excerpt)",
        "description": "Real human historical scholarship with high burstiness, irregular clause rhythms, varied sentence lengths, and specialized archival references.",
        "text": """Braudel was right about the mountains, but he sorely misunderstood the grain barges. When Genoese merchants docked at Chios in the winter of 1432, they brought not silver, but moldy wheat—two thousand bushels of it, shipped under wet canvas from the Crimean depots. The local podestà threw a fit. Taxes hadn't been collected since Michaelmas, the garrison was grumbling about sour wine, and now the Greek bakers refused to touch the damp flour. 

Maritime registers in the Archivio di Stato di Genova reveal that captain Domenico de' Negri spent six weeks bickering over harbor dues. It was petty. It was exhausting. And yet, this was precisely how Mediterranean empires held together: not through grand imperial edicts, but through desperate, greasy negotiations over spoiled pantry staples. 

If we look closely at de' Negri's cargo manifests, the numbers tell an odd story. He took on fifty amphorae of pitch, four crates of mastic resin, and several bundles of raw wool that smelled so foul the dockhands demanded double wages to heave them into the hold. One deckhand simply walked off the job and disappeared into the island's olive groves. Nobody went after him."""
    },
    "human_narrative": {
        "title": "Genuine Human: Night Fishing off Montauk",
        "description": "Authentic personal prose with idiosyncratic sensory details, slang, irregular punctuation, and sudden tonal shifts.",
        "text": """The tide turned at 3:15 a.m., cold and greasy, slapping the aluminum hull of my uncle's beat-up skiff like a wet towel. My thumbs were so numb from cutting bunker chunks that I couldn't feel the line slipping between my knuckles until a bluefish hit it. Wham. Just ripped thirty yards of twenty-pound mono off the spool before I could even set the drag. 

Uncle Sal didn't even look up from his thermos of burnt Chock Full o'Nuts. 'Keep the rod tip up, dummy,' he muttered, blowing steam into the damp cabin air. He'd been doing this forty years and still had salt in his eyebrows. 

The fish gave two vicious head-shakes, broke water under the stern spotlight—all green foam and sharp teeth—and spit the rusted circle hook right back into my chest waders. Sal chuckled, clicked off the lantern, and told me to grab the thermos. Some nights you catch the fish; most nights you just freeze your tail off for nothing."""
    },
    "mixed_student": {
        "title": "Mixed Submission: Student Outline + AI Expanded Body",
        "description": "Common real-world scenario where a student writes their own opening thought, then copies ChatGPT for the middle analysis, and adds a rushed human conclusion.",
        "text": """My history teacher told us to write about the Industrial Revolution, so I decided to look at how steam engines changed London factories because my grandfather used to work near the old rail yards. It got crazy back then with smoke everywhere and people moving from farms into crowded tenements.

In the contemporary era of historical analysis, the steam engine has emerged as a multifaceted technological catalyst that reshaped the economic landscape of Victorian Britain. By harnessing the thermodynamic efficiency of James Watt's separate condenser, urban manufacturing facilities liberated themselves from geographical reliance upon riparian water currents. Furthermore, this mechanized transition fostered unprecedented economies of scale, establishing a rich tapestry of centralized urban production. At its core, the proliferation of steam-driven textile mills underscores the profound realignment of labor dynamics and capital accumulation.

Honestly though, when you think about breathing in coal dust all day for pennies, it sounds awful. I wouldn't have lasted two days in those mills."""
    }
}
