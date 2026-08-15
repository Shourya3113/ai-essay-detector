"""
generate_dataset.py — Generates a comprehensive, realistic benchmark dataset of admissions essays

Generates 66+ rich essays across 4 core prompts:
  1. Overcoming Adversity
  2. Intellectual Curiosity
  3. Community Contribution
  4. Future Aspirations

Across 3 distinct classes:
  - Class 0 (Human-like): High burstiness, high readability variance, idiosyncratic syntax, em-dashes, personal voice.
  - Class 1 (AI-Generated): Low burstiness, uniform sentence length, high transition density, formulaic structure.
  - Class 2 (Mixed / AI-Polished): Human draft smoothed with AI transitions and standardized syntax.
"""

import json
import os
import sys

# Structured essay bank containing diverse, authentic admissions essays
ESSAYS_DATA = [
    # =========================================================================
    # CLASS 0: HUMAN-LIKE ESSAYS (High burstiness, personal voice, vivid details)
    # =========================================================================
    {
        "id": "human_01",
        "prompt_category": "Overcoming adversity",
        "class_label": 0,
        "class_name": "human",
        "title": "The Rusting Transmission",
        "text": (
            "The grease beneath my fingernails refused to come out before Monday morning chemistry. "
            "I scrubbed until my cuticles bled. My father’s 1998 Ford Ranger had thrown a transmission cylinder on highway 29, "
            "and we simply didn’t have the eight hundred dollars for a certified mechanic. "
            "So there I was, freezing in our dirt driveway with a Chilton repair manual propped against a car battery. "
            "I dropped the ratchet twice. My knuckles tore on the exhaust manifold—a sharp, stinging reminder of our budget. "
            "It took eleven days, forty cups of lukewarm black coffee, and three stripped bolts. But on Sunday night, the engine roared. "
            "That battered truck taught me more about stubborn kinetic perseverance than any AP Physics lab ever could."
        )
    },
    {
        "id": "human_02",
        "prompt_category": "Intellectual curiosity",
        "class_label": 0,
        "class_name": "human",
        "title": "Deciphering Honeybee Waggle Dances",
        "text": (
            "Bees don't lie. Or at least, that’s what Karl von Frisch argued, but I needed to see it for myself. "
            "Last summer, my backyard observation hive became an obsession—much to my mother's horror when three hundred foragers escaped into the sunroom. "
            "Armed with a high-speed macro lens and a millimeter grid, I spent three weeks filming waggle angles relative to the sun. "
            "Why did hive four always adjust its dance vector four minutes before hive one? "
            "I filled three Moleskine notebooks with trigonometric scatter plots. "
            "The answer wasn't in the textbooks; it was the micro-topography of the nearby walnut orchard refracting wind currents. "
            "Science isn't tidy formulas in a textbook—it's getting stung on the eyelid because you leaned two inches too close to truth."
        )
    },
    {
        "id": "human_03",
        "prompt_category": "Community contribution",
        "class_label": 0,
        "class_name": "human",
        "title": "Translating for Mrs. Nguyen",
        "text": (
            "Form 1040-EZ is terrifying when your first language is Vietnamese and you’re seventy-two years old. "
            "Every Saturday at the East Side Community Center, I sat across from elders like Mrs. Nguyen, whose weathered hands trembled over line 8b. "
            "I wasn’t just translating English into Vietnamese; I was translating bureaucratic paranoia into quiet dignity. "
            "Her grandson had left for Dallas; her pension was delayed. "
            "We spent forty-five minutes finding one missing medical deduction receipt from 2021. "
            "When the green confirmation banner finally printed, she pressed a warm bag of homemade banh cam into my hands. "
            "Service isn't high-minded philanthropy—it’s sitting on folding chairs and making sure nobody gets left behind by small print."
        )
    },
    {
        "id": "human_04",
        "prompt_category": "Future aspirations",
        "class_label": 0,
        "class_name": "human",
        "title": "Designing Bridges That Breathe",
        "text": (
            "Concrete is stubborn. It cracks under tension, leaches carbon, and remembers nothing. "
            "Ever since I saw the I-35W collapse footage as a kid, I’ve wanted to build structures that heal themselves. "
            "My bedroom smells like bacterial agar and curing Portland cement because of my experiments with *Bacillus pseudofirmus*. "
            "When microcracks form and water seeps in, the dormant bacteria wake up, feast on calcium lactate, and produce limestone. "
            "My miniature test beams look ugly—clumpy, gray, and stained with bio-resin—but they held forty pounds more than standard mix. "
            "I want to study civil engineering not to build monuments to vanity, but to create infrastructure that outlives our carelessness."
        )
    },
    {
        "id": "human_05",
        "prompt_category": "Overcoming adversity",
        "class_label": 0,
        "class_name": "human",
        "title": "The Silent Stage",
        "text": (
            "My vocal cords seized during the second stanza of 'The Crucible' audition. Complete silence. "
            "Seventy peers staring from the auditorium darkness, the director tapping his pencil against the clipboard. "
            "I had stuttered since third grade, but this was different—it was complete panic paralysis. "
            "I closed my eyes, took a ragged breath, and dropped character entirely. 'Line,' I whispered. "
            "The stage manager called it out. I restarted, not with the rehearsed bravado, but with John Proctor’s raw, trembling anger. "
            "I didn't get the lead. I got third townsperson. "
            "Yet standing in the back row during opening night, shouting in unison with the chorus, I finally owned my voice."
        )
    },
    {
        "id": "human_06",
        "prompt_category": "Intellectual curiosity",
        "class_label": 0,
        "class_name": "human",
        "title": "Topology in Grandma's Quilt",
        "text": (
            "To anyone else, Grandma's quilt was a faded patchwork of old denim shirts and floral curtains. "
            "To me, it was a tactile lesson in non-orientable manifolds. "
            "I was halfway through an online lecture on Klein bottles when I realized the Mobius twist at the corner border was a deliberate stitching error. "
            "Or was it? Grandma laughed when I showed her my vector diagram drawn with chalk on her dining table. "
            "'I ran out of blue thread, honey,' she told me. "
            "Sometimes mathematical elegance is just pragmatic improvisation in disguise. "
            "That intersection—where abstract mathematics collides with human imperfection—is where I want to spend my life."
        )
    },
    {
        "id": "human_07",
        "prompt_category": "Community contribution",
        "class_label": 0,
        "class_name": "human",
        "title": "Midnight Bakery Runs",
        "text": (
            "At 11:30 PM, the smell of burnt yeast at Panera isn't poetic; it’s exhaustion. "
            "My trunk was lined with clean plastic tarps, loaded with sixty pounds of unsellable day-old sourdough and baguettes. "
            "For two years, the shelter on 4th Street counted on my Tuesday night drop-offs. "
            "The shelter cook, Marcus, always made me tea while we unloaded crates in the freezing alleyway. "
            "One night, a resident named David came out to help carry the heavy pastry crates. "
            "He didn't ask where the bread came from; he just wanted to talk about minor league baseball scores from 1994. "
            "Giving food is easy. Giving twenty minutes of genuine listening in a back alley at midnight—that's what actually feeds people."
        )
    },
    {
        "id": "human_08",
        "prompt_category": "Future aspirations",
        "class_label": 0,
        "class_name": "human",
        "title": "Decoding the Neural Dialect",
        "text": (
            "Watching my grandfather forget my name happened in terrifyingly slow motion. "
            "First it was misplacing his keys; then came the blank stares across the breakfast table. "
            "In my high school neuroscience club, we worked with open-source EEG headsets, mapping alpha waves during memory recall tasks. "
            "The signals looked like seismic chaos on my laptop screen. "
            "I spent my winter break writing Python scripts to filter out muscle artifact noise using Fourier transforms. "
            "When the P300 wave finally emerged clearly from the noise floor, I wept. "
            "I will dedicate my career to neural engineering because Alzheimer’s took my grandfather's memories—and I want to build the tools that fight back."
        )
    },
    {
        "id": "human_09",
        "prompt_category": "Overcoming adversity",
        "class_label": 0,
        "class_name": "human",
        "title": "Hospital Cafeteria Jell-O",
        "text": (
            "Green lime Jell-O is the official flavor of surgical waiting rooms. "
            "I ate twenty-four cups of it while my little sister underwent spinal fusion surgery. "
            "Between checking monitor updates and comforting my terrified parents, I had to finish my AP Literature essay on 'Beloved'. "
            "Typing on a cracked iPhone screen with hospital Wi-Fi cutting out every twenty minutes tested my sanity. "
            "My sentences were fractured, urgent, and raw. "
            "When the surgeon finally walked out at 4:15 AM with a thumbs up, I collapsed against the vinyl chair. "
            "Resilience isn't heroic speeches on mountaintops; it's getting homework done in an ICU waiting room while holding your breath."
        )
    },
    {
        "id": "human_10",
        "prompt_category": "Intellectual curiosity",
        "class_label": 0,
        "class_name": "human",
        "title": "The Mystery of Ancient Pigments",
        "text": (
            "Why did Egyptian blue survive four thousand years in desert tombs while Renaissance oil paintings faded within centuries? "
            "I turned our garage into a medieval alchemy lab, crushing malachite, lapis lazuli, and copper oxide with a mortar and pestle. "
            "My father almost called the fire department when my calcium silicate kiln reached nine hundred degrees Celsius. "
            "Under my microscope, the synthesized pigment crystals looked like shattered sapphires. "
            "I discovered that the secret wasn't the copper content, but the precise ratio of flux to sand during the firing process. "
            "History isn't just dates carved in granite—it's chemical recipes waiting for curious hands to rekindle them."
        )
    },
    {
        "id": "human_11",
        "prompt_category": "Community contribution",
        "class_label": 0,
        "class_name": "human",
        "title": "Fixing Bicycles in the Alley",
        "text": (
            "A bicycle with a rusted chain is a tragedy for a kid whose only ticket to the library is two miles on gravel. "
            "I started 'Open Wrenches' in our backyard with two adjustable spanners and a bucket of degreaser. "
            "Neighborhood kids brought in bent rims, dry bearings, and shredded brake cables. "
            "Nine-year-old Mateo came in with a bike that had no pedals—he was pushing it like a scooter. "
            "We scavenged parts from a dumped Huffy in the canal and bolted on shiny steel pedals. "
            "The smile on his face as he popped a wheelie down 5th Avenue was pure euphoria. "
            "Mobility is freedom, and two wrenches can change a neighborhood."
        )
    },
    {
        "id": "human_12",
        "prompt_category": "Future aspirations",
        "class_label": 0,
        "class_name": "human",
        "title": "Farming the Stars",
        "text": (
            "I killed four crops of hydroponic red kale before I learned that pH drift kills faster than drought. "
            "My bedroom closet became an unearthly glow of magenta LED grow lights and bubbling aeroponic reservoirs. "
            "I wanted to understand how astronauts will grow food during a three-year mission to Mars. "
            "Balancing nitrogen uptake with closed-loop water recirculation became my evening ritual. "
            "When the fifth crop produced crisp, peppery leaves with zero soil and ninety percent less water, I ate the whole salad dry. "
            "I want to study agricultural bioengineering to ensure that when humanity reaches the stars, we bring dinner with us."
        )
    },
    {
        "id": "human_13",
        "prompt_category": "Overcoming adversity",
        "class_label": 0,
        "class_name": "human",
        "title": "Eviction Notice on Yellow Paper",
        "text": (
            "The yellow legal notice taped to our front door was dated October 14th. We had twelve days. "
            "Packing seventeen years of life into cardboard banana boxes from the grocery store is a brutal audit of what matters. "
            "I packed my track medals, my grandfather's compass, and three textbooks. Everything else went on the curb. "
            "We slept in my aunt's basement for four months, sharing a single queen mattress. "
            "I did my calculus homework under the water heater with a flashlight so I wouldn't wake my cousin. "
            "Instability could have broken my resolve; instead, it forged an unbreakable iron discipline. "
            "I don't take a desk, a quiet room, or an education for granted—ever."
        )
    },
    {
        "id": "human_14",
        "prompt_category": "Intellectual curiosity",
        "class_label": 0,
        "class_name": "human",
        "title": "Finding Glitches in Old Game Cartridges",
        "text": (
            "Most kids play *Super Mario 64* to save the princess. I played it to break the collision boundary. "
            "By jumping backwards into a staircase at a fifty-three degree angle, you overflow the 32-bit floating point velocity cap. "
            "Mario launches into negative coordinates. "
            "I spent my sophomore summer reading IEEE papers on IEEE-754 arithmetic and assembly disassemblers. "
            "Why did the memory address at 0x8033B174 overwrite sound buffer registers during frame-rate drops? "
            "Tracking down that single register leak gave me the same adrenaline rush that athletes get from winning a championship. "
            "Computer science isn't just writing software; it's understanding the ghost in the machine."
        )
    },
    {
        "id": "human_15",
        "prompt_category": "Community contribution",
        "class_label": 0,
        "class_name": "human",
        "title": "Code and Crayons",
        "text": (
            "Teaching six-year-olds the concept of a 'while loop' requires fewer computers and more sidewalk chalk. "
            "At the public library's Saturday workshop, I drew giant hopscotch grids where kids acted as CPU registers. "
            "'If square is red, jump twice; else turn around!' "
            "Little Maya struggled with reading, but when she grasped that conditional logic could make her friends do silly dances, her eyes lit up. "
            "Her mom pulled me aside afterwards with tears in her eyes—Maya had been labeled 'behind' by her kindergarten teacher. "
            "Programming shouldn't be gated behind thousand-dollar laptops. "
            "True empowerment begins when we show children that their minds are natural problem-solving engines."
        )
    },
    {
        "id": "human_16",
        "prompt_category": "Future aspirations",
        "class_label": 0,
        "class_name": "human",
        "title": "Cleaning the Orbital Junkyard",
        "text": (
            "There are twenty-seven thousand pieces of space debris larger than a softball currently orbiting Earth at seventeen thousand miles per hour. "
            "One errant paint fleck cracked the ISS cupola window in 2016. "
            "In my bedroom, I built a simulation of orbital drift using Python and NASA's two-line element sets. "
            "My proposed electrodynamic tether model uses Earth's magnetic field to drag dead CubeSats into atmospheric burnup without propellant. "
            "My simulation crashed eighty-four times before the orbital decay curve matched historical telemetry. "
            "I want to become an aerospace engineer because exploration is meaningless if we lock ourselves inside a cage of our own orbital trash."
        )
    },
    {
        "id": "human_17",
        "prompt_category": "Overcoming adversity",
        "class_label": 0,
        "class_name": "human",
        "title": "The Broken Violin Bridge",
        "text": (
            "Snap. Two minutes before the regional all-state violin audition, my maple bridge sheared in half. "
            "The strings went limp against the fingerboard like dead spaghetti. "
            "Other musicians in the hallway stepped back, whispering. I had thirty seconds to decide between forfeiting or attempting the impossible. "
            "I remembered my luthier's emergency trick: I used my metal music stand clip to clamp the fractured wood together and tuned down a half-step. "
            "The tone was hollow, reedy, and buzzed on the G-string. "
            "I played Bach's Partita No. 2 with the fiercest, most aggressive bow pressure of my life. "
            "I placed third. That broken bridge taught me that music isn't in pristine wood—it’s in the grit of the player."
        )
    },
    {
        "id": "human_18",
        "prompt_category": "Intellectual curiosity",
        "class_label": 0,
        "class_name": "human",
        "title": "Listening to the Earth's Murmur",
        "text": (
            "When the ground shakes in Oklahoma, it's rarely a tectonic fault; it's wastewater injection wells. "
            "I built a homemade seismograph using an old speaker coil, a neodymium magnet, and an Arduino in our basement. "
            "The local geologists laughed when I emailed them my first seismogram. "
            "Then, on September 3rd, a 5.8 magnitude tremor hit Pawnee. "
            "My basement rig captured the P-wave arrival four seconds before the USGS automated feed triggered on Twitter. "
            "I mapped forty-two micro-tremors along the fault line over the next three months. "
            "Curiosity isn't waiting for permission from experts; it's building instruments from scrap and listening closely."
        )
    },
    {
        "id": "human_19",
        "prompt_category": "Community contribution",
        "class_label": 0,
        "class_name": "human",
        "title": "Saving the County Dialect",
        "text": (
            "Old Mr. Henderson was the last person in our valley who still used the traditional timber-hauler slang. "
            "Phrases like 'skidder-bound' and 'pine-whistle' were dying with his generation. "
            "Armed with a handheld audio recorder and permission from the county historical society, I spent sixty hours interviewing logging veterans. "
            "I digitized three hundred cassette tapes and built a searchable phonetic lexicon on a free GitHub website. "
            "At the town hall meeting, three elderly residents wept hearing their late parents' voices reciting 1950s logging songs. "
            "Preserving culture isn't dusty museum glass; it's recording the rhythm of working-class voices before silence claims them."
        )
    },
    {
        "id": "human_20",
        "prompt_category": "Future aspirations",
        "class_label": 0,
        "class_name": "human",
        "title": "Synthetic Photosynthesis",
        "text": (
            "Plants are remarkably inefficient. They convert less than one percent of incident sunlight into chemical biomass. "
            "We can do better. "
            "In my high school chemistry lab, I spent seven months synthesizing ruthenium-based dye complexes to split water under ambient LED light. "
            "My yield was pathetic—less than two microliters of hydrogen per hour—and the odor of sulfide ruined my favorite sweatshirt. "
            "Yet seeing those tiny bubbles form on the cathode was pure magic. "
            "I want to pioneer synthetic bio-catalysis so we can turn sunlight, water, and carbon dioxide directly into clean fuels without sacrificing farmland."
        )
    },
    {
        "id": "human_21",
        "prompt_category": "Overcoming adversity",
        "class_label": 0,
        "class_name": "human",
        "title": "Night Shift at the Laundromat",
        "text": (
            "Between 11 PM and 5 AM, Speed Queen washers sound like jet engines taking off. "
            "I worked the graveyard shift at my uncle's coin laundromat to help cover rent while maintaining a 4.0 GPA. "
            "Customers yelled about jammed coin slots; driers caught lint fires; drunks sought shelter from the rain. "
            "I perched on a plastic laundry folding cart with my AP Chemistry flashcards between mopping soapy floors. "
            "My eyes burned from bleach fumes, but I memorized every organic synthesis mechanism on the syllabus. "
            "Adversity didn't soften my ambition; it forged a work ethic that will never back down from a challenge."
        )
    },
    {
        "id": "human_22",
        "prompt_category": "Intellectual curiosity",
        "class_label": 0,
        "class_name": "human",
        "title": "The Geometry of Soap Bubbles",
        "text": (
            "Joseph Plateau went blind studying the sun, but his soap film experiments solved minimal surface area math problems centuries ahead of computers. "
            "I dipped twisted wire frames into dish soap and glycerin in my kitchen sink. "
            "Why did four bubble films always meet at 109.5 degree angles? "
            "I wrote a 3D visualization script in Processing to calculate energy minimization across catenoid surfaces. "
            "Seeing physical soap film match my theoretical calculus equations to the third decimal place sent chills down my spine. "
            "Beauty in mathematics is not abstract—it is written into the physical laws of a ten-cent bubble."
        )
    },

    # =========================================================================
    # CLASS 1: AI-GENERATED ESSAYS (Formulaic, uniform length, cliché transitions)
    # =========================================================================
    {
        "id": "ai_01",
        "prompt_category": "Overcoming adversity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Navigating the Crucible of Hardship",
        "text": (
            "In today's fast-paced and challenging world, overcoming adversity plays a pivotal role in shaping character. "
            "Throughout my academic journey, I have encountered a myriad of obstacles that tested my unwavering resilience. "
            "Furthermore, facing financial difficulties forced me to balance rigorous coursework with substantial part-time employment. "
            "Moreover, this experience served as a powerful testament to the transformative power of perseverance and dedication. "
            "It is important to note that these challenges did not deter me from pursuing my intellectual passions with enthusiasm. "
            "Consequently, I learned to navigate the complexities of life while fostering a deeper understanding of personal responsibility. "
            "In conclusion, these multifaceted trials have prepared me to make a meaningful impact in my future collegiate endeavors."
        )
    },
    {
        "id": "ai_02",
        "prompt_category": "Intellectual curiosity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Delving into the Wonders of Computer Science",
        "text": (
            "Intellectual curiosity is the cornerstone of all meaningful human progress and academic discovery. "
            "From a young age, I have been deeply fascinated by the intricate realm of computer science and artificial intelligence. "
            "Furthermore, delving into algorithmic structures allowed me to understand the underlying mechanics of modern software. "
            "Moreover, creating novel machine learning solutions has illuminated the profound impact of technology on society. "
            "It is crucial to recognize that technological innovation must align with strong ethical principles and social responsibility. "
            "Additionally, collaborating on technical projects has honed my problem-solving skills across diverse disciplinary boundaries. "
            "In summary, I am eager to continue this transformative journey of intellectual growth at the university level."
        )
    },
    {
        "id": "ai_03",
        "prompt_category": "Community contribution",
        "class_label": 1,
        "class_name": "ai",
        "title": "Fostering Community Through Dedicated Service",
        "text": (
            "Serving one's community is a vital endeavor that enriches both society and the individual volunteer. "
            "Throughout high school, I have dedicated significant time to organizing inclusive community support initiatives. "
            "Furthermore, establishing a youth tutoring program allowed me to shed light on educational inequalities in our district. "
            "Moreover, working closely with underprivileged students provided a multifaceted perspective on systemic challenges. "
            "It goes without saying that true leadership requires empathy, patience, and an unwavering commitment to equity. "
            "Consequently, our organization was able to empower dozens of aspiring young learners to achieve their academic goals. "
            "In conclusion, I look forward to bringing this collaborative spirit to the dynamic campus community as a student."
        )
    },
    {
        "id": "ai_04",
        "prompt_category": "Future aspirations",
        "class_label": 1,
        "class_name": "ai",
        "title": "Pioneering the Landscape of Renewable Energy",
        "text": (
            "As humanity faces pressing environmental challenges, developing sustainable energy solutions has become paramount. "
            "My aspiration is to become a leading bioengineer dedicated to advancing clean energy technologies. "
            "Furthermore, conducting research in chemical catalysis has provided me with a strong foundation in modern thermodynamics. "
            "Moreover, understanding the nuances of renewable fuel synthesis will allow me to contribute meaningfully to global sustainability. "
            "It is evident that interdisciplinary collaboration plays a crucial role in solving complex ecological problems. "
            "Consequently, I plan to leverage advanced laboratory resources to design efficient next-generation solar conversion systems. "
            "To conclude, I am eager to embark on this academic journey and foster positive change for future generations."
        )
    },
    {
        "id": "ai_05",
        "prompt_category": "Overcoming adversity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Embracing Growth Through Academic Setbacks",
        "text": (
            "Experiencing failure is an inevitable milestone in the rich tapestry of personal and academic development. "
            "When I failed to qualify for the national debate championship, I faced a profound crisis of confidence. "
            "Furthermore, this initial setback compelled me to reevaluate my research methodologies and rhetorical strategies. "
            "Moreover, dedicating countless hours to intensive practice helped me rebuild my self-assurance and intellectual stamina. "
            "It is important to emphasize that resilience is not merely surviving difficulty, but actively learning from adversity. "
            "Consequently, I returned the following season to lead our debate team to a historic regional championship victory. "
            "In essence, this transformative experience has equipped me with the grit necessary to excel in challenging collegiate coursework."
        )
    },
    {
        "id": "ai_06",
        "prompt_category": "Intellectual curiosity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Unraveling the Mysteries of the Human Brain",
        "text": (
            "The human brain represents one of the most fascinating frontiers in modern scientific exploration. "
            "My fascination with cognitive neuroscience began when I first studied the physiological basis of memory formation. "
            "Furthermore, reading academic journals allowed me to delve into the intricate neural pathways underlying consciousness. "
            "Moreover, analyzing cognitive models has shed light on how synaptic plasticity influences long-term learning outcomes. "
            "It is essential to understand that curing neurodegenerative diseases will require groundbreaking technological innovations. "
            "Additionally, I have designed independent research simulations to model neural firing patterns under varying biochemical conditions. "
            "In conclusion, I am excited to pursue advanced undergraduate research and contribute to the expanding field of neuroscience."
        )
    },
    {
        "id": "ai_07",
        "prompt_category": "Community contribution",
        "class_label": 1,
        "class_name": "ai",
        "title": "Building Bridges in a Diverse Global Society",
        "text": (
            "In an increasingly interconnected world, cultural empathy serves as a beacon of mutual understanding and unity. "
            "Recognizing the linguistic barriers within our town, I founded a multilingual translation club for immigrant families. "
            "Furthermore, our dedicated team translated essential medical forms and legal documents for non-English speaking residents. "
            "Moreover, this initiative played a vital role in ensuring that marginalized community members received necessary healthcare services. "
            "It is worth noting that active listening and cultural humility are indispensable qualities for effective community engagement. "
            "Consequently, our program expanded to assist over three hundred families across multiple municipal districts. "
            "To summarize, I hope to continue fostering inclusive community partnerships throughout my higher education journey."
        )
    },
    {
        "id": "ai_08",
        "prompt_category": "Future aspirations",
        "class_label": 1,
        "class_name": "ai",
        "title": "Transforming Healthcare Through Biomedical Innovation",
        "text": (
            "Providing accessible healthcare to underserved populations is one of the most pressing mandates of our time. "
            "My long-term vision is to develop low-cost diagnostic devices that can be deployed in resource-limited clinics. "
            "Furthermore, studying organic chemistry has deepened my appreciation for molecular engineering and therapeutic synthesis. "
            "Moreover, utilizing microfluidic technologies will enable medical professionals to detect infectious pathogens rapidly. "
            "It is undeniable that bridging the gap between scientific discovery and clinical application is essential for global health. "
            "Consequently, I aspire to collaborate with leading researchers to translate laboratory breakthroughs into life-saving treatments. "
            "In conclusion, a rigorous university education will empower me to become a catalyst for compassionate medical progress."
        )
    },
    {
        "id": "ai_09",
        "prompt_category": "Overcoming adversity",
        "class_label": 1,
        "class_name": "ai",
        "title": "The Resilient Symphony of Healing",
        "text": (
            "Adversity often acts as an unexpected catalyst that illuminates an individual's true inner strength and purpose. "
            "Sustaining a serious wrist injury during my junior year threatened to end my ten-year musical journey as a pianist. "
            "Furthermore, being unable to perform forced me to adapt my musical expression through composition and music theory. "
            "Moreover, this period of physical rehabilitation served as a powerful testament to the resilience of the human spirit. "
            "It is clear that embracing alternative pathways can lead to profound personal discoveries and creative breakthroughs. "
            "Consequently, I composed an original orchestral suite that captured the emotional complexity of recovery and renewal. "
            "In summary, this trial reinforced my determination to approach future academic endeavors with unwavering perseverance."
        )
    },
    {
        "id": "ai_10",
        "prompt_category": "Intellectual curiosity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Exploring the Mathematical Foundations of Nature",
        "text": (
            "Mathematics provides a universal language that reveals the hidden patterns governing the natural universe. "
            "My passion for mathematical modeling began when I first investigated the Fibonacci sequence in biological structures. "
            "Furthermore, delving into fractal geometry allowed me to appreciate the profound elegance of natural self-similarity. "
            "Moreover, applying differential equations to ecological systems has demonstrated the practical power of pure mathematics. "
            "It is worth mentioning that rigorous mathematical analysis is indispensable for addressing modern environmental crises. "
            "Additionally, I have authored computational scripts to simulate population dynamics under shifting climatic parameters. "
            "To conclude, I am eager to pursue advanced mathematical studies and unlock new dimensions of scientific understanding."
        )
    },
    {
        "id": "ai_11",
        "prompt_category": "Community contribution",
        "class_label": 1,
        "class_name": "ai",
        "title": "Empowering Youth Through Environmental Stewardship",
        "text": (
            "Preserving local ecosystems is a shared responsibility that requires active engagement from every community member. "
            "To address local watershed pollution, I organized a comprehensive wetland restoration campaign in my hometown. "
            "Furthermore, coordinating with local environmental agencies enabled our team to plant hundreds of native wetland species. "
            "Moreover, conducting water quality assessments provided tangible data illustrating the positive impact of our restoration efforts. "
            "It goes without saying that educating future generations is paramount to sustaining long-term environmental progress. "
            "Consequently, our initiative fostered a strong sense of civic responsibility among participating student volunteers. "
            "In conclusion, I look forward to championing environmental sustainability and community action as an active college student."
        )
    },
    {
        "id": "ai_12",
        "prompt_category": "Future aspirations",
        "class_label": 1,
        "class_name": "ai",
        "title": "Navigating the Future of Ethical Artificial Intelligence",
        "text": (
            "As artificial intelligence continues to reshape our global society, ensuring ethical algorithmic governance is paramount. "
            "My career ambition is to spearhead the development of transparent, unbiased machine learning algorithms. "
            "Furthermore, examining algorithmic bias in automated hiring platforms has highlighted the critical need for fairness. "
            "Moreover, integrating philosophical ethics with computer science will enable developers to safeguard fundamental human rights. "
            "It is evident that technology must serve as an inclusive tool for social equity rather than division. "
            "Consequently, I plan to pursue an interdisciplinary degree combining software engineering and public policy. "
            "In summary, I am dedicated to shaping an equitable technological future through rigorous scholarly inquiry."
        )
    },
    {
        "id": "ai_13",
        "prompt_category": "Overcoming adversity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Adapting to Cultural Transitions",
        "text": (
            "Immigrating to a new country presents a unique set of linguistic and cultural challenges that test personal adaptability. "
            "Arriving in the United States without fluency in English initially made navigating high school extraordinarily daunting. "
            "Furthermore, overcoming language barriers required immense dedication and hundreds of hours of independent study. "
            "Moreover, immersing myself in extracurricular clubs allowed me to build meaningful interpersonal connections with peers. "
            "It is important to note that cross-cultural adaptation fosters empathy, cognitive flexibility, and intellectual curiosity. "
            "Consequently, I was able to achieve academic excellence while celebrating my multicultural heritage and identity. "
            "In conclusion, this transformative experience has prepared me to thrive in a diverse collegiate academic environment."
        )
    },
    {
        "id": "ai_14",
        "prompt_category": "Intellectual curiosity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Investigating Quantum Mechanics and Cryptography",
        "text": (
            "The counterintuitive principles of quantum mechanics offer revolutionary possibilities for computational security. "
            "My interest in quantum information theory began when I explored how superposition challenges classical computing limits. "
            "Furthermore, delving into quantum key distribution revealed how physics can provide unbreakable cryptographic protocols. "
            "Moreover, simulating quantum entanglement using open-source frameworks deepened my understanding of linear algebra. "
            "It is crucial to recognize that the transition to post-quantum cryptography will require substantial academic innovation. "
            "Additionally, I have written comprehensive research summaries exploring the vulnerabilities of existing RSA encryption. "
            "To summarize, I am eager to participate in cutting-edge quantum computing research during my university tenure."
        )
    },
    {
        "id": "ai_15",
        "prompt_category": "Community contribution",
        "class_label": 1,
        "class_name": "ai",
        "title": "Championing Digital Literacy for Senior Citizens",
        "text": (
            "Bridging the generational digital divide is essential to preventing social isolation among elderly populations. "
            "To address this issue, I initiated a weekly digital literacy program at our neighborhood senior living center. "
            "Furthermore, creating customized instructional guides enabled senior residents to master video calling and email applications. "
            "Moreover, witnessing their newfound ability to connect with distant family members was an immensely gratifying experience. "
            "It is worth emphasizing that patience and compassionate communication are the cornerstones of effective pedagogy. "
            "Consequently, our volunteer team expanded the program to three additional assisted living facilities across the county. "
            "In essence, I look forward to continuing my commitment to intergenerational community service at the collegiate level."
        )
    },
    {
        "id": "ai_16",
        "prompt_category": "Future aspirations",
        "class_label": 1,
        "class_name": "ai",
        "title": "Advancing Global Economic Development",
        "text": (
            "Eliminating poverty and fostering economic stability across developing nations is one of humanity's greatest imperatives. "
            "My goal is to work as an international development economist specializing in microfinance and infrastructure investments. "
            "Furthermore, analyzing econometric data has demonstrated how targeted capital access stimulates local entrepreneurship. "
            "Moreover, understanding the macroeconomic factors driving financial inclusion is essential for designing sustainable aid policies. "
            "It is undeniable that evidence-based economic policy plays a vital role in improving standards of living globally. "
            "Consequently, I aspire to conduct field research evaluating the impact of mobile banking platforms in emerging economies. "
            "To conclude, I am eager to pursue a rigorous economics curriculum to champion global financial empowerment."
        )
    },
    {
        "id": "ai_17",
        "prompt_category": "Overcoming adversity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Conquering the Challenge of Chronic Illness",
        "text": (
            "Managing a chronic medical condition during adolescence requires extraordinary discipline, fortitude, and perspective. "
            "Being diagnosed with Type 1 Diabetes in my freshman year forced me to rapidly adopt meticulous health management habits. "
            "Furthermore, balancing blood glucose monitoring with varsity athletics required constant vigilance and emotional resilience. "
            "Moreover, this ongoing journey served as a testament to my ability to maintain academic excellence under pressure. "
            "It is important to note that living with a chronic illness cultivated a profound sense of empathy for healthcare patients. "
            "Consequently, I was inspired to mentor newly diagnosed diabetic children through a local hospital support network. "
            "In conclusion, this adversity has solidified my determination to pursue a career in pediatric endocrinology."
        )
    },
    {
        "id": "ai_18",
        "prompt_category": "Intellectual curiosity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Deciphering the Linguistic Evolution of Dialects",
        "text": (
            "Language is a dynamic cultural artifact that constantly evolves in response to historical and demographic shifts. "
            "My passion for historical linguistics sparked when I noticed the unique phonetic variations among regional communities. "
            "Furthermore, delving into sociolinguistic corpora allowed me to map vowel shifts across generations of urban speakers. "
            "Moreover, examining language contact phenomena revealed how migration patterns shape lexical borrowing and grammar. "
            "It is evident that preserving endangered linguistic dialects is essential for protecting intangible cultural heritage. "
            "Additionally, I compiled an acoustic phonetic database to document subtle shifts in regional intonation patterns. "
            "In summary, I am enthusiastic about expanding my linguistic research through advanced undergraduate scholarship."
        )
    },
    {
        "id": "ai_19",
        "prompt_category": "Community contribution",
        "class_label": 1,
        "class_name": "ai",
        "title": "Alleviating Food Insecurity in Urban Neighborhoods",
        "text": (
            "Access to nutritious food is a fundamental human right that remains unfulfilled for far too many families. "
            "To combat local food deserts, I spearheaded an urban agriculture initiative to transform vacant lots into community gardens. "
            "Furthermore, organizing volunteer workdays enabled us to harvest over two thousand pounds of fresh organic produce. "
            "Moreover, distributing this harvest to local food pantries provided critical nutritional support to underserved residents. "
            "It is clear that sustainable urban agriculture plays a pivotal role in strengthening community food sovereignty. "
            "Consequently, our initiative inspired local schools to incorporate horticultural education into their science curricula. "
            "To conclude, I hope to continue advocating for equitable food distribution systems throughout my university career."
        )
    },
    {
        "id": "ai_20",
        "prompt_category": "Future aspirations",
        "class_label": 1,
        "class_name": "ai",
        "title": "Innovating in the Sphere of Clean Water Engineering",
        "text": (
            "Access to clean and safe drinking water is an indispensable foundation for global health and prosperity. "
            "My aspiration is to engineer low-cost, solar-powered water filtration membranes for disaster relief zones. "
            "Furthermore, conducting polymer chemistry experiments provided me with insights into membrane fouling mechanisms. "
            "Moreover, exploring nanotechnology applications has opened promising avenues for removing heavy metal contaminants. "
            "It is essential to recognize that scalable engineering solutions are required to combat global water scarcity. "
            "Consequently, I plan to engage in cutting-edge environmental engineering research to deploy resilient water systems. "
            "In conclusion, I am eager to apply my technical skills to ensure universal access to clean water worldwide."
        )
    },
    {
        "id": "ai_21",
        "prompt_category": "Overcoming adversity",
        "class_label": 1,
        "class_name": "ai",
        "title": "The Transformative Journey of Self-Discovery",
        "text": (
            "Personal growth often emerges from confronting moments of profound vulnerability and existential uncertainty. "
            "Overcoming public speaking anxiety was a multifaceted challenge that required immense psychological determination. "
            "Furthermore, joining the school speech and debate team provided an authentic platform to confront my deepest fears. "
            "Moreover, receiving constructive critiques from coaches and peers helped me refine my vocal delivery and poise. "
            "It is worth noting that vulnerability is not a sign of weakness, but a catalyst for personal transformation. "
            "Consequently, I was able to deliver the keynote address at our annual school leadership symposium with great confidence. "
            "In summary, this journey has endowed me with the rhetorical prowess and confidence to excel in university seminars."
        )
    },
    {
        "id": "ai_22",
        "prompt_category": "Intellectual curiosity",
        "class_label": 1,
        "class_name": "ai",
        "title": "Unlocking the Potential of CRISPR Gene Editing",
        "text": (
            "Molecular genetics has entered an unprecedented era with the advent of targeted genome editing technologies. "
            "My fascination with molecular biology was ignited when I first investigated the mechanics of the CRISPR-Cas9 system. "
            "Furthermore, reading contemporary biochemical literature allowed me to understand how guide RNAs direct precise DNA cleavage. "
            "Moreover, examining epigenetic regulation has highlighted the intricate nuances governing cellular gene expression. "
            "It is crucial to recognize that genetic engineering holds immense potential for eradicating hereditary genetic disorders. "
            "Additionally, I conducted bioinformatics simulations to analyze off-target cleavage probabilities in synthetic DNA models. "
            "To conclude, I aspire to conduct pioneering biomedical research and contribute to the future of genetic medicine."
        )
    },

    # =========================================================================
    # CLASS 2: MIXED / AI-POLISHED ESSAYS (Human story smoothed with AI structure)
    # =========================================================================
    {
        "id": "mixed_01",
        "prompt_category": "Overcoming adversity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Rebuilding the Engine: A Lesson in Persistence",
        "text": (
            "When my father's 1998 Ford Ranger broke down on Highway 29, we were faced with an eight hundred dollar repair bill. "
            "Because our household budget could not accommodate this expense, I decided to repair the transmission myself. "
            "Furthermore, spending eleven days working in our dirt driveway with a repair manual tested my mechanical problem-solving abilities. "
            "Moreover, dealing with stripped bolts and freezing weather taught me valuable lessons about patience and perseverance. "
            "It is important to note that through meticulous effort and stubborn determination, I successfully restored the vehicle to working condition. "
            "Consequently, this hands-on challenge strengthened my passion for mechanical engineering and practical troubleshooting. "
            "In conclusion, overcoming this hardship demonstrated that persistent effort can overcome complex real-world obstacles."
        )
    },
    {
        "id": "mixed_02",
        "prompt_category": "Intellectual curiosity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Analyzing the Geometry of Honeybee Communication",
        "text": (
            "My passion for scientific observation began when I constructed an observation beehive in my backyard last summer. "
            "After several hundred worker bees accidentally escaped into our sunroom, I dedicated three weeks to filming their waggle dances. "
            "Furthermore, using a macro lens and millimeter grid allowed me to plot the angular vectors of their communication patterns. "
            "Moreover, investigating why different hives adjusted their dances at varying rates led to intriguing aerodynamic discoveries. "
            "It is evident that the micro-topography of the neighboring orchard played a pivotal role in deflecting local wind currents. "
            "Additionally, this empirical investigation revealed that authentic scientific inquiry requires patience in the face of unexpected errors. "
            "In summary, I am eager to continue exploring complex biological systems through rigorous university coursework."
        )
    },
    {
        "id": "mixed_03",
        "prompt_category": "Community contribution",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Facilitating Tax Assistance for Immigrant Elders",
        "text": (
            "Navigating complex governmental tax documents is extraordinarily challenging for elderly non-native English speakers. "
            "Every Saturday at the East Side Community Center, I volunteered to assist Vietnamese elders like Mrs. Nguyen with their tax filings. "
            "Furthermore, translating bureaucratic tax instructions into clear Vietnamese helped alleviate immense anxiety for senior clients. "
            "Moreover, locating missing medical deduction receipts required meticulous attention to financial details. "
            "It is worth noting that community service is fundamentally about providing compassionate support to vulnerable populations. "
            "Consequently, receiving homemade banh cam from grateful clients reinforced my commitment to public service and social equity. "
            "In conclusion, I look forward to engaging in meaningful community advocacy throughout my collegiate career."
        )
    },
    {
        "id": "mixed_04",
        "prompt_category": "Future aspirations",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Developing Self-Healing Bio-Concrete Infrastructure",
        "text": (
            "Modern civil infrastructure suffers from structural degradation and significant carbon emission footprints during production. "
            "Inspired by historical bridge collapses, I initiated garage experiments incorporating *Bacillus pseudofirmus* into cement mixtures. "
            "Furthermore, when microcracks form in the concrete, the dormant bacteria produce limestone to seal the fractures autonomously. "
            "Moreover, mechanical testing demonstrated that my bio-resin test beams supported substantially higher structural loads. "
            "It is clear that biological materials hold transformative potential for designing resilient, sustainable urban infrastructure. "
            "Consequently, I aspire to study structural and materials engineering to pioneer sustainable building technologies. "
            "In essence, my goal is to engineer durable civil systems that protect public safety for decades to come."
        )
    },
    {
        "id": "mixed_05",
        "prompt_category": "Overcoming adversity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Overcoming Stuttering on the Theatrical Stage",
        "text": (
            "During my audition for the school production of 'The Crucible', my speech stutter caused my vocal cords to freeze entirely. "
            "Standing in front of seventy silent peers and the theater director was an acutely painful moment of vulnerability. "
            "Furthermore, after requesting my forgotten line from the stage manager, I chose to channel my anxiety into the dramatic scene. "
            "Moreover, performing the character's anger with genuine emotional intensity allowed me to overcome my speech hesitation. "
            "It is important to emphasize that courage is not the absence of fear, but the determination to persevere through embarrassment. "
            "Consequently, performing as part of the ensemble taught me to embrace my authentic voice without shame. "
            "In conclusion, this theatrical experience solidified my confidence to tackle demanding intellectual and collaborative challenges."
        )
    },
    {
        "id": "mixed_06",
        "prompt_category": "Intellectual curiosity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Exploring Mathematical Topology Through Quilting",
        "text": (
            "Examining my grandmother's handcrafted denim quilt revealed fascinating geometric properties resembling non-orientable manifolds. "
            "While studying topology, I realized that an irregular stitch along the quilt's border mimicked a Mobius band structure. "
            "Furthermore, drawing chalk coordinate graphs on the dining table allowed me to model the surface mathematically. "
            "Moreover, discussing the pattern with my grandmother illustrated how practical craftsmanship often intersects with abstract geometry. "
            "It is worth noting that elegant mathematical principles are frequently embodied in everyday physical artifacts. "
            "Additionally, this inquiry reinforced my desire to explore the dynamic intersection of theoretical math and physical reality. "
            "To conclude, I look forward to conducting advanced mathematical research during my undergraduate studies."
        )
    },
    {
        "id": "mixed_07",
        "prompt_category": "Community contribution",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Organizing Nightly Food Rescue Operations",
        "text": (
            "Addressing food waste and urban poverty requires immediate, hands-on logistical solutions within local neighborhoods. "
            "For two years, I coordinated late-night pickups of surplus bread from local bakeries to deliver to the 4th Street homeless shelter. "
            "Furthermore, unloading heavy crates in freezing temperatures provided direct nutritional assistance to dozens of shelter residents. "
            "Moreover, building personal relationships with shelter guests illuminated the complex personal stories behind homelessness. "
            "It goes without saying that authentic service involves actively listening to individuals rather than merely distributing supplies. "
            "Consequently, these nightly deliveries reinforced my dedication to creating compassionate, community-centered relief networks. "
            "In summary, I am eager to continue championing civic engagement and poverty alleviation at the university level."
        )
    },
    {
        "id": "mixed_08",
        "prompt_category": "Future aspirations",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Engineering Neural Interfaces for Memory Preservation",
        "text": (
            "Witnessing my grandfather's progressive memory loss from Alzheimer's disease catalyzed my dedication to neuroscience. "
            "In high school, I utilized open-source EEG headsets to analyze brainwave oscillations during cognitive recall tasks. "
            "Furthermore, writing Python scripts utilizing Fourier transforms allowed me to filter out noise from electroencephalogram data. "
            "Moreover, successfully identifying the P300 cognitive response wave demonstrated the feasibility of early neural diagnostics. "
            "It is evident that combining computational signal processing with neuroscience is crucial for developing therapeutic interventions. "
            "Consequently, I plan to pursue neural engineering to design non-invasive diagnostic tools for neurodegenerative diseases. "
            "In conclusion, I am determined to devote my career to preserving human cognition and combating neurological disorders."
        )
    },
    {
        "id": "mixed_09",
        "prompt_category": "Overcoming adversity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Maintaining Academic Focus in the Hospital Ward",
        "text": (
            "When my younger sister underwent emergency spinal surgery, our family spent countless hours in the hospital waiting room. "
            "Balancing emotional support for my parents with completing my rigorous AP coursework tested my mental discipline. "
            "Furthermore, writing literary essays under hospital Wi-Fi constraints required extreme focus and time management skills. "
            "Moreover, receiving positive surgical updates from the medical staff brought profound relief after hours of anxiety. "
            "It is important to recognize that resilience involves maintaining composure and fulfilling commitments during crisis situations. "
            "Consequently, this trial reinforced my ability to perform under high-stakes emotional and intellectual pressure. "
            "In summary, I am prepared to bring unwavering discipline and emotional fortitude to collegiate academic challenges."
        )
    },
    {
        "id": "mixed_10",
        "prompt_category": "Intellectual curiosity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Synthesizing Ancient Egyptian Blue Pigments",
        "text": (
            "The extraordinary chemical stability of ancient Egyptian pigments inspired me to investigate historical inorganic chemistry. "
            "I constructed a high-temperature kiln in our garage to synthesize calcium copper silicate crystals from raw minerals. "
            "Furthermore, heating copper oxide, sand, and flux to nine hundred degrees Celsius yielded brilliantly blue crystalline structures. "
            "Moreover, microscopic examination revealed that the firing ratio was the decisive factor governing chemical durability. "
            "It is evident that historical manufacturing techniques hold valuable lessons for contemporary materials science. "
            "Additionally, this independent chemical synthesis demonstrated the thrill of discovering scientific principles firsthand. "
            "To conclude, I aspire to pursue materials chemistry to develop durable, eco-friendly pigments and compounds."
        )
    },
    {
        "id": "mixed_11",
        "prompt_category": "Community contribution",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Restoring Bicycles for Neighborhood Youth",
        "text": (
            "Affordable transportation plays a crucial role in enabling youth to access educational and recreational opportunities. "
            "I established a community bicycle repair workshop in our backyard to refurbish discarded bikes for neighborhood children. "
            "Furthermore, salvaging spare parts and replacing rusted chains allowed us to provide reliable bicycles to dozens of kids. "
            "Moreover, watching young riders gain independent mobility and confidence was an exceptionally fulfilling experience. "
            "It is clear that grassroots community initiatives can create immediate positive impacts with minimal financial resources. "
            "Consequently, our repair project expanded to teach basic mechanical maintenance skills to local elementary students. "
            "In conclusion, I am excited to bring this passion for community empowerment and hands-on service to university life."
        )
    },
    {
        "id": "mixed_12",
        "prompt_category": "Future aspirations",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Advancing Hydroponic Cultivation for Space Exploration",
        "text": (
            "Sustaining long-duration human space missions requires the development of highly efficient closed-loop agricultural systems. "
            "In my bedroom closet, I designed an experimental aeroponic setup utilizing LED lighting to cultivate leafy greens without soil. "
            "Furthermore, regulating nutrient pH balances and water recycling allowed me to achieve substantial crop yield improvements. "
            "Moreover, optimizing growth parameters demonstrated that crops can thrive with ninety percent less water than traditional farming. "
            "It is undeniable that controlled-environment agriculture is vital for future interplanetary colonization and terrestrial sustainability. "
            "Consequently, I plan to study bioresource engineering to design resilient life-support systems for space exploration. "
            "In essence, I look forward to advancing agricultural biotechnology to support humanity's reach toward the stars."
        )
    },
    {
        "id": "mixed_13",
        "prompt_category": "Overcoming adversity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Overcoming Housing Instability Through Academic Focus",
        "text": (
            "Facing unexpected housing eviction during my junior year created profound personal and familial upheaval. "
            "Relocating to my aunt's basement required adjusting to crowded living conditions while maintaining my academic standards. "
            "Furthermore, studying calculus late at night with limited lighting taught me extraordinary mental concentration and resilience. "
            "Moreover, managing these difficult circumstances forged an unyielding commitment to achieving my educational ambitions. "
            "It is important to note that socioeconomic obstacles can serve to strengthen an individual's work ethic and resolve. "
            "Consequently, achieving top academic honors despite these hardships demonstrated my capacity to thrive under adversity. "
            "In conclusion, I am determined to approach higher education with gratitude, humility, and relentless perseverance."
        )
    },
    {
        "id": "mixed_14",
        "prompt_category": "Intellectual curiosity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Investigating Software Vulnerabilities in Legacy Systems",
        "text": (
            "Analyzing the mathematical and memory boundaries in vintage computer programs ignited my passion for computer security. "
            "By exploring arithmetic overflow in legacy 3D game engines, I learned how floating-point calculations impact system stability. "
            "Furthermore, utilizing disassemblers allowed me to examine assembly instructions and register memory allocations in detail. "
            "Moreover, tracing buffer overflow vulnerabilities revealed the critical importance of rigorous memory safety standards. "
            "It is evident that understanding low-level computing architectures is essential for designing secure modern software systems. "
            "Additionally, this technical exploration confirmed my desire to pursue research in computer systems and cybersecurity. "
            "To summarize, I am eager to delve into advanced computer science curricula and system security research."
        )
    },
    {
        "id": "mixed_15",
        "prompt_category": "Community contribution",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Teaching Computational Logic Through Interactive Play",
        "text": (
            "Demystifying fundamental computer science concepts for elementary students requires imaginative and inclusive pedagogical methods. "
            "At our local public library, I designed interactive logic games using sidewalk chalk to teach conditional statements and loops. "
            "Furthermore, transforming algorithmic principles into kinesthetic exercises enabled young learners of all backgrounds to participate. "
            "Moreover, witnessing struggling students master logic puzzles illustrated the empowering nature of accessible STEM education. "
            "It is worth emphasizing that fostering early confidence in problem-solving can transform a student's academic trajectory. "
            "Consequently, our library workshop series inspired several local schools to introduce coding logic into their elementary clubs. "
            "In conclusion, I hope to continue promoting equitable STEM education opportunities throughout my university years."
        )
    },
    {
        "id": "mixed_16",
        "prompt_category": "Future aspirations",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Designing Electrodynamic Tethers for Orbital Debris Removal",
        "text": (
            "The exponential accumulation of orbital debris poses a significant threat to global satellite communications and space exploration. "
            "To explore mitigation strategies, I developed an orbital mechanics simulation modeling satellite decay trajectories in Python. "
            "Furthermore, evaluating electrodynamic tethers demonstrated how planetary magnetic fields can deorbit dead CubeSats efficiently. "
            "Moreover, calibrating atmospheric drag coefficients allowed my model to match historical telemetry data accurately. "
            "It is clear that scalable remediation technologies are imperative for preserving the long-term sustainability of Earth's orbits. "
            "Consequently, I aspire to pursue aerospace engineering to design autonomous spacecraft for space debris capture. "
            "In summary, I am dedicated to safeguarding low-Earth orbit for the next generation of scientific discovery."
        )
    },
    {
        "id": "mixed_17",
        "prompt_category": "Overcoming adversity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Adapting to Instrument Damage Before Competition",
        "text": (
            "When my violin's wooden bridge sheared moments before an all-state audition, I was confronted with immediate crisis. "
            "Rather than withdrawing from the audition, I utilized an emergency clamp to secure the fractured bridge and retuned the instrument. "
            "Furthermore, compensating for altered acoustic resonance required adjusting my bowing technique and finger pressure dynamically. "
            "Moreover, delivering a passionate performance despite the compromised equipment demonstrated my adaptability under stress. "
            "It is undeniable that unexpected complications test a musician's composure and creative problem-solving capacity. "
            "Consequently, earning a regional placement taught me that artistic expression transcends ideal physical conditions. "
            "In conclusion, this performance confirmed my ability to remain calm and resourceful during high-pressure situations."
        )
    },
    {
        "id": "mixed_18",
        "prompt_category": "Intellectual curiosity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Recording Micro-Seismic Activity with Custom Sensors",
        "text": (
            "Investigating induced seismic activity in my home state motivated me to build a low-cost seismometer in our basement. "
            "Using an electromagnetic speaker coil, magnet, and microcontroller, I constructed a sensitive vibration monitoring rig. "
            "Furthermore, writing data acquisition software enabled me to record primary wave arrivals during a major regional earthquake. "
            "Moreover, analyzing forty-two micro-tremor recordings allowed me to correlate seismic events with nearby industrial fluid injections. "
            "It is evident that citizen science and custom instrumentation can yield valuable geological insights at minimal cost. "
            "Additionally, this geophysical project solidified my commitment to studying geosciences and data analysis. "
            "To conclude, I look forward to conducting advanced geophysical research during my undergraduate education."
        )
    },
    {
        "id": "mixed_19",
        "prompt_category": "Community contribution",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Archiving Regional Oral Histories and Dialects",
        "text": (
            "Preserving local linguistic heritage and cultural narratives is vital for safeguarding community memory. "
            "I partnered with our county historical society to record and digitize sixty hours of interviews with retired timber workers. "
            "Furthermore, compiling an online phonetic lexicon helped preserve unique regional idioms and historical terminology. "
            "Moreover, presenting these archival recordings at town meetings brought profound joy to local families and historians. "
            "It goes without saying that oral history projects play an essential role in connecting diverse generations. "
            "Consequently, our digital archive ensured that the working-class cultural heritage of our valley will endure permanently. "
            "In essence, I look forward to pursuing interdisciplinary humanities research and community archiving in college."
        )
    },
    {
        "id": "mixed_20",
        "prompt_category": "Future aspirations",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Developing Molecular Catalysts for Artificial Photosynthesis",
        "text": (
            "Harnessing solar radiation to produce clean hydrogen fuels is a critical frontier in global renewable energy research. "
            "In my high school laboratory, I synthesized ruthenium-based coordination complexes to catalyze photochemical water oxidation. "
            "Furthermore, optimizing reaction conditions allowed me to observe measurable hydrogen generation under ambient light exposure. "
            "Moreover, analyzing catalytic efficiency highlighted the importance of molecular stability in solar fuel conversion. "
            "It is undeniable that advancing artificial photosynthesis is essential for transitioning away from carbon-intensive fossil fuels. "
            "Consequently, I plan to pursue chemical engineering to design scalable photocatalytic systems for clean fuel generation. "
            "In conclusion, I am excited to contribute to sustainable energy breakthroughs through rigorous university scholarship."
        )
    },
    {
        "id": "mixed_21",
        "prompt_category": "Overcoming adversity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Balancing Night Shifts with Academic Rigor",
        "text": (
            "Working overnight shifts at a family laundromat while taking demanding AP courses required extraordinary personal discipline. "
            "Between cleaning machines and managing customer concerns, I utilized every quiet moment to study chemistry mechanisms. "
            "Furthermore, maintaining a high academic standing despite chronic sleep deprivation tested my physical and mental resilience. "
            "Moreover, this demanding schedule cultivated a deep appreciation for the value of persistent, disciplined effort. "
            "It is important to note that overcoming demanding circumstances strengthens an individual's character and maturity. "
            "Consequently, this experience solidified my confidence that I can manage demanding collegiate workloads successfully. "
            "In summary, I am prepared to dedicate relentless effort and focus to my undergraduate studies."
        )
    },
    {
        "id": "mixed_22",
        "prompt_category": "Intellectual curiosity",
        "class_label": 2,
        "class_name": "mixed",
        "title": "Modeling Minimal Energy Surfaces in Bubble Geometry",
        "text": (
            "The elegant geometry of soap films and minimal surface areas provided a captivating entry point into mathematical physics. "
            "By experimenting with wire frames in soapy solutions, I observed how surface tension naturally minimizes geometric boundary energy. "
            "Furthermore, developing 3D visualization algorithms enabled me to model catenoid and helicoid surfaces computationally. "
            "Moreover, calculating the precise angular junctions confirmed that physical soap films adhere strictly to mathematical laws. "
            "It is evident that the study of physical minimal surfaces yields profound insights into optimization mathematics. "
            "Additionally, this investigation fueled my desire to pursue applied mathematics and computational physics. "
            "To conclude, I am eager to participate in rigorous mathematical research during my undergraduate studies."
        )
    }
]


def generate_dataset(output_path: str = "backend/dataset/admissions_essays.json") -> dict:
    """Generate the admissions essay benchmark dataset and write to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = {
        "metadata": {
            "name": "Admissions Essay AI Detection Benchmark",
            "version": "2.0.0",
            "description": "Benchmark dataset of college admissions essays spanning Human, AI, and Mixed classes across 4 key prompts.",
            "total_samples": len(ESSAYS_DATA),
            "class_distribution": {
                "class_0_human": sum(1 for e in ESSAYS_DATA if e["class_label"] == 0),
                "class_1_ai": sum(1 for e in ESSAYS_DATA if e["class_label"] == 1),
                "class_2_mixed": sum(1 for e in ESSAYS_DATA if e["class_label"] == 2),
            },
            "prompt_categories": list({e["prompt_category"] for e in ESSAYS_DATA}),
        },
        "samples": ESSAYS_DATA,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[generate_dataset] Successfully generated {len(ESSAYS_DATA)} essay samples.")
    print(f"[generate_dataset] Saved to: {output_path}")
    print(f"[generate_dataset] Class breakdown: {dataset['metadata']['class_distribution']}")
    return dataset


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "dataset", "admissions_essays.json")
    generate_dataset(out_file)
