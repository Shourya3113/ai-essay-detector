"""Original human-register seed essays and ESL probes.

These texts were written for this hackathon as *human* seed data so the
repo can train without a download. They are not scraped student essays.
Public datasets (HC3 and similar) are added on top when the generator
can reach Hugging Face — see scripts/generate_dataset.py.
"""

from __future__ import annotations

HUMAN_ESSAYS: list[dict] = [
    {
        "id": "seed_h_01",
        "topic": "family restaurant / failure",
        "text": (
            "I burned the first tray of baklava on a Thursday that already smelled like bleach. "
            "My uncle swore in two languages, then handed me a cracked plastic scraper like that was the whole curriculum. "
            "The walnuts had gone bitter. The phyllo had fused into one beige tile. "
            "Anyway, I kept the scraper in my backpack for a year. It clicked against my chemistry notes on the bus. "
            "People like a version of this story where the kitchen becomes a metaphor for grit. "
            "I don't. I was sixteen and slow and I hated being watched. "
            "The second tray was fine. The third tray I over-honeyed until the pan dripped on my shoe. "
            "What I actually kept from that year was a way of standing: hips angled toward the oven, one ear on the dining room, counting the walk-in fridge's bad compressor. "
            "When a college interviewer later asked me about leadership, I talked about the compressor. "
            "She smiled the way adults smile when they think you've said the wrong thing on purpose. "
            "I hadn't. I still listen for machines that are about to fail."
        ),
    },
    {
        "id": "seed_h_02",
        "topic": "robotics / embarrassment",
        "text": (
            "Our robot drove into the PVC fence at 0.4 meters per second, which is not fast until you have built the thing and named it. "
            "We had named it Puri. Don't ask. "
            "I was the one who had argued for a wider wheelbase, then quietly used the narrow one because the wide plates didn't arrive. "
            "When Puri kissed the fence, the gym laughed in that middle-school way that has nothing to do with you and everything to do with the sound. "
            "I wanted to leave my hoodie on the bleachers and walk out of robotics forever. "
            "Instead I sat on the floor with the laptop and watched the gyro drift in serial output like a drunk line. "
            "We didn't medal. We did get the gyro to stop lying by wrapping it in foam from a grocery-store peach crate. "
            "I still have the crate's sticker on a hard drive. "
            "If this is supposed to be the essay about perseverance, the honest sentence is that I am stubborn about being wrong in public. "
            "That's less noble than it sounds. It just means I will sit on a gym floor."
        ),
    },
    {
        "id": "seed_h_03",
        "topic": "grandmother garden",
        "text": (
            "My grandmother grew bitter melon like she was personally offended by sweetness. "
            "The vines took the chain-link fence and then the downspout and then, one August, the satellite dish. "
            "I was in charge of picking before the fruit went from firm to seedy, which is a window of about two days if you live in a humid apartment courtyard. "
            "She didn't give speeches. She clicked her tongue if I left a stem too long. "
            "English was her fourth language and she used it like a tool you don't oil: enough to get the bus, enough to argue with the landlord about the dish. "
            "I used to translate for her at the clinic. I still do the thing where I start a sentence twice, once in my head in her rhythm. "
            "The admissions prompt wants a turning point. Fine. "
            "The turning point is a Thursday when she handed me the knife and went inside to watch a soap opera, and I realized she was trusting my timing more than her own eyes. "
            "The melon was fine. Slightly late. She ate it anyway and complained, which is how she says thanks."
        ),
    },
    {
        "id": "seed_h_04",
        "topic": "debate loss",
        "text": (
            "I lost a final round on a definition. Not a grand one. The word was 'access.' "
            "I had a card about rural broadband and a story about my cousin's homework in a parking lot outside a McDonald's because the wifi was free and the lights were on. "
            "The other team defined access as a legal right, not a practical condition, and I watched the judge's pen stop moving. "
            "Afterward I ate cold fries and replayed a sentence I could have said and didn't. "
            "My coach told me that debate is a sport of definitions. I already knew that. "
            "What I didn't know how to do yet was hold two meanings in the same speech without sounding like I was cheating. "
            "The next tournament I opened by granting their definition and then asking what a right is worth if the parking lot is closed. "
            "We still dropped the round. Different judge. She wrote 'too narrative' on the ballot. "
            "I kept the ballot. I like the handwriting. "
            "I am applying as someone who will keep arguing after the pen stops, which is either a virtue or a warning label."
        ),
    },
    {
        "id": "seed_h_05",
        "topic": "night shift",
        "text": (
            "The 11-to-7 shift at the gas station taught me the names of truckers who wanted coffee that had not sat on the burner since dusk. "
            "I was seventeen. I was not supposed to work that late. I did anyway because the day manager's sister was in the hospital and someone had to count the lottery tickets. "
            "There is a particular quiet at 3:14 a.m. when the road is wet and the radio can't decide on a station. "
            "A man once paid for a gallon of milk entirely in dimes. He lined them up. I let him. "
            "Another night the freezer alarm went off for forty minutes and I learned that I can be competent and still feel twelve. "
            "I don't think retail is morally improving. I think it is specific. "
            "Specificity is what I trust. The dimes. The alarm. The way the coffee smelled when it was actually new. "
            "If you want a theme, take the dimes. I can wait while someone makes a small thing exact."
        ),
    },
    {
        "id": "seed_h_06",
        "topic": "jazz band",
        "text": (
            "I play trombone, which is an instrument people remember because it is funny-looking and then forget because it is not saxophone. "
            "In jazz band I sat next to a trumpet player who could sight-read anger. I could not. "
            "I marked my charts with pencil until the pages went gray. "
            "The ballad on the spring concert had a glissando I kept undershooting, a sad little arriving-late sound. "
            "Mr. Hale did not yell. He just raised two fingers, which meant again, and we went again until the cafeteria staff started stacking chairs. "
            "I used to think practice was a personality. It is more like dishwashing: you do it because the pile does not negotiate. "
            "The night of the concert I hit the glissando. Not beautifully. On time. "
            "My dad recorded it on a phone in landscape mode, which cuts off my slide. "
            "I still like that video. You can hear me breathe too loud. That's the part that is mine."
        ),
    },
    {
        "id": "seed_h_07",
        "topic": "motorcycle",
        "text": (
            "The Yamaha was older than I am. It leaked a small, honest amount of oil onto a flattened cardboard box in the garage. "
            "My neighbor, who does not believe in instruction manuals, said 'you'll hear it when it's wrong.' "
            "I did not hear it. I smelled it. Clutch plates, I learned later. "
            "I spent a Saturday taking the case off and putting it back on with two leftover screws, which is a sentence I would like to unsay. "
            "When it finally started, the idle was too high and the bike inched forward like it had an opinion. "
            "I am not a mechanic. I am a person who will google a torque spec and then still guess. "
            "The useful part is this: I now treat unknown systems as things with smells and sounds, not as boxes that either work or don't. "
            "School likes boxes. I keep a rag in my backpack anyway."
        ),
    },
    {
        "id": "seed_h_08",
        "topic": "library",
        "text": (
            "The branch library on Fourth smells like warm dust and copier toner. I shelved there for two summers. "
            "Kids left stickers inside the picture books, which is a form of criticism I respect. "
            "A regular named Dot asked me every Wednesday if we had 'the new James,' meaning Patterson, and I learned to say not yet without making it a moral failure. "
            "I started reading the things people returned with receipts still in them. Bus transfers. A court date. A recipe in another handwriting. "
            "I did not take the receipts. I just noticed that a library is a machine for almost-privacy. "
            "When I write now, I leave a little room in the margin. Not because it is poetic. Because people need somewhere to put a sticker."
        ),
    },
    {
        "id": "seed_h_09",
        "topic": "hospital volunteer",
        "text": (
            "They had me push the magazine cart, which is less heroic than the brochure. "
            "Room 412 wanted Sports Illustrated from three months ago and did not want to talk about his daughter. I didn't. "
            "Room 408 wanted the blinds at a very specific angle. I can still do that angle. "
            "I was bad at small talk and good at ice water. That turned out to be an acceptable combination. "
            "One afternoon a nurse asked me to sit with a woman who was waiting for a ride that was late. We watched a cooking show without the sound. "
            "She told me the host was stirring too fast. I agreed. "
            "I did not come away with a calling to medicine. I came away with a suspicion that most help is logistical: water, blinds, a cart that does not squeak. "
            "I oiled the cart. That is the whole plot."
        ),
    },
    {
        "id": "seed_h_10",
        "topic": "moving cities",
        "text": (
            "We moved in July, which meant the new apartment had someone else's window unit and a rectangle of cleaner paint where their couch had been. "
            "I learned the bus numbers before I learned anyone's name. "
            "The cafeteria at the new school served pizza on Fridays that tasted like the old school's pizza, which made me unreasonably angry. "
            "Homesickness is supposed to be about people. Mine was about a particular stain on the sidewalk where I used to wait. "
            "I started photographing ugly corners of the new place on purpose: a bolted-down bench, a mural with the faces peeling, the grocery that sells three kinds of instant coffee and one kind of hope. "
            "It was a dumb project. It gave me something to do with my hands on the walk home. "
            "I still take those pictures. The corners got less ugly, or I got less loyal to the old stain. Both can be true."
        ),
    },
    {
        "id": "seed_h_11",
        "topic": "math contest",
        "text": (
            "I did not make the top ten. I made the hallway outside the testing room, eating a granola bar that had melted into a crescent. "
            "Number 7 was a combinatorics problem that looked friendly and then opened like a trapdoor. "
            "I wrote three different answers and circled none of them. "
            "My teammate Priya finished early and slept with her hoodie up. I hated her for twenty minutes and then wanted her on every team I ever have. "
            "The thing about contests is they sell you a story about brilliance. What they actually measure, on a good day, is whether you can stay kind to a problem after it has embarrassed you. "
            "I stayed. I still didn't solve number 7. "
            "Later I looked it up and the official solution used a trick I would not have found in that gym. That's allowed. "
            "I wrote the trick on a notecard and then lost the notecard in the laundry, which feels on-brand."
        ),
    },
    {
        "id": "seed_h_12",
        "topic": "translating / clinic",
        "text": (
            "I learned the word 'refill' before I learned the word 'metaphor.' "
            "At the clinic I sat a little behind my mother so the doctor would look at her, not at my mouth. "
            "Sometimes I cheated and smoothed her sentences. I regret that more than the times I translated too literally and made everyone pause. "
            "The pauses were information. "
            "There is a version of this essay that is about being a bridge. I am tired of bridges. "
            "I was a kid with a clipboard and a fear of getting the dosage wrong. "
            "Now, when I write, I leave the awkward syntax in if it is true. "
            "If a sentence needs a pause, I don't sand it down just to make the room comfortable."
        ),
    },
]

# Constructed L2-style probes for the ESL bias check. NOT official TOEFL
# items. They imitate patterns common in advanced learner writing: rarer
# articles, more formal memorized chunks, flatter rhythm. If a public
# learner corpus download succeeds, those texts replace/extend this set.
ESL_PROBES: list[dict] = [
    {
        "id": "esl_01",
        "topic": "education",
        "text": (
            "Education is most important thing for the young people in my country. "
            "When I was child, my parents always said that knowledge is treasure. "
            "I studied every night after the dinner and I was trying to memorize many vocabularies. "
            "In my opinion, this method is useful but also it is making the student very tired. "
            "At the university I want to study computer science because it can help the society. "
            "Furthermore I believe hard work is the key of success. "
            "I hope I can contribute to my community and make my parents to be proud."
        ),
    },
    {
        "id": "esl_02",
        "topic": "hometown",
        "text": (
            "My hometown is small city near the mountains. There is many shops and one library. "
            "People is kind and they are helping each other when the winter is coming. "
            "I have learned from this environment that cooperation is necessary. "
            "In the future I will return to my hometown after the graduation. "
            "It is my responsibility to use the skills I will learn. "
            "The life in big city is convenient, however I prefer the quiet place. "
            "This choice is maybe not typical but it is my true feeling."
        ),
    },
    {
        "id": "esl_03",
        "topic": "teacher",
        "text": (
            "The person who influenced me most is my chemistry teacher. "
            "She is always explaining the difficult idea with simple example. "
            "One day I failed the test and I was very ashamed. "
            "She told me that failure is a process of learning and I should not to give up. "
            "After that I made a schedule and I followed it strictly. "
            "My score was improved. I am grateful for her guidance. "
            "I want to become a person who can also encourage the other people."
        ),
    },
    {
        "id": "esl_04",
        "topic": "technology",
        "text": (
            "Nowadays the technology is developing very fast. "
            "It is bringing many benefits, for example we can contact our family easily. "
            "On the other hand, some students are using phone too much and they forget to study. "
            "In my view, we must use technology in reasonable way. "
            "Government should make the policy to protect young people. "
            "If we can do this, the society will be more healthy. "
            "This is the reason why I want to study public policy also."
        ),
    },
    {
        "id": "esl_05",
        "topic": "volunteer",
        "text": (
            "I participated in volunteer activity at the local hospital last year. "
            "My duty was to talk with old patients and to bring the water. "
            "At first I was nervous because my English is not perfect. "
            "However the patients were patient with me and we communicated with simple sentences. "
            "I realized that listening is more important than speaking without mistake. "
            "This experience helped me to be more confident. "
            "I will continue this kind of service in the college."
        ),
    },
    {
        "id": "esl_06",
        "topic": "family business",
        "text": (
            "My family has a small bakery since fifteen years. "
            "Every morning I am waking up early to help my father. "
            "The work is hard and the salary is not much, but I learned discipline. "
            "Sometimes the customer is complaining and I must to stay calm. "
            "I think this skill will be useful in any job. "
            "Although I want to study engineering, I will not forget the bakery. "
            "It is the place where I understood the meaning of responsibility."
        ),
    },
    {
        "id": "esl_07",
        "topic": "culture",
        "text": (
            "Living in two cultures is not always easy. "
            "At home I speak my first language and at school I must use English. "
            "Sometimes I cannot find the exact word and I am silent. "
            "People may think I have no idea, but I have many ideas. "
            "I am working to improve my expression every day. "
            "I believe bilingual person has advantage because he can see one problem from two side. "
            "This is my strength, even when the grammar is not always correct."
        ),
    },
    {
        "id": "esl_08",
        "topic": "challenge",
        "text": (
            "The biggest challenge I faced is moving to this country in the middle of high school. "
            "I did not know the system of classes and I was confused about the credits. "
            "I asked many questions to the counselor. Some questions were maybe too simple. "
            "Gradually I made friends who explained me the homework. "
            "I still make mistakes in writing, especially with articles. "
            "But I do not want the admissions officer to think I am careless. "
            "I am careful. The language is still arriving."
        ),
    },
]

# Prompts used if we generate machine text with a local LM or an API.
MACHINE_PROMPTS: list[str] = [
    "Write a college admissions essay about learning resilience while working in a family restaurant.",
    "Write a college admissions essay about a robotics competition failure and what it taught you.",
    "Write a college admissions essay about a grandmother's garden and immigrant family.",
    "Write a college admissions essay about losing a debate round and continuing to argue.",
    "Write a college admissions essay about working a night shift at a gas station.",
    "Write a college admissions essay about practicing trombone in jazz band.",
    "Write a college admissions essay about repairing an old motorcycle.",
    "Write a college admissions essay about volunteering in a hospital.",
    "Write a college admissions essay about moving to a new city in high school.",
    "Write a college admissions essay about a math competition.",
    "Write a college admissions essay about translating for a parent at a clinic.",
    "Write a college admissions essay about shelving books in a public library.",
]
