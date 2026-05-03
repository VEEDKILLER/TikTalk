"""
Prompt templates for generating child-appropriate images.

Design principles:
- Images must be easy to describe orally by children aged 6-12
- Contain clear objects and actions
- Avoid visual clutter & abstract concepts
- Consistent flat-illustration / children's book style
- Suitable for PSLE Stimulus-Based Conversation practice

Action lists are CATEGORY-SPECIFIC so that the filled template is always
semantically coherent. The global ACTIONS list is kept only for reference.
"""
import random

# ── Style suffix ──────────────────────────────────────────────────────────────
# Stronger directives for visual consistency across all generated images
STYLE_SUFFIX = (
    "flat vector illustration, children's picture book style, "
    "bright pastel colors, clean simple lines, "
    "uncluttered background, single focal scene, "
    "friendly cartoon characters, no text, no watermark, "
    "consistent warm lighting, child-friendly, G-rated"
)

NEGATIVE_PROMPT = (
    "realistic photo, 3D render, scary, violent, dark, gloomy, "
    "complex cluttered background, abstract, blurry, "
    "text, watermark, signature, adult content, "
    "multiple overlapping scenes, confusing composition, "
    "weapons, blood, horror"
)

# ── Category-specific actions ─────────────────────────────────────────────────
# Only actions that make sense within the category context.
# Used to fill {action} placeholders so prompts stay coherent.
CATEGORY_ACTIONS: dict[str, list[str]] = {
    "daily_life": [
        "cooking", "eating", "washing dishes", "folding clothes",
        "setting the table", "tidying up toys", "sweeping the floor",
        "watering a plant on the windowsill",
    ],
    "school": [
        "reading a book", "drawing a picture", "writing in a notebook",
        "painting on an easel", "doing homework", "raising a hand to answer",
        "presenting a project",
    ],
    "outdoor": [
        "playing with a ball", "riding a bicycle", "flying a kite",
        "playing catch", "jumping rope", "blowing bubbles",
        "collecting seashells", "skating",
    ],
    "community": [
        "buying fruits at a stall", "borrowing books", "waiting for the bus",
        "posting a letter", "queuing up", "buying a drink",
    ],
    "nature": [
        "looking at animals", "planting flowers", "watering plants",
        "watching butterflies", "feeding a bird", "picking apples",
        "looking at stars through a telescope",
    ],
    "festivals": [
        "decorating a Christmas tree", "giving red packets",
        "lighting a lantern", "opening presents",
        "making mooncakes", "watching fireworks", "wearing a costume",
    ],
    "helping": [
        "helping an elderly person cross the road",
        "picking up litter", "sharing food with a friend",
        "donating toys to a donation box", "carrying groceries for a neighbour",
        "watering plants in a community garden",
        "teaching a younger child to read",
        "holding the door open for others",
        "comforting a crying friend on a bench",
    ],
    "transportation": [
        "looking out the window", "waiting at the airport departure gate",
        "getting on a school bus", "crossing the road at a zebra crossing",
        "looking at a map at a bus interchange", "loading luggage into a car",
    ],
}

# ── Templates ─────────────────────────────────────────────────────────────────
# {character} → filled with a character string
# {action}    → filled with a CATEGORY-SPECIFIC action (never a global one)
SCENARIO_TEMPLATES: dict[str, list[str]] = {
    "daily_life": [
        "A {character} {action} in the kitchen",
        "A {character} {action} in the living room",
        "A {character} {action} at the dining table",
        "A family having breakfast together at home",
        "A child helping to wash dishes in the kitchen",
        "A {character} folding clothes on a bed",
        "A {character} feeding a pet cat at home",
        "A family watching television together on a sofa",
        "A {character} setting the table for dinner",
        "A {character} tidying up toys in the bedroom",
    ],
    "school": [
        "Children {action} in the classroom",
        "A {character} reading a book in the school library",
        "Children playing on the school playground",
        "A teacher and students in a classroom",
        "Children eating lunch in the school canteen",
        "A {character} raising a hand to answer a question in class",
        "Children doing a science experiment with test tubes",
        "A {character} painting on an easel during art class",
        "Children running a relay race on a sports field",
        "A {character} presenting a project in front of the class",
    ],
    "outdoor": [
        "A {character} {action} in the park",
        "Children playing at the playground with slides and swings",
        "A family having a picnic under a big tree",
        "A {character} riding a bicycle on a path",
        "Children flying kites in an open field",
        "A {character} feeding ducks at a pond in the park",
        "Children playing hide and seek behind trees",
        "A {character} skating on a path in the park",
        "A family walking along a beach collecting seashells",
        "Children building sandcastles on a sunny beach",
    ],
    "community": [
        "People shopping at a supermarket",
        "A {character} buying fruits at a market stall",
        "Children visiting a fire station",
        "A {character} at the bus stop waiting for a bus",
        "People at a community event in the neighbourhood",
        "A {character} borrowing books at a public library counter",
        "Children watching a lion dance performance at a festival",
        "A {character} buying a drink from a hawker centre stall",
        "People queuing up at a clinic waiting room",
        "A {character} posting a letter at a red postbox",
    ],
    "nature": [
        "A {character} looking at animals in the zoo",
        "Birds and squirrels in a garden",
        "A {character} planting flowers in a garden",
        "Fish swimming in a clear pond",
        "A {character} watching butterflies in a meadow",
        "A {character} picking apples from a tree in an orchard",
        "A rabbit and a turtle on a grassy hill",
        "A {character} holding an umbrella walking in the rain",
        "Children exploring a nature trail in a forest",
        "A {character} looking at stars through a telescope at night",
    ],
    "festivals": [
        "A family decorating a Christmas tree at home",
        "Children wearing costumes for a Halloween party",
        "A family reunion dinner with dumplings on the table",
        "A {character} lighting a lantern during Mid-Autumn Festival",
        "Children opening presents on Christmas morning",
        "A family making mooncakes together in the kitchen",
        "Children watching fireworks in the night sky",
        "A {character} giving red packets during Chinese New Year",
        "Children decorating eggs for Easter",
        "A family preparing a barbecue for National Day",
    ],
    "helping": [
        "A {character} helping an elderly person cross the road",
        "Children picking up litter in a park",
        "A {character} sharing food with a friend at school",
        "Children donating toys to a donation box",
        "A {character} carrying groceries for a neighbour",
        "Children watering plants in a community garden",
        "A {character} teaching a younger child to read",
        "Children sorting recyclable items into different bins",
        "A {character} holding the door open for others",
        "A {character} comforting a crying friend on a bench",
    ],
    "transportation": [
        "A {character} riding on an MRT train looking out the window",
        "A family waiting at an airport departure gate",
        "A {character} getting on a yellow school bus",
        "Children riding a double-decker sightseeing bus",
        "A {character} waving goodbye from a taxi",
        "A family boarding a ferry at a harbour",
        "Children crossing the road at a zebra crossing with traffic lights",
        "A {character} looking at a map at a bus interchange",
        "A family loading luggage into a car boot for a road trip",
        "A {character} riding a scooter on a park path",
    ],
}

# ── Characters ────────────────────────────────────────────────────────────────
CHARACTERS = [
    "boy", "girl", "child",
    "mother", "father", "grandmother", "grandfather",
    "young boy", "young girl", "brother and sister",
    "group of children", "little girl with pigtails",
    "boy wearing glasses", "girl in a school uniform",
]

# Legacy global actions list (kept for API compatibility only)
ACTIONS = sorted({a for actions in CATEGORY_ACTIONS.values() for a in actions})


# ── Public helpers ────────────────────────────────────────────────────────────

def build_prompt(base_prompt: str) -> str:
    """Append the standard style suffix to any prompt."""
    return f"{base_prompt}, {STYLE_SUFFIX}"


def fill_template(template: str, character: str, category: str) -> str:
    """Fill a template using a category-appropriate action.

    The {action} placeholder is filled from CATEGORY_ACTIONS[category] so
    that the resulting scene is always semantically coherent.
    """
    action = random.choice(CATEGORY_ACTIONS.get(category, ACTIONS))
    try:
        return template.format(character=character, action=action, scene="")
    except KeyError:
        return template


def pick_template(category: str, character: str) -> str:
    """Pick a random template for the category and fill it."""
    templates = SCENARIO_TEMPLATES.get(category, [])
    if not templates:
        raise ValueError(f"Unknown category '{category}'.")
    template = random.choice(templates)
    return fill_template(template, character, category)


def get_all_scenarios() -> list[str]:
    """Return all scenarios with defaults filled in (for testing)."""
    prompts = []
    for category, templates in SCENARIO_TEMPLATES.items():
        for t in templates:
            prompts.append(fill_template(t, "boy", category))
    return prompts
