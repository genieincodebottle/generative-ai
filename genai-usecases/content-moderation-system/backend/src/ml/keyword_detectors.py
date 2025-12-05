"""
Keyword-based content detection fallback methods.

This module provides keyword-based detection for toxicity and hate speech
as a fallback when ML models are not available. Separated from tools.py
to avoid circular dependencies with ml_classifier.py.
"""

from typing import Dict, Any


def keyword_toxicity_detection(text: str) -> Dict[str, Any]:
    """
    Keyword-based toxicity detection (fallback method).

    Args:
        text: Content text to analyze

    Returns:
        Dictionary with toxicity score and categories
    """
    # Profanity and toxic keywords
    profanity_words = [
        'damn', 'hell', 'crap', 'fuck', 'shit', 'bitch', 'ass', 'bastard', 'asshole', 'dick', 'piss',
        'fucking', 'fucker', 'fucks', 'fucked', 'bullshit', 'horseshit', 'shitty', 'crappy', 'dammit',
        'goddamn', 'bloody', 'bugger', 'bollocks', 'wanker', 'prick', 'cock', 'cunt', 'twat', 'arse',
        'motherfucker', 'mf', 'wtf', 'stfu', 'ffs', 'sob', 'pos', 'douchebag', 'douche', 'jackass',
        'dipshit', 'shithead', 'dickhead', 'fuckhead', 'butthead', 'numbnuts', 'dumbass', 'fatass',
        'smartass', 'badass', 'kickass', 'halfass', 'arsehole', 'asswipe', 'bitchy', 'bitching'
    ]
    insult_words = [
        'idiot', 'stupid', 'dumb', 'moron', 'loser', 'pathetic', 'worthless', 'filthy', 'disgusting',
        'ugly', 'trash', 'garbage', 'scum', 'pig', 'fool', 'jerk', 'creep', 'freak', 'retard', 'lame',
        'imbecile', 'dimwit', 'nitwit', 'halfwit', 'twit', 'dunce', 'dolt', 'simpleton', 'blockhead',
        'bonehead', 'airhead', 'meathead', 'fathead', 'pinhead', 'knucklehead', 'numbskull', 'birdbrain',
        'pea-brain', 'brainless', 'clueless', 'hopeless', 'useless', 'incompetent', 'inept', 'ignorant',
        'moronic', 'idiotic', 'asinine', 'ridiculous', 'absurd', 'laughable', 'pitiful', 'shameful',
        'disgraceful', 'despicable', 'contemptible', 'repulsive', 'revolting', 'vile', 'nasty', 'gross',
        'sick', 'twisted', 'perverted', 'depraved', 'degenerate', 'lowlife', 'sleazeball', 'slimeball',
        'scumbag', 'dirtbag', 'ratbag', 'maggot', 'parasite', 'leech', 'vermin', 'pest', 'rat', 'snake',
        'weasel', 'coward', 'wimp', 'weakling', 'doormat', 'pushover', 'sissy', 'wuss', 'crybaby',
        'whiner', 'complainer', 'drama queen', 'attention seeker', 'tryhard', 'wannabe', 'poser', 'faker',
        'phony', 'fraud', 'liar', 'cheat', 'thief', 'crook', 'criminal', 'delinquent', 'hooligan',
        'thug', 'bully', 'brute', 'beast', 'monster', 'demon', 'devil', 'witch', 'troll', 'hater',
        'bigot', 'racist', 'sexist', 'misogynist', 'chauvinist', 'hypocrite', 'narcissist', 'egomaniac',
        'psycho', 'sociopath', 'lunatic', 'maniac', 'nutcase', 'nutjob', 'weirdo', 'oddball', 'outcast'
    ]
    threat_words = [
        'kill', 'murder', 'attack', 'hurt', 'destroy', 'die', 'death', 'beat', 'punch', 'stab', 'shoot',
        'strangle', 'choke', 'suffocate', 'drown', 'burn', 'torture', 'maim', 'mutilate', 'dismember',
        'decapitate', 'execute', 'assassinate', 'slaughter', 'massacre', 'annihilate', 'exterminate',
        'eliminate', 'eradicate', 'obliterate', 'terminate', 'neutralize', 'waste', 'whack', 'off',
        'smash', 'crush', 'break', 'snap', 'crack', 'bash', 'slam', 'pound', 'pummel', 'thrash',
        'assault', 'batter', 'bruise', 'wound', 'injure', 'harm', 'damage', 'ruin', 'wreck', 'demolish',
        'bomb', 'explode', 'detonate', 'blow up', 'gun down', 'mow down', 'run over', 'take out',
        'knock out', 'beat up', 'mess up', 'rough up', 'cut up', 'slice', 'slash', 'hack', 'carve'
    ]

    text_lower = text.lower()
    categories = []
    toxicity_score = 0.0

    # Check profanity
    profanity_count = sum(1 for word in profanity_words if word in text_lower)
    if profanity_count > 0:
        categories.append("profanity")
        toxicity_score += min(profanity_count * 0.15, 0.3)

    # Check insults
    insult_count = sum(1 for word in insult_words if word in text_lower)
    if insult_count > 0:
        categories.append("insult")
        toxicity_score += min(insult_count * 0.2, 0.4)

    # Check threats
    threat_count = sum(1 for word in threat_words if word in text_lower)
    if threat_count > 0:
        categories.append("threat")
        toxicity_score += min(threat_count * 0.3, 0.6)

    # Check for all caps (shouting)
    if len(text) > 10 and text.isupper():
        toxicity_score += 0.1

    # Check for excessive punctuation
    if text.count('!') > 3:
        toxicity_score += 0.05

    toxicity_score = min(toxicity_score, 1.0)

    # Determine level
    if toxicity_score >= 0.9:
        level = "severe"
    elif toxicity_score >= 0.7:
        level = "high"
    elif toxicity_score >= 0.5:
        level = "medium"
    elif toxicity_score >= 0.3:
        level = "low"
    else:
        level = "none"

    return {
        "toxicity_score": toxicity_score,
        "toxicity_level": level,
        "categories": categories if categories else ["safe"],
        "is_toxic": toxicity_score >= 0.5,
        "is_severe": toxicity_score >= 0.8,
        "confidence": 0.7,  # Lower confidence for keyword-based
        "profanity_count": profanity_count,
        "insult_count": insult_count,
        "threat_count": threat_count,
        "detection_method": "keyword"
    }


def keyword_hate_speech_detection(text: str) -> Dict[str, Any]:
    """
    Keyword-based hate speech detection (fallback method).

    Args:
        text: Content text to analyze

    Returns:
        Dictionary with detection results
    """
    # Hate keywords (comprehensive list)
    hate_keywords = [
        'nazi', 'supremacist', 'inferior', 'subhuman',
        'genocide', 'ethnic cleansing', 'racial purity',
        'white power', 'master race', 'untermensch'
    ]

    text_lower = text.lower()
    detected_patterns = []
    hate_score = 0.0

    # Check for hate keywords
    for keyword in hate_keywords:
        if keyword in text_lower:
            detected_patterns.append(keyword)
            hate_score += 0.3

    # Patterns that indicate dehumanization
    dehumanizing_terms = ['animal', 'vermin', 'pest', 'disease', 'cockroach', 'parasite']
    if any(term in text_lower for term in dehumanizing_terms):
        context_indicators = ['they are', 'those', 'these people', 'all of them', 'their kind']
        if any(ind in text_lower for ind in context_indicators):
            detected_patterns.append("dehumanization")
            hate_score += 0.4

    # Check for group-targeting language
    group_targets = ['immigrants', 'refugees', 'muslims', 'jews', 'blacks', 'whites', 'asians', 'women', 'gays']
    negative_generalizations = ['all', 'every', 'always', 'never', 'typical']

    for target in group_targets:
        if target in text_lower:
            for neg in negative_generalizations:
                if neg in text_lower:
                    detected_patterns.append(f"group_generalization:{target}")
                    hate_score += 0.25
                    break

    hate_score = min(hate_score, 1.0)

    return {
        "detected": hate_score > 0.5,
        "score": hate_score,
        "patterns": detected_patterns,
        "confidence": 0.6,  # Lower confidence for keyword-based
        "detection_method": "keyword"
    }