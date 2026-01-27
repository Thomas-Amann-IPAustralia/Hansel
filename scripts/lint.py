# scripts/lint.py

import os
import re
import json
import spacy
import logging
import bisect
from typing import List, Dict, Callable, Any, Optional, Pattern
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher
from spacy.symbols import ORTH

# --- Configuration ---
MARKDOWN_DIR: str = 'scraped'  # Updated to read from the new directory
REPORT_FILE: str = 'report.json'
LOG_DIR: str = 'logs'
RULEBOOK_FILE: str = 'Trinity.json' 

# --- Setup Structured Logging ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'linter.log')),
        logging.StreamHandler()
    ]
)

# --- spaCy Model Loading ---
try:
    nlp = spacy.load("en_core_web_sm")

# --- Add special case tokens for semantic placeholders ---
    placeholder_texts = [
        "__SEMANTIC_ITALIC_START__", "__SEMANTIC_ITALIC_END__",
        "__SEMANTIC_BOLD_START__", "__SEMANTIC_BOLD_END__",
        "__SEMANTIC_CAPTION_START__", "__SEMANTIC_CAPTION_END__"
    ]
    for i in range(1, 7):
        placeholder_texts.append(f"__SEMANTIC_H{i}_START__")
        placeholder_texts.append(f"__SEMANTIC_H{i}_END__")

    for text in placeholder_texts:
        nlp.tokenizer.add_special_case(text, [{ORTH: text}])
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please ensure it's in your requirements.txt or run 'python -m spacy download en_core_web_sm'")
    exit()
# --- Global Constants ---
MONTHS = {
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
}
# --- Helper Functions ---
def get_line_number_from_offset(offset: int, line_offsets: List[int]) -> int:
    """Finds the line number for a given character offset using binary search."""
    return bisect.bisect_right(line_offsets, offset)

def _add_finding(findings: List[Dict], line_number: int, offending_text: str):
    """Helper to prevent duplicate findings for the same line and rule."""
    if not any(f['line_number'] == line_number and f['offending_text'] == offending_text.strip() for f in findings):
        findings.append({
            "line_number": line_number,
            "offending_text": offending_text.strip()
        })

# --- Heuristic Rule Implementations ---
def check_passive_voice(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Heuristic check for passive voice constructions (Rule: APS-GPC-Partsofsentences-H-009)."""
    findings = []
    for token in doc:
        if token.dep_ in ("nsubjpass", "auxpass"):
            _add_finding(findings, get_line_number_from_offset(token.sent.start_char, line_offsets), token.sent.text)
    return findings

def check_complete_sentence(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Heuristic to check for potential sentence fragments (Rule: APS-GPC-Partsofsentences-H-001)."""
    findings = []
    for sent in doc.sents:
        has_root = any(token.dep_ == "ROOT" for token in sent)
        has_subject = any("subj" in token.dep_ for token in sent)
        if not (has_root and has_subject) and len(sent.text.strip().split()) > 3:
            _add_finding(findings, get_line_number_from_offset(sent.start_char, line_offsets), sent.text)
    return findings

def check_collective_noun_agreement(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for plural verbs with typically singular collective nouns (Rule: APS-GPC-Nouns-R-004)."""
    findings = []
    collective_nouns = {"government", "committee", "crowd", "team", "family", "group", "staff"}
    plural_verbs = {"are", "were", "have", "do"}
    for token in doc:
        if token.lemma_.lower() in collective_nouns and token.head.lemma_.lower() in plural_verbs:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_hyphenated_modifier(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for unhyphenated compound modifiers before a noun (Rule: APS-GPC-Adjectives-H-002)."""
    findings = []
    for i in range(len(doc) - 2):
        token1, token2, token3 = doc[i], doc[i+1], doc[i+2]
        is_potential_compound = (token1.pos_ in ['ADJ', 'ADV']) and (token2.pos_ in ['NOUN', 'ADJ', 'VERB'])
        is_before_noun = token3.pos_ == 'NOUN'
        if is_potential_compound and is_before_noun and token2.head == token3 and token1.head == token2:
             _add_finding(findings, get_line_number_from_offset(token1.idx, line_offsets), f"{token1.text} {token2.text} {token3.text}")
    return findings

def check_that_vs_which(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for 'which' without a preceding comma, suggesting it might need to be 'that' for a restrictive clause (Rule: APS-GPC-Pronouns-H-005)."""
    findings = []
    for i, token in enumerate(doc):
        if token.text.lower() == 'which' and i > 0 and doc[i-1].text != ',':
            if token.dep_ == 'relcl':
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_missing_determiner(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for singular countable nouns used as subjects that might be missing a determiner (e.g., 'a', 'the') (Rule: APS-GPC-Nouns-H-001)."""
    findings = []
    for token in doc:
        if token.pos_ == 'NOUN' and token.tag_ == 'NN' and 'subj' in token.dep_:
            children_deps = {child.dep_ for child in token.children}
            if 'det' not in children_deps and 'poss' not in children_deps:
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_exclamation_marks(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags any use of exclamation marks in formal text (Rule: APS-GPC-Exclamationmarks-H-001)."""
    findings = []
    for token in doc:
        if token.text == '!':
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_matched_correlatives(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for mismatched correlative conjunctions like 'either/nor' or 'neither/or' (Rule: APS-GPC-Conjunctions-H-001)."""
    findings = []
    text = doc.text.lower()
    if ('either' in text and 'nor' in text) or ('neither' in text and 'or' in text):
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if ('either' in sent_text and 'nor' in sent_text) or ('neither' in sent_text and 'or' in sent_text):
                _add_finding(findings, get_line_number_from_offset(sent.start_char, line_offsets), sent.text)
    return findings

def check_prefer_english_forms(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags common Latin abbreviations that should be written in English for clarity (Rule: APS-GPC-Latinshortenedforms-H-001)."""
    findings = []
    latin_forms = {'e.g.', 'i.e.', 'etc.'}
    for token in doc:
        if token.text.lower() in latin_forms:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings
    
def check_unique_grading(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags phrases that grade the absolute adjective 'unique', such as 'very unique' (Rule: APS-GPC-Adjectives-R-002)."""
    findings = []
    graders = {"very", "more", "most", "less", "least", "extremely", "highly", "quite"}
    for i, token in enumerate(doc):
        if token.lemma_.lower() == "unique" and i > 0:
            if doc[i-1].lemma_.lower() in graders:
                _add_finding(findings, get_line_number_from_offset(doc[i-1].idx, line_offsets), f"{doc[i-1].text} {token.text}")
    return findings

def check_misplaced_only(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags the word 'only' to prompt a manual review of its placement, as it's often misplaced (Rule: APS-GPC-Typesofwords-H-002)."""
    findings = []
    for token in doc:
        if token.lemma_.lower() == "only":
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_filler_adverbs(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags common, often unnecessary, adverbs and intensifiers that can be removed for more direct writing (Rule: APS-GPC-Adverbs-H-001)."""
    findings = []
    filler_adverbs = {"very", "really", "quite", "extremely", "highly", "absolutely", "totally", "actually", "basically", "literally"}
    for token in doc:
        if token.lemma_.lower() in filler_adverbs:
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_modal_verb_to(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for the incorrect use of 'to' immediately following a modal verb (e.g., 'must to go') (Rule: APS-GPC-Verbs-R-007)."""
    findings = []
    for i in range(len(doc) - 1):
        token = doc[i]
        next_token = doc[i+1]
        if token.tag_ == 'MD' and next_token.lemma_.lower() == 'to':
            offending_phrase = f"{token.text} {next_token.text}"
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), offending_phrase)
    return findings

def check_improper_reflexive_pronoun(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for reflexive pronouns used incorrectly as a subject (e.g., 'Myself and John went...') (Rule: APS-GPC-Pronouns-H-004)."""
    findings = []
    for token in doc:
        is_reflexive = token.text.lower().endswith(('self', 'selves'))
        if is_reflexive and "subj" in token.dep_:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_a_vs_an(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for incorrect use of 'a' vs 'an' based on the following word's sound (Covers rules APS-GPC-Determiners-R-001 to R-006)."""
    findings = []
    vowel_sounds = 'aeiou'
    u_exceptions = {'university', 'universal', 'unique', 'user', 'unit'}
    h_exceptions = {'hour', 'honor', 'honour', 'honest', 'heir'}
    initialism_exceptions = {'f', 'h', 'l', 'm', 'n', 'r', 's', 'x'}

    for i in range(len(doc) - 1):
        det = doc[i]
        next_word = doc[i+1]
        
        if det.lemma_.lower() not in ['a', 'an']:
            continue

        next_word_lower = next_word.text.lower()
        starts_with_vowel_sound = False

        if next_word.pos_ == 'NOUN' and all(c.isupper() for c in next_word.text if c.isalpha()):
            if next_word_lower[0] in initialism_exceptions:
                starts_with_vowel_sound = True
        elif next_word_lower.startswith('h') and any(next_word_lower.startswith(ex) for ex in h_exceptions):
             starts_with_vowel_sound = True
        elif next_word_lower.startswith('u') and any(next_word_lower.startswith(ex) for ex in u_exceptions):
            starts_with_vowel_sound = False
        elif next_word_lower[0] in vowel_sounds:
            starts_with_vowel_sound = True

        if det.lemma_.lower() == 'an' and not starts_with_vowel_sound:
            _add_finding(findings, get_line_number_from_offset(det.idx, line_offsets), f"{det.text} {next_word.text}")
        elif det.lemma_.lower() == 'a' and starts_with_vowel_sound:
            _add_finding(findings, get_line_number_from_offset(det.idx, line_offsets), f"{det.text} {next_word.text}")
    
    return findings

    # --- New spaCy Heuristic Functions (October 2025) ---

def check_generic_organisation_reference(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags capitalized generic organisation types that are not part of a formal name.
    Rule: APS-GPC-Organisationnames-H-007
    """
    findings = []
    generic_types = {"Department", "Agency", "University", "Company", "Council", "Commission", "Authority", "Summit", "Academy"}
    for token in doc:
        if token.text in generic_types and token.i > 0 and doc[token.i-1].text.lower() == 'the':
            # Check if it's followed by "of <Proper Noun>"
            is_formal_name = False
            if token.i < len(doc) - 2 and doc[token.i + 1].text.lower() == 'of' and doc[token.i + 2].pos_ == 'PROPN':
                is_formal_name = True
            
            # If not part of a formal name, it's a generic reference and should be lowercase.
            if not is_formal_name:
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_generic_official_titles(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags capitalized official titles used in a generic context.
    Covers Rules:
    - APS-GPC-Governmentterms-H-007 (for 'Government')
    - APS-GPC-Governmentterms-H-011 (for 'Minister', 'Secretary')
    - APS-GPC-Governmentterms-H-012 (for 'Chief Justice', 'Premier', etc.)
    """
    findings = []
    # Titles that should be lowercase in generic use
    titles = {"Government", "Minister", "Secretary", "Premier", "Treasurer", "Mayor", 
              "President", "Speaker", "Chancellor", "Governor"}
              
    for token in doc:
        # Check if the token is a capitalized title
        if token.text in titles and not token.is_sent_start:
            # A title is formal (and thus correctly capitalized) if it precedes a person's name
            # or is part of a compound title like 'Prime Minister'.
            is_formal_use = False
            # Check for title followed by a proper name (e.g., "Minister Smith")
            if token.i < len(doc) - 1 and doc[token.i + 1].pos_ == 'PROPN' and doc[token.i + 1].ent_type_ == 'PERSON':
                is_formal_use = True
            # Check for compound titles like 'Prime Minister' where 'Prime' is an adjective modifying the title.
            if any(child.dep_ == 'amod' for child in token.children):
                is_formal_use = True
            
            if not is_formal_use:
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_generic_parliamentary_terms(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags incorrect capitalization of specific parliamentary terms using spaCy's Matcher.
    Covers Rules:
    - APS-GPC-Governmentterms-H-014 (Formal names must be capitalized)
    - APS-GPC-Governmentterms-H-015 (Generic terms must be lowercase)
    """
    findings = []
    matcher = Matcher(nlp.vocab)

    # Rule H-014: Formal names that MUST be capitalized.
    # We will match on case-insensitive versions and then check if the casing is correct.
    formal_names = {
        "FORMAL_PH": ("Parliament House", [{"LOWER": "parliament"}, {"LOWER": "house"}]),
        "FORMAL_PL": ("Parliamentary Library", [{"LOWER": "parliamentary"}, {"LOWER": "library"}]),
        "FORMAL_S": ("the Senate", [{"LOWER": "the"}, {"LOWER": "senate"}]),
        "FORMAL_HR": ("the House of Representatives", [{"LOWER": "the"}, {"LOWER": "house"}, {"LOWER": "of"}, {"LOWER": "representatives"}]),
    }

    # Rule H-015: Generic terms that are INCORRECTLY capitalized.
    # We will match on the exact, incorrect case-sensitive versions.
    generic_terms = {
        "GENERIC_PP": ("Parliamentary Procedures", [{"TEXT": "Parliamentary"}, {"TEXT": "Procedures"}]),
        "GENERIC_MP": ("Member of Parliament", [{"TEXT": "Member"}, {"TEXT": "of"}, {"TEXT": "Parliament"}]),
        "GENERIC_HP": ("Houses of Parliament", [{"TEXT": "Houses"}, {"TEXT": "of"}, {"TEXT": "Parliament"}]),
    }

    # Add all patterns to the matcher
    for key, (text, pattern) in formal_names.items():
        matcher.add(key, [pattern])
    for key, (text, pattern) in generic_terms.items():
        matcher.add(key, [pattern])

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        rule_name = nlp.vocab.strings[match_id]

        if rule_name.startswith("FORMAL_"):
            # This is a potential violation of Rule H-014.
            # Check if the matched text is NOT correctly capitalized.
            correct_text = formal_names[rule_name][0]
            if span.text != correct_text:
                offending_text = f"{span.text} (should be '{correct_text}')"
                _add_finding(findings, get_line_number_from_offset(span.start_char, line_offsets), offending_text)
        
        elif rule_name.startswith("GENERIC_"):
            # This is a direct violation of Rule H-015.
            # The term should be lowercase unless at the start of a sentence.
            if not span.is_sent_start:
                offending_text = f"{span.text} (should be lowercase)"
                _add_finding(findings, get_line_number_from_offset(span.start_char, line_offsets), offending_text)
    
    return findings
    
def check_capitalized_common_noun_definitions(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags acronym definitions where the full term is capitalized but contains no proper nouns.
    Rule: APS-GPC-Medicalterms-H-004
    """
    findings = []
    for i, token in enumerate(doc):
        # Heuristic to find an acronym: an opening bracket, followed by an all-caps word, then a closing bracket.
        if token.text == '(' and i < len(doc) - 2 and doc[i+1].is_upper and doc[i+2].text == ')':
            # Look backwards from the bracket to find the start of the noun phrase being defined.
            start_of_term_index = -1
            for j in range(i - 1, -1, -1):
                if doc[j].pos_ not in ('NOUN', 'ADJ', 'CCONJ', 'ADP'):
                    start_of_term_index = j + 1
                    break
            if start_of_term_index == -1 : start_of_term_index = 0

            term_span = doc[start_of_term_index:i]
            
            # Check conditions for a violation:
            # 1. The term is not at the start of a sentence.
            # 2. The first word of the term is capitalized.
            # 3. The term contains no proper nouns.
            if not term_span[0].is_sent_start and term_span[0].is_title:
                has_proper_noun = any(t.pos_ == 'PROPN' for t in term_span)
                if not has_proper_noun:
                    _add_finding(findings, get_line_number_from_offset(term_span.start_char, line_offsets), term_span.sent.text)
    return findings

def check_capitalized_the_before_country(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags the capitalized word 'The' when it appears before a country/GPE name mid-sentence.
    Rule: APS-GPC-Nationalities,peoplesandplacesoutsideAustralia-H-003
    """
    findings = []
    for token in doc:
        # Find 'The' that is not at the start of a sentence
        if token.text == "The" and not token.is_sent_start:
            # Check if the next token is a Geopolitical Entity (GPE) or part of a multi-word GPE
            if token.i < len(doc) - 1:
                next_token = doc[token.i + 1]
                if next_token.ent_type_ == 'GPE':
                    _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_improperly_cased_regions(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags capitalized compass directions used as adjectives for generic regions.
    Rule: APS-GPC-Nationalities,peoplesandplacesoutsideAustralia-H-006
    """
    findings = []
    compass_adjectives = {"Northern", "Southern", "Eastern", "Western"}
    for token in doc:
        if token.text in compass_adjectives and token.pos_ == 'ADJ':
            # Check if it modifies a GPE that isn't part of its own entity
            # e.g., "Southern Germany" is a finding, but "South Australia" is not (spaCy sees 'South' as part of the GPE)
            if token.head.ent_type_ == 'GPE' and token.ent_type_ != 'GPE':
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings
    
def check_org_the_capitalisation(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for 'The [Proper Noun]' mid-sentence (Rule: APS-GPC-Organisationnames-H-008)."""
    findings = []
    for i in range(len(doc) - 1):
        token = doc[i]
        # Find "The" that is not at the start of a sentence
        if token.text == "The" and not token.is_sent_start:
            next_token = doc[i+1]
            # Check if next token is a Proper Noun (proxy for org name)
            # or an Adjective followed by a Proper Noun
            is_org_name = False
            if next_token.pos_ == 'PROPN':
                is_org_name = True
            elif next_token.pos_ == 'ADJ' and i + 2 < len(doc) and doc[i+2].pos_ == 'PROPN':
                is_org_name = True
            
            if is_org_name:
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_org_verb_agreement(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for organization names (as subjects) using plural verbs (Rule: APS-GPC-Organisationnames-H-010)."""
    findings = []
    plural_verbs = {"are", "were", "have", "do"}
    for token in doc:
        # Find a subject that is a Proper Noun (proxy for org name)
        if token.dep_ == 'nsubj' and token.pos_ == 'PROPN':
            verb = token.head
            # Check if the verb is plural
            if verb.lemma_ in plural_verbs:
                # Avoid flagging compound subjects like "Google and Apple are..."
                has_conjunction = any(child.dep_ == 'conj' for child in token.children)
                if not has_conjunction:
                    _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_numerals_vs_words(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for spelled-out numbers 2 and above (Rule: APS-GPC-Choosingnumeralsorwords-H-001)."""
    findings = []
    # Note: The inverse (flagging '0' or '1') is handled by regex rule R-001
    for token in doc:
        # Check for tokens spaCy identifies as numeric words
        if token.pos_ == 'NUM' and token.lemma_.lower() in SPELLED_NUMS_GT_ONE:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_large_rounded_numbers(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags large, round numbers that could be written as '2.5 million' (Rule: APS-GPC-Choosingnumeralsorwords-H-009)."""
    findings = []
    for token in doc:
        if token.pos_ == 'NUM' and token.like_num and ',' in token.text:
            text = token.text.replace(',', '')
            prefix = ""
            
            # Check for a preceding currency symbol using dependency parsing
            for child in token.lefts:
                if child.pos_ == 'SYM' or child.text == '$':
                    prefix = f"{child.text} "
                    break
            
            if text.isdigit():
                val = int(text)
                suggestion = ""
                # Check for billions, rounded to nearest 100 million
                if val >= 1_000_000_000 and val % 100_000_000 == 0:
                    num_billions = val / 1_000_000_000
                    suggestion = f"{prefix}{num_billions:g} billion"
                # Check for millions, rounded to nearest 100 thousand
                elif val >= 1_000_000 and val % 100_000 == 0:
                    num_millions = val / 1_000_000
                    suggestion = f"{prefix}{num_millions:g} million"
                
                if suggestion:
                    offending_text = f"{prefix}{token.text}".strip()
                    _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), f"{offending_text} (Consider: '{suggestion}')")
    return findings

def check_aud_currency(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags use of 'A$' or 'AUD' to prompt manual review for clear-context (Rule: APS-GPC-Currency-H-001)."""
    findings = []
    for token in doc:
        # Find a number
        if token.pos_ == 'NUM' and token.like_num:
            # Check its left-hand modifiers for 'A$' or 'AUD'
            for child in token.lefts:
                if child.text in ["A$", "AUD"]:
                    offending_text = f"{child.text} {token.text}"
                    _add_finding(findings, get_line_number_from_offset(child.idx, line_offsets), offending_text)
    return findings

def check_ambiguous_dollar_sign(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags lone '$' symbols that might need 'A$' for clarity (Rule: APS-GPC-Currency-H-002)."""
    findings = []
    for i, token in enumerate(doc):
        # Find a '$' token
        if token.text == '$':
            is_qualified = False
            # Check if it's already qualified (e.g., A$, US$)
            if i > 0 and doc[i - 1].text.upper() in ['A', 'US', 'NZ', 'C']:
                is_qualified = True
            
            is_followed_by_num = False
            # Check if it's followed by a number
            if i < len(doc) - 1 and doc[i + 1].like_num:
                is_followed_by_num = True

            # If it's a lone $ followed by a number, flag it
            if is_followed_by_num and not is_qualified:
                # Get a slightly wider context
                start_char = max(0, token.idx - 10)
                end_char = min(len(doc.text), doc[i + 1].idx + doc[i + 1].n_chars + 10)
                offending_text = doc.text[start_char:end_char]
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), f"...{offending_text}...")
    return findings

def check_year_span_words(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags 'YYYY–YYYY' spans, preferring 'from YYYY to YYYY' in general text (Rule: APS-GPC-Datesandtime-H-001)."""
    findings = []
    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i], doc[i+1], doc[i+2]
        
        # Check for YYYY–YYYY (four-digit year, en-dash, four-digit year)
        if t1.like_num and len(t1.text) == 4 and t2.text == '–' and t3.like_num and len(t3.text) == 4:
            # This is a general span. Check if it's an exception (H-002)
            is_exception = False
            if i < len(doc) - 4:
                t4 = doc[i+3]
                t5 = doc[i+4]
                # Exception: '2020–2021 financial year'
                if t4.lemma_ in ['financial', 'calendar'] and t5.lemma_ == 'year':
                    is_exception = True
            
            if not is_exception:
                 _add_finding(findings, get_line_number_from_offset(t1.idx, line_offsets), f"{t1.text}{t2.text}{t3.text}")
    return findings

def check_financial_year_dash(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags 'YYYY to YY' or 'YYYY-YY' for financial/calendar years, preferring 'YYYY–YY' (Rule: APS-GPC-Datesandtime-H-002)."""
    findings = []
    for i in range(len(doc) - 3):
        t1, t2, t3, t4 = doc[i], doc[i+1], doc[i+2], doc[i+3]
        
        # Check for YYYY (to|-) (YY|YYYY)
        is_year_span = (t1.like_num and len(t1.text) == 4 and
                        (t2.lemma_ == 'to' or t2.text == '-') and
                        t3.like_num and (len(t3.text) == 2 or len(t3.text) == 4))
        
        # Check if it's explicitly a financial or calendar year
        is_specified_type = t4.lemma_ in ['financial', 'calendar']
        
        if is_year_span and is_specified_type:
            offending_text = f"{t1.text} {t2.text} {t3.text} {t4.text}"
            _add_finding(findings, get_line_number_from_offset(t1.idx, line_offsets), offending_text)
    return findings

def check_day_month_span_words(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags 'D–D Month' spans, preferring 'from D to D Month' (Rule: APS-GPC-Datesandtime-H-003)."""
    findings = []
    for i in range(len(doc) - 3):
        t1, t2, t3, t4 = doc[i], doc[i+1], doc[i+2], doc[i+3]
        
        # Check for [Num]–[Num] [Month]
        if t1.like_num and t2.text == '–' and t3.like_num and t4.text in MONTHS:
            offending_text = f"{t1.text}{t2.text}{t3.text} {t4.text}"
            _add_finding(findings, get_line_number_from_offset(t1.idx, line_offsets), offending_text)
    return findings

def check_worded_decimals(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags decimals written as words (e.g., 'seven point five'), preferring numerals (e.g., '7.5') (Rule: APS-GPC-Fractionsanddecimals-H-001)."""
    findings = []
    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i], doc[i+1], doc[i+2]
        
        # Check for [NumWord] point [NumWord]
        # spaCy tags 'seven', 'five', etc. as 'NUM'
        if t1.pos_ == 'NUM' and t2.lemma_ == 'point' and t3.pos_ == 'NUM':
            offending_text = f"{t1.text} {t2.text} {t3.text}"
            _add_finding(findings, get_line_number_from_offset(t1.idx, line_offsets), offending_text)
    return findings

def check_prose_ordinals_for_steps(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags ordinals ('first', 'secondly') used in prose that might be better as a numbered list (Rule: APS-GPC-Ordinalnumbers-H-001)."""
    findings = []
    # 'thirdly' is banned by a regex rule, but we include it for completeness
    ordinal_lemmas = {'first', 'second', 'third', 'firstly', 'secondly', 'thirdly'}
    
    for i, token in enumerate(doc):
        if token.lemma_.lower() in ordinal_lemmas:
            # Flag 'Firstly, ...' or 'Secondly, ...'
            if token.lemma_.endswith('ly'):
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
            
            # Flag 'First, ...' when used as an adverb
            elif token.pos_ == 'ADV':
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)

            # Flag 'The first step is...'
            elif token.pos_ == 'ADJ' and i < len(doc) - 1 and doc[i+1].lemma_ == 'step':
                offending_text = f"{token.text} {doc[i+1].text}"
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), offending_text)
    return findings

def check_compound_clause_comma(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for missing commas before conjunctions joining independent clauses with different subjects (Rule: APS-GPC-Partsofsentences-H-005)."""
    findings = []
    conjunctions = {'and', 'or', 'but', 'so'}
    
    for token in doc:
        # 1. Find a coordinating conjunction
        if token.lemma_ in conjunctions and token.dep_ == 'cc':
            verb1 = token.head
            # Ensure we are coordinating verbs/clauses, not nouns
            if verb1.pos_ != 'VERB': 
                continue 

            # 2. Find the second clause (the conjunct)
            verb2 = None
            for child in verb1.children:
                if child.dep_ == 'conj':
                    verb2 = child
                    break
            
            if verb2 and verb2.pos_ == 'VERB':
                # 3. Find subject of first clause
                subj1 = None
                for child in verb1.children:
                    if 'subj' in child.dep_:
                        subj1 = child
                        break
                
                # 4. Find subject of second clause
                subj2 = None
                for child in verb2.children:
                    if 'subj' in child.dep_:
                        subj2 = child
                        break
                
                # 5. If both subjects exist, are different, and there's no comma
                if subj1 and subj2 and subj1.lemma_.lower() != subj2.lemma_.lower():
                    if doc[token.i - 1].text != ',':
                        _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings
    
def check_comma_with_shared_subject(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for a comma before a coordinating conjunction (e.g., and, but)
    when the subject of both clauses is the same.
    (Rule: APS-GPC-Partsofsentences-H-006)
    """
    findings = []
    for token in doc:
        # Find a coordinating conjunction
        if token.dep_ == 'cc' and token.i > 0:
            conj = token
            verb2 = conj.head
            
            # Find the subject of the second clause
            subj2_list = [c for c in verb2.children if 'nsubj' in c.dep_]
            
            # Find the first clause it's coordinating with
            verb1_list = [c for c in verb2.conjuncts if c.i < conj.i]

            # If verb2 has no subject (subj2_list is empty) and verb1 exists
            if not subj2_list and verb1_list:
                verb1 = verb1_list[0]
                # Find the subject of the first clause
                subj1_list = [c for c in verb1.children if 'nsubj' in c.dep_]

                # If verb1 has a subject (meaning it's a shared subject)
                if subj1_list:
                    # Check if there is a comma right before the conjunction
                    if doc[conj.i - 1].text == ',':
                        _add_finding(findings, get_line_number_from_offset(conj.sent.start_char, line_offsets), conj.sent.text)
    return findings

def check_prefer_active_voice(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Heuristic check to flag passive voice constructions, as active voice is preferred.
    (Rule: APS-GPC-Partsofsentences-H-008)
    """
    findings = []
    # This rule is the preference, H-009 is the exception.
    # The check is identical: find passive voice.
    for token in doc:
        if token.dep_ in ("nsubjpass", "auxpass"):
            _add_finding(findings, get_line_number_from_offset(token.sent.start_char, line_offsets), token.sent.text)
    return findings

def check_passive_missing_agent(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for passive voice sentences that are missing an agent ('by' phrase).
    (Rule: APS-GPC-Partsofsentences-H-010)
    """
    findings = []
    for sent in doc.sents:
        is_passive = False
        has_agent = False
        for token in sent:
            if token.dep_ in ("nsubjpass", "auxpass"):
                is_passive = True
            if token.dep_ == 'agent':
                has_agent = True
        
        if is_passive and not has_agent:
            _add_finding(findings, get_line_number_from_offset(sent.start_char, line_offsets), sent.text)
    return findings

def check_possessive_in_compound(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for possessive markers ('s) that are not at the end of a
    hyphenated compound (e.g., "commander's-in-chief").
    (Rule: APS-GPC-Phrases-H-001)
    """
    findings = []
    for token in doc:
        # Find a hyphen used as punctuation
        if token.text == '-' and token.dep_ == 'punct' and token.i > 0:
            prev_token = doc[token.i - 1]
            # Check if the token *before* the hyphen has a possessive 's
            if prev_token.text.endswith(("'s", "’s")):
                 _add_finding(findings, get_line_number_from_offset(prev_token.sent.start_char, line_offsets), prev_token.sent.text)
    return findings

def check_phrase_fragments(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Heuristic to check for potential sentence fragments that lack a subject or finite verb.
    (Rule: APS-GPC-Phrases-H-002)
    """
    findings = []
    # This is logically identical to APS-GPC-Partsofsentences-H-001
    for sent in doc.sents:
        has_root = any(token.dep_ == "ROOT" for token in sent)
        has_subject = any("subj" in token.dep_ for token in sent)
        
        # Avoid flagging short headings or list items
        if not (has_root and has_subject) and len(sent.text.strip().split()) > 3:
            _add_finding(findings, get_line_number_from_offset(sent.start_char, line_offsets), sent.text)
    return findings

def check_descriptive_apostrophe(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags plural possessives (e.g., 'visitors' book') which are often
    incorrectly used for descriptive phrases ('visitors book').
    (Rule: APS-GPC-Apostrophes-H-001)
    """
    findings = []
    for token in doc:
        # Find a token that is a possessive modifier
        if token.dep_ == 'poss':
            # Flag if it's a plural possessive (ends in s' or s’)
            if token.text.endswith(("s'", "s’")):
                _add_finding(findings, get_line_number_from_offset(token.sent.start_char, line_offsets), token.sent.text)
    return findings

def check_nested_parentheses(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Heuristic check for nested round parentheses (Rule: APS-GPC-Bracketsandparentheses-H-003).
    Flags a '(' if its governing word is itself governed by a word that also has a '('.
    """
    findings = []
    for token in doc:
        # Find an opening parenthesis
        if token.text == '(':
            head = token.head
            # Check if the head's ancestor (head.head) also has a parenthesis as a child.
            # This implies the current parenthesis is nested within another.
            if head.head != head:  # Ensure we are not at the root
                ancestor = head.head
                for child in ancestor.children:
                    if child.text == '(' and child.dep_ == 'punct':
                        _add_finding(
                            findings,
                            get_line_number_from_offset(token.idx, line_offsets),
                            token.sent.text
                        )
                        break  # Found nesting, no need to check other children
    return findings

def check_round_brackets_in_quotes(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Heuristic check for round brackets inside quotes (Rule: APS-GPC-Bracketsandparentheses-H-004).
    Flags '()' which should likely be '[]' for insertions.
    """
    findings = []
    for token in doc:
        # Find an opening parenthesis
        if token.text == '(':
            head = token.head
            in_quote = False
            
            # Check if the head or its direct ancestors are associated with quote punctuation
            current = head
            while current.head != current:  # Traverse up the tree
                if any(t.is_quote and t.dep_ == 'punct' for t in current.children):
                    in_quote = True
                    break
                current = current.head
            
            # Check root as well
            if not in_quote and any(t.is_quote and t.dep_ == 'punct' for t in current.children):
                in_quote = True

            if in_quote:
                _add_finding(
                    findings,
                    get_line_number_from_offset(token.idx, line_offsets),
                    token.sent.text
                )
    return findings

def check_comma_after_short_intro(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Heuristic check for commas after short introductory phrases (Rule: APS-GPC-Commas-H-001).
    Flags commas after introductory phrases (advcl, prep) of 3 or fewer tokens
    in short sentences (10 or fewer tokens).
    """
    findings = []
    for sent in doc.sents:
        sent_len_tokens = len(sent)
        if sent_len_tokens > 10:  # Rule applies to "very short sentences"
            continue

        root = sent.root
        for child in root.children:
            # Check for introductory adverbial clauses or prepositional phrases
            if child.dep_ in ('advcl', 'prep') and child.i < root.i:
                intro_span = doc[child.left_edge.i : child.right_edge.i + 1]
                intro_len_tokens = len(intro_span)
                
                # Check if it's "short" (e.g., 3 tokens or less)
                if intro_len_tokens <= 3:
                    # Check if it's followed by a comma
                    if intro_span.end < len(doc) and doc[intro_span.end].text == ',':
                        _add_finding(
                            findings,
                            get_line_number_from_offset(doc[intro_span.end].idx, line_offsets),
                            sent.text
                        )
    return findings

def check_appositive_commas(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Heuristic check for missing commas around non-essential appositives (Rule: APS-GPC-Commas-H-002).
    Flags appositives (dep_ == 'appos') that are not enclosed by commas.
    """
    findings = []
    for token in doc:
        if token.dep_ == 'appos':
            # Get the full span of the appositive phrase
            appos_span = doc[token.left_edge.i : token.right_edge.i + 1]
            
            # Check for comma *before* the appositive
            has_comma_before = (appos_span.start == 0) or doc[appos_span.start - 1].text == ','
            
            # Check for comma *after* (or end of sentence)
            is_at_sent_end = appos_span.end == appos_span.sent.end
            has_comma_after = is_at_sent_end or (appos_span.end < len(doc) and doc[appos_span.end].text == ',')

            # If either comma is missing (and it's not at the start/end of sentence where one is normal)
            if not has_comma_before or not has_comma_after:
                 # Avoid flagging if the appositive is just a single token at the start of a sentence
                if appos_span.start == appos_span.sent.start and len(appos_span) == 1:
                    continue
                    
                _add_finding(
                    findings,
                    get_line_number_from_offset(appos_span.start_char, line_offsets),
                    appos_span.sent.text
                )
    return findings

def check_for_example_commas(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Heuristic check for commas around 'for example' (Rule: APS-GPC-Commas-H-004).
    Uses spaCy's Matcher to find the phrase and then checks token properties.
    """
    findings = []
    matcher = Matcher(nlp.vocab)
    # Note: This rule is for "for example", not "e.g." (which has a different rule).
    pattern = [{"LOWER": "for"}, {"LOWER": "example"}]
    matcher.add("FOR_EXAMPLE", [pattern])

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        sent = span.sent

        try:
            # Case 1: At the start of a sentence
            if span.start == sent.start:
                # Check for missing comma *after*
                if end < len(doc) and doc[end].text != ',':
                    _add_finding(findings, get_line_number_from_offset(span.start_char, line_offsets), sent.text)
            
            # Case 2: In the middle of a sentence
            elif span.start > sent.start:
                # Check for comma *before*
                has_comma_before = doc[start - 1].text == ','
                
                # Check for comma *after* (or end of sentence)
                is_at_sent_end = end == sent.end
                has_comma_after = is_at_sent_end or (end < len(doc) and doc[end].text == ',')

                if not has_comma_before or not has_comma_after:
                    _add_finding(findings, get_line_number_from_offset(span.start_char, line_offsets), sent.text)
        
        except IndexError:
            # Handle edge cases where the span is at the very end of the doc
            continue
            
    return findings
    
def check_oxford_comma(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for the presence of an Oxford (serial) comma.
    Rule: APS-GPC-Commas-H-006 (Prefer no Oxford comma)
    Rule: APS-GPC-Commas-H-007 (Exception: Use for clarity)
    
    This function flags all instances of a comma before a final conjunction
    in a list of 3+ items. The user must then decide if it's required
    for clarity per H-007.
    """
    findings = []
    for i in range(1, len(doc) - 1):
        token = doc[i]
        prev_token = doc[i - 1]

        # Find a comma followed by a coordinating conjunction (e.g., ", and")
        if prev_token.text == ',' and token.pos_ == 'CCONJ':
            
            # Find the true head of the list.
            # 'token.head' is the item the 'cc' is attached to (e.g., 'pears' in 'A, pears, and oranges')
            # We need to traverse up to find the final item ('oranges')
            list_head = token.head
            while list_head.dep_ == 'conj':
                list_head = list_head.head
            
            # Get all items in the list (the final item + its conjuncts)
            all_items = [list_head] + list(list_head.conjuncts)
            
            # If the list has 3 or more items, this is an Oxford comma.
            if len(all_items) > 2:
                    _add_finding(findings, get_line_number_from_offset(prev_token.idx, line_offsets), 
                                 prev_token.sent.text)
    return findings

def check_en_dash_spans(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for en dashes used for spans, which should be words in general content.
    Rule: APS-GPC-Dashes-H-001 (Prefer 'from...to' or 'between...and')
    Rule: APS-GPC-Dashes-H-002 (Exception: En dash OK for technical spans)
    
    This check flags en dashes between numbers, reminding the user of the
    preference for words unless it's a technical context.
    """
    findings = []
    for token in doc:
        # The en dash '–' (U+2013)
        if token.text == '–':
            # Check if it's between two tokens that are numbers
            if token.i > 0 and token.i < len(doc) - 1:
                prev_token = doc[token.i - 1]
                next_token = doc[token.i + 1]
                
                # Use spaCy's 'like_num' attribute to check for numerals
                if prev_token.like_num and next_token.like_num:
                    
                    # We assume rule APS-GPC-Dashes-R-002 (a regex rule)
                    # already catches "from 2020–2021", so we just flag the dash.
                    span_text = f"{prev_token.text} {token.text} {next_token.text}"
                    _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), 
                                 span_text)
    return findings

def check_word_slashes(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for forward slashes used to join words (e.g., 'Sydney/Brisbane').
    Rule: APS-GPC-Forwardslashes-H-003 (Don't join words with slashes)
    
    This rule flags slashes between two words, focusing on Proper Nouns
    (like the example) and common Nouns, while attempting to exclude
    allowed uses like units ('km/h') or dates ('d/m/y').
    """
    findings = []
    for token in doc:
        if token.text == '/':
            if token.i > 0 and token.i < len(doc) - 1:
                prev_token = doc[token.i - 1]
                next_token = doc[token.i + 1]

                # 1. Exclude allowed patterns
                # Exclude dates (d/m/y) or numbers (1/2)
                if prev_token.like_num or next_token.like_num:
                    continue
                # Exclude 'and/or'
                if prev_token.lemma_ == 'and' and next_token.lemma_ == 'or':
                    continue
                # Exclude common units (e.g., km/h, m/s). This is a heuristic.
                if (prev_token.pos_ == 'NOUN' and len(prev_token.text) < 3 and
                    next_token.pos_ == 'NOUN' and len(next_token.text) < 3):
                    continue

                # 2. Find disallowed patterns
                # Target Proper Noun pairs as in the 'Sydney/Brisbane' example
                is_propn_pair = prev_token.pos_ == 'PROPN' and next_token.pos_ == 'PROPN'
                
                # Target other common word pairs (Noun/Noun, Adj/Adj, etc.)
                is_word_pair = prev_token.is_alpha and next_token.is_alpha
                
                if is_propn_pair or (is_word_pair and prev_token.pos_ in ('NOUN', 'ADJ') and next_token.pos_ in ('NOUN', 'ADJ')):
                    slash_text = f"{prev_token.text}{token.text}{next_token.text}"
                    _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets),
                                 slash_text)
    return findings

def check_semicolon_usage(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for multiple semicolon rules:
    - H-001: Avoids linking two complete sentences (stylistic preference).
    - H-002: Flags if either side of the semicolon is not a complete sentence.
    - H-003: Ignores semicolons used correctly in complex lists (items with commas).
    """
    
    # --- Helper Function ---
    def _span_is_complete(span: Span) -> bool:
        """
        Checks if a span is a 'complete sentence' (has ROOT and subj,
        or is a valid imperative).
        """
        if not span or len(span.text.strip()) == 0:
            return False
            
        # Re-parse just the span to get its internal dependency structure
        clause_doc = nlp(span.text) 
        has_root = False
        has_subject = False
        is_imperative = False

        for token in clause_doc:
            if token.dep_ == "ROOT":
                has_root = True
                # Check for imperative verb (e.g., "Do this.")
                if token.pos_ == "VERB" and token.tag_ == "VB":
                    is_imperative = True
            if "subj" in token.dep_:
                has_subject = True
        
        # A clause is "complete" if it has a subject+verb or is imperative
        return (has_root and has_subject) or (has_root and is_imperative)
    # --- End Helper ---

    findings_H001 = [] # Rule: APS-GPC-Semicolons-H-001
    findings_H002 = [] # Rule: APS-GPC-Semicolons-H-002
    
    for sent in doc.sents:
        semicolons = [t for t in sent if t.text == ';']
        if not semicolons:
            continue

        # Split the sentence into parts based on the semicolons
        parts = []
        start_idx = sent.start
        for sc_token in semicolons:
            parts.append(doc[start_idx : sc_token.i])
            start_idx = sc_token.i + 1
        parts.append(doc[start_idx : sent.end]) # Add the last part

        # Rule H-003 Check: Is this a complex list?
        # Our heuristic: if *any* of the semicolon-separated parts
        # contains its own comma, we assume it's a complex list and is allowed.
        is_complex_list = any(',' in part.text for part in parts)
        if is_complex_list:
            continue # This usage is allowed per H-003, so we don't flag it.

        # If it's NOT a complex list, it must be linking clauses.
        # Now we check rules H-001 and H-002.
        for i in range(len(parts) - 1):
            left_part = parts[i]
            right_part = parts[i+1]
            sc_token = semicolons[i] # The token separating these parts
            
            left_is_complete = _span_is_complete(left_part)
            right_is_complete = _span_is_complete(right_part)
            
            line_num = get_line_number_from_offset(sc_token.idx, line_offsets)

            if left_is_complete and right_is_complete:
                # Rule H-001: Both are complete sentences. This is grammatically
                # correct, but the style guide says to avoid it.
                _add_finding(findings_H001, line_num, sent.text)
            else:
                # Rule H-002: One or both sides are fragments. This is
                # grammatically incorrect.
                _add_finding(findings_H002, line_num, sent.text)

    # Return findings specific to each rule
    rule_id = "APS-GPC-Semicolons-H-001"
    if rule_id in HEURISTIC_CHECKS and HEURISTIC_CHECKS[rule_id] == check_semicolon_usage:
        return findings_H001
        
    rule_id = "APS-GPC-Semicolons-H-002"
    if rule_id in HEURISTIC_CHECKS and HEURISTIC_CHECKS[rule_id] == check_semicolon_usage:
        return findings_H002

    # Default return (should be empty if H-003 was called, which it won't be)
    return []

def check_gov_acronyms(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags subsequent use of government-like acronyms after their first definition,
    prompting for the use of a generic descriptor per APS-GPC-Acronymsandinitialisms-H-005.
    """
    findings = []
    # Keywords to identify a "government organisation"
    gov_keywords = {"department", "agency", "authority", "commission", "office", "bureau", "government", "service"}
    
    # Stores { "ACRONYM_STR": first_definition_end_char_offset }
    defined_gov_acronyms: Dict[str, int] = {}

    # Use spaCy's Matcher to find "Proper Noun(s) (ACRONYM)"
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"POS": "PROPN", "OP": "+"}, # One or more proper nouns
        {"TEXT": "("},
        {"IS_UPPER": True, "LENGTH": {">=": 2}}, # An all-caps acronym
        {"TEXT": ")"}
    ]
    matcher.add("ACRONYM_DEF", [pattern])

    matches = matcher(doc)
    
    for match_id, start, end in matches:
        span = doc[start:end]           # The full "Name (ACRONYM)"
        full_name_span = doc[start:end-3] # The "Name" part
        acronym_token = doc[end-2]        # The "ACRONYM" part
        
        acronym = acronym_token.text
        full_name = full_name_span.text

        # If it's a "government-like" name and not already defined...
        if any(kw in full_name.lower() for kw in gov_keywords):
            if acronym not in defined_gov_acronyms:
                 # Store the *first* definition's end character offset
                defined_gov_acronyms[acronym] = span.end_char

    if not defined_gov_acronyms:
        return [] # No relevant acronyms were defined

    # Now, scan the entire doc for uses *after* that first definition
    for token in doc:
        acronym = token.text
        if acronym in defined_gov_acronyms:
            definition_end_offset = defined_gov_acronyms[acronym]
            # If this token appears *after* its definition...
            if token.idx > definition_end_offset:
                # ...flag it as a subsequent use.
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    
    return findings

def check_etc_in_lists(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for 'etc.' used in lists introduced by 'such as', 'for example', 
    or 'including' (Rule: APS-GPC-Latinshortenedforms-H-003).
    """
    findings = []
    
    # Use Matcher to find the introductory phrases
    matcher = Matcher(nlp.vocab)
    matcher.add("LIST_INTRO", [
        [{"LOWER": "such"}, {"LOWER": "as"}],
        [{"LOWER": "for"}, {"LOWER": "example"}],
        [{"LOWER": "including"}]
    ])
    
    # Find all sentences that contain one of these introductory phrases
    list_intro_sents = set()
    for match_id, start, end in matcher(doc):
        list_intro_sents.add(doc[start].sent)

    if not list_intro_sents:
        return [] # No such sentences found, so no violations possible

    # Now, only check for 'etc.' within those specific sentences
    for sent in list_intro_sents:
        for token in sent:
            # Check for "etc." or "etc"
            if token.text.lower() == 'etc.' or token.lemma_ == 'etc':
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), sent.text)
                break # Only need to flag the sentence once
    
    return findings

def check_weak_adjectives(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags common 'filler' or weak adjectives that could be replaced with more specific language.
    (Rule: APS-GPC-Adjectives-H-001)
    """
    findings = []
    # This list is subjective but targets common words that weaken professional writing.
    weak_adjectives = {
        "good", "bad", "big", "small", "nice", "great", "lovely", "amazing", 
        "important", "significant", "wonderful", "terrible"
    }
    for token in doc:
        if token.lemma_.lower() in weak_adjectives and token.pos_ == 'ADJ':
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_predicate_hyphenation(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for hyphenated compound modifiers in the predicate (after the noun), 
    where they should often be open. e.g., "The report was well-written." -> "well written".
    (Rule: APS-GPC-Adjectives-H-003)
    """
    findings = []
    linking_verbs = {'be', 'seem', 'appear', 'become', 'feel', 'look', 'sound', 'taste'}
    for token in doc:
        # Find a hyphenated token that is an adjective
        if '-' in token.text and token.pos_ == 'ADJ':
            # Check if it's in a predicate position (adjectival complement or attribute)
            if token.dep_ in ('acomp', 'attr'):
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
            # Also check if it follows a linking verb, as it's likely in the predicate
            elif token.head.lemma_ in linking_verbs and token.head.pos_ == 'VERB':
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_adjective_strings(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags strings of 2+ adjectives before a noun that are not separated by a comma.
    This prompts manual review for correct adjective order (H-004) and 
    comma use for coordinate adjectives (H-005).
    e.g., Flags "a big red car" (correct, no comma) and "a long tedious meeting" (incorrect, needs comma)
    for review.
    (Rules: APS-GPC-Adjectives-H-004, APS-GPC-Adjectives-H-005)
    """
    findings = []
    for noun in doc:
        if noun.pos_ not in ('NOUN', 'PROPN'):
            continue
        
        # Find all adjectival modifiers (amod) that come *before* this noun
        child_adjs = [c for c in noun.children if c.dep_ == 'amod' and c.i < noun.i]
        if len(child_adjs) < 2:
            continue

        # Sort them by their position in the sentence
        child_adjs.sort(key=lambda x: x.i)

        for i in range(len(child_adjs) - 1):
            adj1 = child_adjs[i]
            adj2 = child_adjs[i+1]
            
            # Check if the token directly after adj1 is adj2.
            # If it is, there is no comma (or other token) between them.
            if adj1.i + 1 == adj2.i:
                # Reconstruct the phrase for context
                phrase_span = doc[adj1.i : noun.i + 1]
                _add_finding(findings, get_line_number_from_offset(adj1.idx, line_offsets), phrase_span.text)
    return findings

def check_adjective_as_adverb(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for adjectives used as adverbs (e.g., 'he runs quick' instead of 'he runs quickly').
    (Rule: APS-GPC-Adverbs-H-002)
    """
    findings = []
    for token in doc:
        # Find an adjective (ADJ) that is functioning as an adverbial modifier (advmod)
        # for a verb (VERB).
        if token.pos_ == 'ADJ' and token.dep_ == 'advmod' and token.head.pos_ == 'VERB':
            phrase = f"{token.head.text} {token.text}"
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), phrase)
    return findings

def check_missing_italics_for_works(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Flags named entities identified as 'WORK_OF_ART' (books, films, etc.) 
    that are not italicised in the Markdown.
    (Rule: APS-GPC-Italics-H-001)
    """
    findings = []
    for ent in doc.ents:
        # Check for entities spaCy classifies as 'WORK_OF_ART'
        if ent.label_ != 'WORK_OF_ART':
            continue
        
        # Check for surrounding italics characters (* or _) in the raw doc
        is_italicised = False
        if ent.start > 0 and ent.end < len(doc):
            token_before = doc[ent.start - 1]
            token_after = doc[ent.end] # ent.end is the index *after* the last token
            
            # Check if the entity is wrapped in *...* or _..._
            if token_before.text in ('*', '_') and token_after.text in ('*', '_'):
                is_italicised = True
        
        if not is_italicised:
            _add_finding(findings, get_line_number_from_offset(ent.start_char, line_offsets), ent.text)
    return findings

def check_punctuation_in_structural_tags(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for full stops inside headings or captions.
    (Rules: APS-GPC-Punctuationandcapitalisation-H-001)
    """
    findings = []
    in_tag_type = None
    tag_start_token = None

    for token in doc:
        if token.text.startswith('__SEMANTIC_H') and token.text.endswith('_START__'):
            in_tag_type = 'heading'
            tag_start_token = token
        elif token.text.startswith('__SEMANTIC_CAPTION') and token.text.endswith('_START__'):
            in_tag_type = 'caption'
            tag_start_token = token
        elif (token.text.startswith('__SEMANTIC_H') and token.text.endswith('_END__')) or \
             (token.text.startswith('__SEMANTIC_CAPTION') and token.text.endswith('_END__')):
            in_tag_type = None
            tag_start_token = None
        
        if in_tag_type and token.text == '.':
            # Find the full text of the heading/caption from the start token
            sent_span = token.sent
            offending_text = ""
            in_span = False
            for t in sent_span:
                if t == tag_start_token:
                    in_span = True
                if in_span:
                    offending_text += t.text_with_ws
                if (in_tag_type == 'heading' and t.text.startswith('__SEMANTIC_H') and t.text.endswith('_END__')) or \
                   (in_tag_type == 'caption' and t.text.startswith('__SEMANTIC_CAPTION') and t.text.endswith('_END__')):
                    break
            
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), offending_text)
    return findings

def check_italic_case(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for potential incorrect casing or content within italic tags.
    - Flags Title Case in italics (should be sentence case for titles) (H-002)
    - Flags definitions in italics (should use quotes) (H-005)
    """
    findings = []
    in_italics = False
    italic_span_tokens = []

    for token in doc:
        if token.text == '__SEMANTIC_ITALIC_START__':
            in_italics = True
            italic_span_tokens = [token]
            continue
        elif token.text == '__SEMANTIC_ITALIC_END__':
            in_italics = False
            italic_span_tokens.append(token)
            
            # --- Process the completed italic span ---
            if len(italic_span_tokens) > 2:
                # Get the text content, excluding the placeholders
                text_tokens = italic_span_tokens[1:-1]
                italic_text = "".join(t.text_with_ws for t in text_tokens).strip()

                if not italic_text:
                    continue

                # Rule APS-GPC-Punctuationandcapitalisation-H-002
                # Check for Title Case (more than 2 words, most are capitalized)
                words = [t for t in text_tokens if t.is_alpha and not t.is_stop]
                if len(words) > 2:
                    title_cased_words = sum(1 for w in words if w.text.istitle())
                    # If more than half the significant words are title-cased, flag it.
                    if title_cased_words / len(words) > 0.5:
                        _add_finding(findings, get_line_number_from_offset(italic_span_tokens[0].idx, line_offsets), f"[Title Case]: {italic_text}")

                # Rule APS-GPC-Italics-H-005
                # Check for definition-like phrases
                if " is defined as " in italic_text.lower() or " means " in italic_text.lower():
                    _add_finding(findings, get_line_number_from_offset(italic_span_tokens[0].idx, line_offsets), f"[Definition]: {italic_text}")
            
            italic_span_tokens = []
            continue
        
        if in_italics:
            italic_span_tokens.append(token)
            
    return findings

def check_unitalicised_acts(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Checks for unitalicised legal Acts (Rule: APS-GPC-Italics-H-002).
    Finds patterns like "Name Act [Year]" that are not inside italic placeholders.
    """
    findings = []
    
    # Store (start_char, end_char) of all italic blocks
    italic_spans = []
    in_italics = False
    start_char = -1
    for token in doc:
        if token.text == '__SEMANTIC_ITALIC_START__':
            in_italics = True
            start_char = token.idx
        elif token.text == '__SEMANTIC_ITALIC_END__':
            if in_italics:
                italic_spans.append((start_char, token.idx + len(token.text)))
            in_italics = False
            start_char = -1

    # Now, find all Act names in the doc's plain text
    act_regex = re.compile(r'\b([A-Z][A-Za-z\s]+)\s+Act\s+\d{4}\b')
    
    for match in act_regex.finditer(doc.text):
        is_italicised = False
        for (start, end) in italic_spans:
            # Check if the match is *inside* an italic span
            if match.start() >= start and match.end() <= end:
                is_italicised = True
                break
        
        if not is_italicised:
            _add_finding(findings, get_line_number_from_offset(match.start(), line_offsets), match.group(0))
            
    return findings

# --- New Heuristics for APS Style Guide (January 2026) ---

def check_gene_vs_protein(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags potential protein names in italics (should be roman) or genes in roman (should be italics). (H-007)"""
    findings = []
    # Heuristic: Gene/Protein names often have specific alphanumeric patterns (e.g., BRCA1, p53)
    bio_pattern = re.compile(r'\b[A-Z]{2,}[0-9]*\b|\b[a-z]{2,}[0-9]+\b')
    
    in_italics = False
    for token in doc:
        if token.text == "__SEMANTIC_ITALIC_START__": in_italics = True
        elif token.text == "__SEMANTIC_ITALIC_END__": in_italics = False
        
        if bio_pattern.match(token.text):
            # If it's All-Caps (Protein) but inside Italics tags
            if token.text.isupper() and in_italics:
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), f"Protein '{token.text}' should be roman.")
            # If it's lowercase/mixed (Gene) but NOT in Italics
            elif not token.text.isupper() and not in_italics:
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), f"Gene '{token.text}' should be italicised.")
    return findings

def check_fractions_as_words(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags numeric fractions (1/2) in general text, preferring words (one-half). (H-002)"""
    findings = []
    for token in doc:
        if '/' in token.text and any(char.isdigit() for char in token.text):
            # Exclude units like km/h
            if not any(c.isalpha() for c in token.text):
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.text)
    return findings

def check_math_decimal_numerals(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags spelled out numbers used in mathematical contexts or ratios. (H-005)"""
    findings = []
    ratio_indicators = {":", "to", "ratio"}
    for token in doc:
        if token.pos_ == "NUM" and token.text.isalpha():
            # Check for ratio context: "a ratio of five to one"
            if any(child.text in ratio_indicators for child in token.head.children):
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.sent.text)
    return findings

def check_sentence_starting_percent(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags sentences beginning with a % symbol or a numeral percent. (H-001)"""
    findings = []
    for sent in doc.sents:
        first_token = sent[0]
        if first_token.text == "%" or (first_token.like_num and sent[1:2].text == "%"):
            _add_finding(findings, get_line_number_from_offset(first_token.idx, line_offsets), sent.text)
    return findings

def check_verb_presence(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags sentences that appear to lack a finite verb. (H-001)"""
    findings = []
    for sent in doc.sents:
        if len(sent) < 4: continue # Skip fragments/headers
        has_verb = any(t.pos_ == "VERB" or t.pos_ == "AUX" for t in sent)
        if not has_verb:
            _add_finding(findings, get_line_number_from_offset(sent.start_char, line_offsets), sent.text)
    return findings

def check_scientific_italics(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags potential binomial nomenclature (scientific names) not in italics. (H-006)"""
    findings = []
    # Basic pattern for Genus species
    latin_pattern = re.compile(r'\b[A-Z][a-z]+ [a-z]+\b')
    
    # Track italic spans to avoid false flags
    italic_text = []
    in_italics = False
    for token in doc:
        if token.text == "__SEMANTIC_ITALIC_START__": in_italics = True
        elif token.text == "__SEMANTIC_ITALIC_END__": in_italics = False
        if in_italics: italic_text.append(token.idx)

    for match in latin_pattern.finditer(doc.text):
        if match.start() not in italic_text:
            # Cross-reference with spaCy to ensure it looks like a Noun Phrase
            if doc.char_span(match.start(), match.end()) is not None:
                _add_finding(findings, get_line_number_from_offset(match.start(), line_offsets), match.group())
    return findings

def check_foreign_word_italics(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags foreign words (identified by spaCy as X) that aren't italicised. (H-003)"""
    findings = []
    for token in doc:
        if token.tag_ == "FW": # Foreign Word tag
            # Check if surrounded by placeholder tags
            prev_t = doc[token.i-1] if token.i > 0 else None
            next_t = doc[token.i+1] if token.i < len(doc)-1 else None
            if not (prev_t and prev_t.text == "__SEMANTIC_ITALIC_START__"):
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.text)
    return findings

def check_first_nations_italics(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Ensures First Nations words are NOT italicised. (H-009)"""
    findings = []
    # This requires a lookup or NER; using a sample common set for demonstration
    fn_keywords = {"Dreaming", "Yarn", "Country", "Lore", "Makarrata"}
    
    in_italics = False
    for token in doc:
        if token.text == "__SEMANTIC_ITALIC_START__": in_italics = True
        elif token.text == "__SEMANTIC_ITALIC_END__": in_italics = False
        
        if in_italics and token.text in fn_keywords:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), token.text)
    return findings

def check_telephone_formatting(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags telephone numbers with standard hyphens; prefers spaces/no-break spaces. (H-001)"""
    findings = []
    tel_pattern = re.compile(r'\+?\d{2,4}-\d{3,4}-\d{3,4}')
    for match in tel_pattern.finditer(doc.text):
        _add_finding(findings, get_line_number_from_offset(match.start(), line_offsets), match.group())
    return findings

# --- Additional/Refined Heuristics for APS Style Guide (2026) ---

def check_ordinal_pairing(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks that 'Firstly' is paired with 'Secondly' (H-002)."""
    findings = []
    for sent in doc.sents:
        text = sent.text.lower()
        if "firstly" in text and "secondly" not in text:
            # Check if 'secondly' appears in the subsequent sentence
            next_sent = doc[sent.end:].sents.__next__() if sent.end < len(doc) else None
            if next_sent and "secondly" not in next_sent.text.lower():
                _add_finding(findings, get_line_number_from_offset(sent.start_char, line_offsets), "Pair 'Firstly' with 'Secondly'.")
    return findings

def check_million_context(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags 'one million' vs '1 million' based on context (H-010)."""
    findings = []
    for token in doc:
        if token.lemma_ == "million":
            prev = doc[token.i - 1] if token.i > 0 else None
            if prev:
                # Rule: Numerals for exact data/stats, words for informal/general
                if prev.text.lower() == "one" and any(t.pos_ == "SYM" or t.dep_ == "nummod" for t in token.sent):
                    _add_finding(findings, get_line_number_from_offset(prev.idx, line_offsets), "Use '1 million' for statistical context.")
                elif prev.text == "1" and not any(t.pos_ == "SYM" for t in token.sent):
                    _add_finding(findings, get_line_number_from_offset(prev.idx, line_offsets), "Use 'one million' for general text.")
    return findings

def check_iso_datetime(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags ISO date formats in general text unless specified as required (H-004)."""
    findings = []
    iso_pattern = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
    for match in iso_pattern.finditer(doc.text):
        _add_finding(findings, get_line_number_from_offset(match.start(), line_offsets), f"ISO date '{match.group()}' may be too technical for general text.")
    return findings

def check_mathematical_symbols(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Prefer words 'plus', 'minus', 'equals' to symbols in general content (H-001)."""
    findings = []
    symbols = {"+", "=", ">", "<"}
    for token in doc:
        if token.text in symbols and token.pos_ == "SYM":
            # Exclude if inside a specialized 'math' block or table (if identifiable)
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), f"Replace symbol '{token.text}' with words in general prose.")
    return findings

def check_reverse_italics(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for roman text inside an italicized title (H-008)."""
    findings = []
    in_italics = False
    for token in doc:
        if token.text == "__SEMANTIC_ITALIC_START__":
            in_italics = True
        elif token.text == "__SEMANTIC_ITALIC_END__":
            in_italics = False
        
        # If we see Markdown bold/italic syntax *inside* our semantic tags, it indicates a flip
        if in_italics and (token.text == "*" or token.text == "_"):
             _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), "Ensure reverse italics are used correctly within italic titles.")
    return findings
# --- Heuristic Rule Implementations (Additions) ---

def check_au_place_name_punctuation(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags hyphens or apostrophes in place names for manual verification (Rule: APS-GPC-Australianplacenames-H-002)."""
    findings = []
    for ent in doc.ents:
        if ent.label_ == "GPE" and any(char in ent.text for char in ("-", "'", "’")):
            _add_finding(findings, get_line_number_from_offset(ent.start_char, line_offsets), 
                         f"Check official spelling: {ent.text}")
    return findings

def check_au_state_short_forms(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags short state names to ensure they meet one of the 4 allowed situations (Rule: APS-GPC-Australianplacenames-H-005)."""
    findings = []
    for token in doc:
        if token.text in AU_STATE_SHORT_FORMS:
            # Situation 1: Adjectival (e.g. "the NSW Government")
            is_adjectival = token.dep_ == "compound" or (token.head.pos_ in ["NOUN", "PROPN"] and token.i < token.head.i)
            if not is_adjectival:
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), 
                             f"Verify usage of short form '{token.text}'")
    return findings

def check_generic_plurals_lowercase(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags capitalized plural generic terms after named features (Rule: APS-GPC-Australianplacenames-H-008)."""
    findings = []
    for token in doc:
        # Look for plural nouns that are capitalized but not at the start of a sentence
        if token.tag_ == "NNS" and token.text[0].isupper() and not token.is_sent_start:
            # Check if it follows coordinated proper nouns (e.g., "The Murray and Darling Rivers")
            prev_tokens = [t for t in token.sent if t.i < token.i]
            if any(t.pos_ == "PROPN" for t in prev_tokens):
                 _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), 
                              f"'{token.text}' should likely be lowercase (generic plural)")
    return findings

def check_australian_government_casing(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Ensures 'Australian Government' is capped only together (Rule: APS-GPC-Governmentterms-H-001)."""
    findings = []
    for i in range(len(doc) - 1):
        # Violation 1: 'Australian government' (lower g)
        if doc[i].text == "Australian" and doc[i+1].text == "government":
            _add_finding(findings, get_line_number_from_offset(doc[i].idx, line_offsets), "Australian Government")
        # Violation 2: Generic 'the Government' (capped G when alone)
        if doc[i].text == "Government" and (i == 0 or doc[i-1].text != "Australian") and not doc[i].is_sent_start:
            _add_finding(findings, get_line_number_from_offset(doc[i].idx, line_offsets), 
                         "Use lowercase 'government' for generic mentions")
    return findings

def check_budget_casing(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags generic 'Budget' terms that should be lowercase (Rule: APS-GPC-Governmentterms-H-002)."""
    findings = []
    for token in doc:
        if token.text == "Budget" and not token.is_sent_start:
            # Formal if it follows 'the' or a year (e.g. 'the 2024 Budget')
            is_formal = token.i > 0 and (doc[token.i-1].text.lower() == "the" or doc[token.i-1].like_num)
            if not is_formal:
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), 
                             "Use lowercase 'budget' for generic/adjectival mentions")
    return findings

def check_federal_casing(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags 'Federal' used generically (Rule: APS-GPC-Governmentterms-H-005)."""
    findings = []
    for token in doc:
        if token.text == "Federal" and not token.is_sent_start:
            # Check if part of a formal title (compounded with a Proper Noun)
            is_formal = token.head.pos_ == "PROPN" or any(child.pos_ == "PROPN" for child in token.children)
            if not is_formal:
                _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), 
                             "Use lowercase 'federal' for generic mentions")
    return findings

def check_commercial_generic_preference(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags specific brands to suggest generic alternatives (Rule: APS-GPC-Commercialterms-H-001)."""
    findings = []
    brand_to_generic = {"Band-Aid": "adhesive bandage", "Xerox": "photocopy", "Post-it": "sticky note"}
    for token in doc:
        if token.text in brand_to_generic:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), 
                         f"Consider generic '{brand_to_generic[token.text]}' instead of '{token.text}'")
    return findings

def check_brand_stylization(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Ensures stylised brands like 'eBay' or 'iPhone' are not auto-capitalised (Rule: APS-GPC-Commercialterms-H-004/H-005)."""
    findings = []
    stylised = {"eBay", "iPhone", "iPad", "macOS"}
    for token in doc:
        # Flag if a known stylised brand is written in standard title case (e.g., 'Ebay')
        if token.text.capitalize() in stylised and token.text not in stylised:
            _add_finding(findings, get_line_number_from_offset(token.idx, line_offsets), 
                         f"Preserve brand stylisation: '{token.text}' -> '{[s for s in stylised if s.lower()==token.text.lower()][0]}'")
    return findings

def check_personal_name_casing(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Flags lowercase personal names (Rule: APS-GPC-Personalnames-H-003)."""
    findings = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if not all(word[0].isupper() for word in ent.text.split() if len(word) > 2):
                _add_finding(findings, get_line_number_from_offset(ent.start_char, line_offsets), 
                             f"Ensure name is capitalised: {ent.text}")
    return findings

def check_nickname_formatting(doc: Doc, line_offsets: List[int]) -> List[Dict[str, Any]]:
    """Checks for quoted nicknames on first mention (Rule: APS-GPC-Personalnames-H-004)."""
    findings = []
    for i in range(len(doc) - 2):
        # Look for: Name "Nickname" Surname
        if doc[i].pos_ == "PROPN" and doc[i+1].text in ('"', "“") and doc[i+2].pos_ == "PROPN":
             pass # Correctly formatted
        # Implementation of "first mention" detection requires state tracking across the doc
    return findings

# --- Master Dictionary of Heuristic Checks ---
HEURISTIC_CHECKS: Dict[str, Callable[[Doc, List[int]], List[Dict[str, Any]]]] = {
    "APS-GPC-Partsofsentences-H-009": check_passive_voice,
    "APS-GPC-Partsofsentences-H-001": check_complete_sentence,
    "APS-GPC-Nouns-R-004": check_collective_noun_agreement,
    "APS-GPC-Adjectives-H-002": check_hyphenated_modifier,
    "APS-GPC-Pronouns-H-005": check_that_vs_which,
    "APS-GPC-Nouns-H-001": check_missing_determiner,
    "APS-GPC-Exclamationmarks-H-001": check_exclamation_marks,
    "APS-GPC-Conjunctions-H-001": check_matched_correlatives,
    "APS-GPC-Latinshortenedforms-H-001": check_prefer_english_forms,
    "APS-GPC-Adjectives-R-002": check_unique_grading,
    "APS-GPC-Typesofwords-H-002": check_misplaced_only,
    "APS-GPC-Adverbs-H-001": check_filler_adverbs,
    "APS-GPC-Verbs-R-007": check_modal_verb_to,
    "APS-GPC-Pronouns-H-004": check_improper_reflexive_pronoun,
    "APS-GPC-Determiners-R-001": check_a_vs_an,
    "APS-GPC-Determiners-R-002": check_a_vs_an,
    "APS-GPC-Determiners-R-003": check_a_vs_an,
    "APS-GPC-Determiners-R-004": check_a_vs_an,
    "APS-GPC-Determiners-R-005": check_a_vs_an,
    "APS-GPC-Determiners-R-006": check_a_vs_an,
    # Added Context-Aware Heuristics 17/10/2025
    "APS-GPC-Organisationnames-H-007": check_generic_organisation_reference,
    "APS-GPC-Governmentterms-H-007": check_generic_official_titles,
    "APS-GPC-Governmentterms-H-011": check_generic_official_titles,
    "APS-GPC-Governmentterms-H-012": check_generic_official_titles,
    "APS-GPC-Governmentterms-H-014": check_generic_parliamentary_terms,
    "APS-GPC-Governmentterms-H-015": check_generic_parliamentary_terms,
    "APS-GPC-Medicalterms-H-004": check_capitalized_common_noun_definitions,
    "APS-GPC-Nationalities,peoplesandplacesoutsideAustralia-H-003": check_capitalized_the_before_country,
    "APS-GPC-Nationalities,peoplesandplacesoutsideAustralia-H-006": check_improperly_cased_regions,
    "APS-GPC-Organisationnames-H-008": check_org_the_capitalisation,
    "APS-GPC-Organisationnames-H-010": check_org_verb_agreement,
    "APS-GPC-Choosingnumeralsorwords-H-001": check_numerals_vs_words,
    "APS-GPC-Choosingnumeralsorwords-H-009": check_large_rounded_numbers,
    "APS-GPC-Currency-H-001": check_aud_currency,
    "APS-GPC-Currency-H-002": check_ambiguous_dollar_sign,
    "APS-GPC-Datesandtime-H-001": check_year_span_words,
    "APS-GPC-Datesandtime-H-002": check_financial_year_dash,
    "APS-GPC-Datesandtime-H-003": check_day_month_span_words,
    "APS-GPC-Fractionsanddecimals-H-001": check_worded_decimals,
    "APS-GPC-Ordinalnumbers-H-001": check_prose_ordinals_for_steps,
    "APS-GPC-Partsofsentences-H-005": check_compound_clause_comma,
    "APS-GPC-Partsofsentences-H-006": check_comma_with_shared_subject,
    "APS-GPC-Partsofsentences-H-008": check_prefer_active_voice,
    "APS-GPC-Partsofsentences-H-010": check_passive_missing_agent,
    "APS-GPC-Phrases-H-001": check_possessive_in_compound,
    "APS-GPC-Phrases-H-002": check_phrase_fragments,
    "APS-GPC-Apostrophes-H-001": check_descriptive_apostrophe,
    "APS-GPC-Bracketsandparentheses-H-003": check_nested_parentheses,
    "APS-GPC-Bracketsandparentheses-H-004": check_round_brackets_in_quotes,
    "APS-GPC-Commas-H-001": check_comma_after_short_intro,
    "APS-GPC-Commas-H-002": check_appositive_commas,
    "APS-GPC-Commas-H-004": check_for_example_commas,
    "APS-GPC-Commas-H-006": check_oxford_comma,
    "APS-GPC-Commas-H-007": check_oxford_comma,  # H-007 is the exception, handled by the same check
    "APS-GPC-Dashes-H-001": check_en_dash_spans,
    "APS-GPC-Dashes-H-002": check_en_dash_spans,  # H-002 is the exception, handled by the same check
    "APS-GPC-Forwardslashes-H-003": check_word_slashes,
    "APS-GPC-Semicolons-H-001": check_semicolon_usage,
    "APS-GPC-Semicolons-H-002": check_semicolon_usage,
    "APS-GPC-Semicolons-H-003": check_semicolon_usage, # This rule is an exception, handled *inside* the function
    "APS-GPC-Acronymsandinitialisms-H-005": check_gov_acronyms,
    "APS-GPC-Latinshortenedforms-H-003": check_etc_in_lists,
    "APS-GPC-Adjectives-H-001": check_weak_adjectives,
    "APS-GPC-Adjectives-H-003": check_predicate_hyphenation,
    "APS-GPC-Adjectives-H-004": check_adjective_strings,
    "APS-GPC-Adjectives-H-005": check_adjective_strings,
    "APS-GPC-Adverbs-H-002": check_adjective_as_adverb,
    "APS-GPC-Italics-H-001": check_missing_italics_for_works,
    "APS-GPC-Punctuationandcapitalisation-H-001": check_punctuation_in_structural_tags,
    "APS-GPC-Punctuationandcapitalisation-H-002": check_italic_case,
    "APS-GPC-Italics-H-002": check_unitalicised_acts,
    "APS-GPC-Italics-H-005": check_italic_case,
    # --- New Heuristics Mapping (January 2026) ---
    "APS-GPC-Plantsandanimals-H-007": check_gene_vs_protein,
    "APS-GPC-Choosingnumeralsorwords-H-002": check_fractions_as_words,
    "APS-GPC-Choosingnumeralsorwords-H-005": check_math_decimal_numerals,
    "APS-GPC-Percentages-H-001": check_sentence_starting_percent,
    "APS-GPC-Verbs-H-001": check_verb_presence,
    "APS-GPC-Italics-H-003": check_foreign_word_italics,
    "APS-GPC-Italics-H-006": check_scientific_italics,
    "APS-GPC-Italics-H-009": check_first_nations_italics,
    "APS-GPC-Telephonenumbers-H-001": check_telephone_formatting,
    "APS-GPC-Ordinalnumbers-H-002": check_ordinal_pairing,
    "APS-GPC-Choosingnumeralsorwords-H-010": check_million_context,
    "APS-GPC-Datesandtime-H-004": check_iso_datetime,
    "APS-GPC-Mathematicalrelationships-H-001": check_mathematical_symbols,
    "APS-GPC-Italics-H-008": check_reverse_italics,
    "APS-GPC-Australianplacenames-H-002": check_au_place_name_punctuation,
    "APS-GPC-Australianplacenames-H-005": check_au_state_short_forms,
    "APS-GPC-Australianplacenames-H-008": check_generic_plurals_lowercase,
    "APS-GPC-Governmentterms-H-001": check_australian_government_casing,
    "APS-GPC-Governmentterms-H-002": check_budget_casing,
    "APS-GPC-Governmentterms-H-005": check_federal_casing,
    "APS-GPC-Commercialterms-H-001": check_commercial_generic_preference,
    "APS-GPC-Commercialterms-H-004": check_brand_stylization,
    "APS-GPC-Personalnames-H-003": check_personal_name_casing,
}

def load_rules_from_rulebook(file_path: str) -> List[Dict[str, Any]]:
    """Loads, validates, and compiles linting rules from the specified JSON rulebook."""
    if not os.path.exists(file_path):
        logging.error(f"Rulebook file '{file_path}' not found.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rule_sets = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from rulebook: {e}")
        return []

    all_rules = [rule for rule_set in rule_sets for rule in rule_set.get('rules', [])]
    
    transformed_rules = []
    unimplemented_heuristics = 0
    for rule in all_rules:
        rule_type = rule.get("category")
        rule_id = rule.get("id")

        new_rule = { "id": rule_id, "description": rule.get("message"), "severity": rule.get("severity"), "type": rule_type }

        if rule_type == "regex" and "pattern" in rule:
            pattern = rule.get("pattern", "")
            try:
                flags = re.IGNORECASE if not pattern.startswith("(?i)") else 0
                new_rule["compiled_pattern"] = re.compile(pattern, flags)
                transformed_rules.append(new_rule)
            except re.error as e:
                logging.warning(f"Skipping invalid regex for rule '{rule_id}': {e}")

        elif rule_type == "heuristic":
            if rule_id in HEURISTIC_CHECKS:
                new_rule["check"] = HEURISTIC_CHECKS[rule_id]
                transformed_rules.append(new_rule)
            else:
                unimplemented_heuristics += 1
    
    if unimplemented_heuristics > 0:
        logging.info(f"Skipped {unimplemented_heuristics} heuristic rules that do not have a Python implementation.")

    return transformed_rules

def build_github_url(file_name: str, line_number: int) -> str:
    """Constructs a permalink to a specific line in a file on GitHub if CI environment variables are present."""
    server_url = os.getenv("GITHUB_SERVER_URL")
    repository = os.getenv("GITHUB_REPOSITORY")
    sha = os.getenv("GITHUB_SHA")

    if not all([server_url, repository, sha]):
        return f"local://{file_name}#L{line_number}"

    # Updated to use the correct directory in the URL path
    return f"{server_url}/{repository}/blob/{sha}/{MARKDOWN_DIR}/{file_name}#L{line_number}"

def lint_file(file_path: str, file_name: str, linting_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies all defined linting rules to a single file."""
    findings: List[Dict[str, Any]] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"Could not find file {file_path}")
        return []

    lines = content.splitlines()
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    doc = nlp(content)
    reported_findings = set()

    for rule in linting_rules:
        try:
            if rule.get('type') == 'regex':
                compiled_pattern = rule.get("compiled_pattern")
                if not compiled_pattern: continue
                for line_num, line in enumerate(lines, 1):
                    if compiled_pattern.search(line):
                        finding_tuple = (file_name, line_num, rule.get('id'), line.strip())
                        if finding_tuple not in reported_findings:
                            findings.append({
                                "fileName": file_name, "lineNumber": line_num,
                                "ruleId": rule.get('id'), "ruleDescription": rule.get('description'),
                                "severity": rule.get('severity'), "offendingText": line.strip(),
                                "githubUrl": build_github_url(file_name, line_num)
                            })
                            reported_findings.add(finding_tuple)

            elif rule.get('type') == 'heuristic':
                heuristic_findings = rule['check'](doc, line_offsets)
                for h_finding in heuristic_findings:
                    finding_tuple = (file_name, h_finding['line_number'], rule.get('id'), h_finding['offending_text'])
                    if finding_tuple not in reported_findings:
                        findings.append({
                            "fileName": file_name, "lineNumber": h_finding['line_number'],
                            "ruleId": rule.get('id'), "ruleDescription": rule.get('description'),
                            "severity": rule.get('severity'), "offendingText": h_finding['offending_text'],
                            "githubUrl": build_github_url(file_name, h_finding['line_number'])
                        })
                        reported_findings.add(finding_tuple)
        except Exception as e:
            logging.error(f"Error applying rule '{rule.get('id', 'N/A')}' to {file_name}: {e}")

    return findings

def main() -> None:
    """Main function to orchestrate the linting process and generate the report."""
    all_findings = []
    linting_rules = load_rules_from_rulebook(RULEBOOK_FILE)
    
    if not linting_rules:
        logging.warning("No linting rules were loaded. An empty report will be created.")
    else:
        logging.info(f"Successfully loaded {len(linting_rules)} rules from {RULEBOOK_FILE}.")

    if os.path.exists(MARKDOWN_DIR):
        for file_name in sorted(os.listdir(MARKDOWN_DIR)):
            if file_name.endswith('.md'):
                file_path = os.path.join(MARKDOWN_DIR, file_name)
                logging.info(f"Linting {file_path}...")
                findings = lint_file(file_path, file_name, linting_rules)
                all_findings.extend(findings)
    else:
        logging.warning(f"Markdown directory '{MARKDOWN_DIR}' not found. No files to lint.")

    all_findings.sort(key=lambda x: (x['fileName'], x['lineNumber'], x['ruleId']))

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_findings, f, indent=2)

    logging.info(f"Linting complete. Report generated at {REPORT_FILE}")
    logging.info(f"Found {len(all_findings)} issues.")

if __name__ == "__main__":
    main()
