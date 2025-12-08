import json
import random
import pandas as pd
import re

# --------------------------------------------------------------------------
# Helper: Get Examples
# --------------------------------------------------------------------------
def get_few_shot_examples(job, example_banks):
    """
    Retrieves 2-3 examples from the CSV based on CEFR and Type.
    """
    bank = example_banks.get(job['type'].lower())
    if bank is None or bank.empty: 
        return ""
    
    bank.columns = [c.strip() for c in bank.columns]
    
    if 'CEFR rating' in bank.columns:
        relevant = bank[bank['CEFR rating'].astype(str).str.strip() == str(job['cefr']).strip()]
    else:
        relevant = bank

    if len(relevant) >= 2:
        samples = relevant.sample(2)
    elif len(bank) >= 2:
        samples = bank.sample(2) 
    else:
        return "" 

    output = ""
    for _, row in samples.iterrows():
        ex_dict = {
            "Question Prompt": row.get("Question Prompt", "N/A"),
            "Answer A": row.get("Answer A", "N/A"),
            "Answer B": row.get("Answer B", "N/A"),
            "Answer C": row.get("Answer C", "N/A"),
            "Answer D": row.get("Answer D", "N/A"),
            "Correct Answer": row.get("Correct Answer", "N/A")
        }
        output += "### EXAMPLE:\n" + json.dumps(ex_dict) + "\n\n"
    return output

# =============================================================================
# HELPER FUNCTIONS FOR VOCABULARY SELECTION
# =============================================================================

def clean_vocab_item(text):
    """
    Aggressively cleans vocabulary items for distractor generation.
    - 'build/built/built' -> 'build'
    - 'belong (to)' -> 'belong'
    """
    if not isinstance(text, str):
        return str(text)
    
    # Take first option if slashes exist (build/built -> build)
    if '/' in text:
        text = text.split('/')[0]
        
    # Remove parentheses and content (belong (to) -> belong)
    text = re.sub(r'\([^)]*\)', '', text)
    
    return text.strip()

def get_first_word(vocab_item):
    """Extract the first word from multi-word vocabulary items."""
    cleaned = clean_vocab_item(vocab_item)
    words = cleaned.split()
    return words[0] if words else cleaned

def get_initial_letter(vocab_item):
    """Get the first letter of the first word."""
    first_word = get_first_word(vocab_item)
    return first_word[0].lower() if first_word else ''

def get_phonetic_similar_letters(letter):
    """Return phonetically similar letters for fallback matching."""
    phonetic_groups = {
        'c': ['k', 'q'], 'k': ['c', 'q'], 'q': ['c', 'k'],
        's': ['c', 'z'], 'z': ['s'],
        'f': ['ph'], 'ph': ['f'],
        'j': ['g'], 'g': ['j'],
        'i': ['y'], 'y': ['i'],
        'b': ['p'], 'p': ['b'], 
        'd': ['t'], 't': ['d'],
        'v': ['f', 'w']
    }
    return phonetic_groups.get(letter.lower(), [])

def python_select_by_pos(vocab_df, target_vocab, target_pos, max_items=4):
    """
    Select distractors by matching part of speech.
    Returns CLEANED items.
    """
    target_vocab_clean = clean_vocab_item(target_vocab).lower()
    target_pos_lower = target_pos.lower().strip()
    
    # Filter by same part of speech
    same_pos = vocab_df[
        vocab_df['Part of Speech'].str.lower().str.strip() == target_pos_lower
    ]
    
    # Filter out target
    same_pos = same_pos[
        same_pos['Base Vocabulary Item'].apply(clean_vocab_item).str.lower() != target_vocab_clean
    ]
    
    if len(same_pos) >= max_items:
        selected = same_pos.sample(n=max_items)
    else:
        selected = same_pos
    
    # Return cleaned items
    return [clean_vocab_item(x) for x in selected['Base Vocabulary Item'].tolist()]

def python_select_by_initial_letter(vocab_df, target_vocab, max_items=4, exclude_items=None):
    """
    Select distractors by matching initial letter of first word (with phonetic fallback).
    Returns CLEANED items.
    """
    if exclude_items is None:
        exclude_items = []
    
    target_vocab_clean = clean_vocab_item(target_vocab)
    target_letter = get_initial_letter(target_vocab_clean)
    
    # Helper to check letter match
    def matches_initial_letter(vocab_item):
        return get_initial_letter(vocab_item) == target_letter
    
    same_letter = vocab_df[
        vocab_df['Base Vocabulary Item'].apply(matches_initial_letter)
    ]
    
    # Exclude target and already selected
    exclude_clean = [clean_vocab_item(x).lower() for x in exclude_items + [target_vocab]]
    same_letter = same_letter[
        ~same_letter['Base Vocabulary Item'].apply(clean_vocab_item).str.lower().isin(exclude_clean)
    ]
    
    # Selection logic
    if len(same_letter) >= max_items:
        selected = same_letter.sample(n=max_items)
        return [clean_vocab_item(x) for x in selected['Base Vocabulary Item'].tolist()]
    
    # Fallback logic (Phonetic)
    candidates = [clean_vocab_item(x) for x in same_letter['Base Vocabulary Item'].tolist()]
    phonetic_letters = get_phonetic_similar_letters(target_letter)
    
    for phon_letter in phonetic_letters:
        if len(candidates) >= max_items:
            break
        
        def matches_phonetic_letter(vocab_item):
            return get_initial_letter(vocab_item) == phon_letter
        
        phonetic_matches = vocab_df[
            vocab_df['Base Vocabulary Item'].apply(matches_phonetic_letter)
        ]
        
        # Filter exclusions again
        phonetic_matches = phonetic_matches[
            ~phonetic_matches['Base Vocabulary Item'].apply(clean_vocab_item).str.lower().isin(exclude_clean + [c.lower() for c in candidates])
        ]
        
        needed = max_items - len(candidates)
        if len(phonetic_matches) > 0:
            additional = phonetic_matches.sample(n=min(needed, len(phonetic_matches)))
            candidates.extend([clean_vocab_item(x) for x in additional['Base Vocabulary Item'].tolist()])
    
    return candidates[:max_items]

# =============================================================================
# STAGE 1: SENTENCE GENERATION (With Inflection Instruction)
# =============================================================================

def create_vocab_list_stage1_prompt(job_list, question_form):
    """
    Generates complete sentences. 
    Crucial: Generates the INFLECTED correct answer.
    """
    question_form_instructions = {
        "Random Mix": "Use diverse question forms.",
        "Simple gap fill": """ALL questions must use simple gap fill format. Example: "Our new sidewalk is made of ___________." """,
        "Definition through function/description": "ALL questions must use definition through function/description format.",
        "Cause-Effect completion": "ALL questions must use cause-effect completion format.",
        "Dialogue completion": "ALL questions must use dialogue completion format.",
        "Logical relationship completion": "ALL questions must use logical relationship completion format."
    }
    
    form_instruction = question_form_instructions.get(question_form, question_form_instructions["Random Mix"])
    
    system_msg = f"""You are an expert ELT content creator. You will generate exactly {len(job_list)} complete test questions in a single JSON response targeting specific vocabulary items.

CRITICAL: Your entire response must be a JSON object with a "questions" key containing an array of exactly {len(job_list)} question objects."""
    
    job_specs = []
    for job in job_list:
        # Clean target for the prompt
        target_vocab = clean_vocab_item(job['target_vocabulary'])
        
        job_specs.append({
            "job_id": job['job_id'],
            "cefr": job['cefr'],
            "target_vocabulary": target_vocab,
            "definition": job.get('definition', ''),
            "part_of_speech": job.get('part_of_speech', '')
        })
 
    user_msg = f"""
TASK: Create exactly {len(job_list)} vocabulary test questions.

VOCABULARY TARGETS:
{json.dumps(job_specs, indent=2)}

{form_instruction}

GENERATION INSTRUCTIONS:
1. **CONTEXT:** Create a natural sentence where the target word fits perfectly.
2. **INFLECTION (CRITICAL):** 
   - You MUST inflect the target word to match the sentence grammar.
   - If the sentence is past tense, "blow" MUST become "blew".
   - The "Correct Answer" field MUST contain the INFLECTED form.
3. **CLARITY:** Ensure the context makes the meaning clear without definitions.

MANDATORY OUTPUT FORMAT:
{{
  "questions": [
    {{
      "Item Number": "...",
      "Target Vocabulary": "...",
      "Complete Sentence": "...",
      "Correct Answer": "...[INFLECTED FORM]...",
      "Context Clue Location": "...",
      "Context Clue Explanation": "...",
      "CEFR rating": "...",
      "Category": "Vocabulary"
    }},
    ... 
  ]
}}
"""
    return system_msg, user_msg

# =============================================================================
# STAGE 2: CANDIDATE GENERATION (With Morphological Adaptation)
# =============================================================================

def create_vocab_list_stage2_prompt(job_list, stage1_outputs, vocabulary_list_df):
    """
    Generates candidates.
    CRITICAL: Forces LLM to inflect the Python-selected candidates.
    API FIX: System prompt now explicitly mentions 'JSON'.
    """
    system_msg = f"""You are an expert ELT test designer. You will create exactly 8 candidate distractors for exactly {len(job_list)} questions in JSON format.
    
CRITICAL: You must ADAPT the input words to match the grammatical context of the sentences."""
    
    pre_selected_data = []
    
    for i, job in enumerate(job_list):
        stage1_data = stage1_outputs[i]
        target_vocab = clean_vocab_item(job['target_vocabulary'])
        target_pos = job['part_of_speech']
        
        # PYTHON SELECTION (Cleaned items)
        pos_selected = python_select_by_pos(
            vocabulary_list_df, target_vocab, target_pos, max_items=4
        )
        
        letter_selected = python_select_by_initial_letter(
            vocabulary_list_df, target_vocab, max_items=4, exclude_items=pos_selected
        )
        
        total_python = len(pos_selected) + len(letter_selected)
        needed_from_llm = max(0, 8 - total_python)
        
        pre_selected_data.append({
            "Item Number": stage1_data.get("Item Number"),
            "Target (Base)": target_vocab,
            "Complete Sentence": stage1_data.get("Complete Sentence"),
            "Correct Answer (In Sentence)": stage1_data.get("Correct Answer"),
            "Raw Candidates (from Database)": pos_selected + letter_selected,
            "Additional Candidates Needed": needed_from_llm
        })
    
    user_msg = f"""
TASK: Create a final pool of exactly 8 candidates for each question.
OUTPUT FORMAT: JSON

INPUT DATA:
{json.dumps(pre_selected_data, indent=2)}

INSTRUCTIONS FOR CANDIDATE GENERATION:

1. **SOURCE MATERIAL:** 
   - Start with the "Raw Candidates (from Database)".
   - Generate "Additional Candidates" to reach exactly 8 total.
   - Priority for additional candidates: Antonyms of target, then synonyms of the raw candidates.

2. **MORPHOLOGICAL ADAPTATION (CRITICAL):** 
   - The "Raw Candidates" are in BASE DICTIONARY FORM (e.g., "burn", "slip").
   - You MUST CONJUGATE/MODIFY these words to match the "Correct Answer (In Sentence)".
   - **TENSE MATCHING:** If Correct Answer is "blew" (past), "burn" must become "burned".
   - **NUMBER MATCHING:** If Correct Answer is "apples", "fruit" must become "fruits".
   - **FORM MATCHING:** If Correct Answer is "running", "walk" must become "walking".

   **EXAMPLE:**
   - Sentence: "When the movie was over..."
   - Correct Answer: "was over" (Past)
   - Raw Candidate: "burn"
   - **YOUR OUTPUT:** "burned" (Past Tense) -- *Do NOT output "burn"*

3. **CLEANING:**
   - Ensure all output candidates are single words (unless the distractor is a phrasal verb).
   - NEVER output slashes (e.g., "build/built/built" -> output "built").

MANDATORY OUTPUT FORMAT:
{{
  "candidates": [
    {{
      "Item Number": "...",
      "Candidate A": "...[Adapted/Inflected Item]...",
      "Candidate B": "...[Adapted/Inflected Item]...",
      "Candidate C": "...[Adapted/Inflected Item]...",
      "Candidate D": "...[Adapted/Inflected Item]...",
      "Candidate E": "...[Adapted/Inflected Item]...",
      "Candidate F": "...[Adapted/Inflected Item]...",
      "Candidate G": "...[Adapted/Inflected Item]...",
      "Candidate H": "...[Adapted/Inflected Item]...",
      "Transformation Notes": "e.g., Adapted 'burn' to 'burned' to match past tense"
    }},
    ... 
  ]
}}
"""
    return system_msg, user_msg

# =============================================================================
# STAGE 3: VALIDATION (With Morphological Parity)
# =============================================================================

def create_vocab_list_stage3_prompt(job_list, stage1_outputs, stage2_outputs):
    """
    STAGE THREE: Validation with MORPHOLOGICAL PARITY enforcement.
    API FIX: System prompt now explicitly mentions 'JSON'.
    """
    system_msg = f"""You are an expert English vocabulary validator. You will filter candidate distractors using strict morphological rules. Output results in JSON format."""
    
    validation_input = []
    for i, (job, s1, s2) in enumerate(zip(job_list, stage1_outputs, stage2_outputs)):
        # Safe extraction of candidates list
        candidates = []
        if isinstance(s2, dict):
            # Try to get candidate fields A-H
            candidates = [s2.get(f"Candidate {k}", "") for k in "ABCDEFGH"]
            # Filter out empty strings
            candidates = [c for c in candidates if c]
        
        validation_input.append({
            "Item Number": s1.get("Item Number", ""),
            "Complete Sentence": s1.get("Complete Sentence", ""),
            "Correct Answer": s1.get("Correct Answer", ""),
            "Candidates": candidates,
            "Target Word Class": job.get('part_of_speech', 'Unknown')
        })
    
    user_msg = f"""
TASK: Validate candidates and select the final 3 distractors per question.
OUTPUT FORMAT: JSON

INPUT:
{json.dumps(validation_input, indent=2)}

VALIDATION PROTOCOL (Apply in Order):

**STEP 1: MORPHOLOGICAL PARITY CHECK (The "Shape" Test)**
Look at the **Correct Answer** to determine the rule:

*   **CASE A: The Correct Answer is INFLECTED** (e.g., "walked", "tables", "going", "happiest", "mice").
    *   **RULE:** STRICT PARITY. Distractors MUST match the word class and inflection.
    *   **Action:** REJECT any candidate that is a different part of speech or lacks the matching inflection.
    *   *Example:* Target "blew" (Past V) -> REJECT "house" (Noun). REJECT "blow" (Base V). ACCEPT "cooked" (Past V).

*   **CASE B: The Correct Answer is BASE FORM** (e.g., "walk", "table", "big", "be").
    *   **RULE:** BASE PARITY. Distractors can be different word classes, BUT MUST BE BASE FORM.
    *   **Action:** REJECT any candidate that has an inflection suffix (-ed, -ing, -s).
    *   *Example:* Target "be" -> REJECT "burned" (Past). REJECT "building" (Gerund). ACCEPT "burn" (Base).

**STEP 2: SYNTACTIC & SEMANTIC FIT (The "Fit & Logic" Test)**
For candidates that pass Step 1, apply these two filters:

*   **A. SYNTACTIC FIT (Grammar Structure):** 
    Does the word fit the **sentence structure** (prepositions, transitivity, collocations)?
    *   *Example:* "He ____ me the truth." (Target: told).
    *   *Test:* "said" -> "He said me..." -> **REJECT** (Syntax error: 'said' cannot take indirect object).
    *   *Test:* "spoke" -> "He spoke me..." -> **REJECT** (Syntax error: needs 'to').

*   **B. EXAMINER ACCEPTANCE (Semantic Logic):**
    If a student wrote this answer, would it be **logically acceptable** (even if not the target)?
    *   *Test:* "We spent the day at the ____." (Target: beach).
    *   *Candidate:* "college". -> "We spent the day at the college." -> **REJECT** (Logically valid = Bad Distractor).
    *   *Candidate:* "decision". -> "We spent the day at the decision." -> **ACCEPT** (Grammatically fits, but logically absurd).

**STEP 3: FINAL SELECTION**
*   Select the 3 candidates that survive Step 1 & 2.
*   Prioritize candidates that look "plausible" to a learner but are definitely wrong.

MANDATORY OUTPUT FORMAT:
{{
  "validated": [
    {{
      "Item Number": "...",
      "Selected Distractor A": "...",
      "Selected Distractor B": "...",
      "Selected Distractor C": "...",
      "Validation Notes": "Refers to parity rule applied (e.g., 'Target inflected, rejected noun distractor')."
    }},
    ...
  ]
}}
"""
    return system_msg, user_msg


# =============================================================================
# LEGACY / BATCH FUNCTIONS (Preserved from input)
# =============================================================================

def create_sequential_batch_stage1_prompt(job_list, example_banks):
    # This remains unchanged from your existing file
    examples = get_few_shot_examples(job_list[0], example_banks) if job_list else ""
    system_msg = f"""You are an expert ELT content creator. You will generate exactly {len(job_list)} complete test questions in a single JSON response. 

CRITICAL: Your entire response must be a JSON object with a "questions" key containing an array of exactly {len(job_list)} question objects. Do not generate fewer questions than requested."""
    
    job_specs = []
    has_grammar_distinction = False
    has_vocabulary = False
    
    for job in job_list:
        raw_context = job.get('context', 'General')
        main_topic = raw_context
        
        job_specs.append({
            "job_id": job['job_id'],
            "cefr": job['cefr'],
            "type": job['type'],
            "focus": job['focus'],
            "topic": main_topic
        })
        
        if job['type'] == 'Grammar' and 'vs' in job['focus'].lower():
            has_grammar_distinction = True
        if job['type'] == 'Vocabulary':
            has_vocabulary = True
            
    constraint_instruction = ""
    if has_grammar_distinction:
        constraint_instruction += "GRAMMATICAL EXCLUSIVITY RULE: Include a grammatical signal."
    if has_vocabulary:
        constraint_instruction += "SEMANTIC EXCLUSIVITY RULE: Include semantic context clues."

    user_msg = f"""
TASK: Create exactly {len(job_list)} complete, original test questions from scratch.

JOB SPECIFICATIONS:
{json.dumps(job_specs, indent=2)}

{constraint_instruction}

MANDATORY OUTPUT FORMAT:
{{
  "questions": [
    {{
      "Item Number": "...",
      "Assessment Focus": "...",
      "Complete Sentence": "...",
      "Correct Answer": "...",
      "Context Clue Location": "...",
      "Context Clue Explanation": "...",
      "CEFR rating": "...",
      "Category": "..."
    }},
    ...
  ]
}}

STYLE REFERENCE:
{examples}
"""
    return system_msg, user_msg

def create_sequential_batch_stage2_grammar_prompt(job_list, stage1_outputs):
    system_msg = f"""You are an expert ELT test designer specializing in grammar assessment. You will generate candidate distractors for exactly {len(job_list)} grammar questions in a single JSON response with a "candidates" key."""
    
    user_msg = f"""
TASK: Generate 5 candidate distractors for ALL {len(job_list)} GRAMMAR questions.

INPUT FROM STAGE 1:
{json.dumps(stage1_outputs, indent=2)}

GENERATION INSTRUCTIONS:
1. WORD COUNT LIMIT: Max 3 words.
2. GRAMMATICAL PARALLELISM: Match word count and construction type of correct answer.
3. NO LEXICAL OVERLAP: Do NOT repeat words from the question stem.
4. TARGET FORM COVERAGE: For "vs" topics, include both forms.

MANDATORY OUTPUT FORMAT:
{{
  "candidates": [
    {{
      "Item Number": "...",
      "Candidate A": "...",
      "Candidate B": "...",
      "Candidate C": "...",
      "Candidate D": "...",
      "Candidate E": "..."
    }},
    ...
  ]
}}
"""
    return system_msg, user_msg

def create_sequential_batch_stage2_vocabulary_prompt(job_list, stage1_outputs):
    system_msg = f"""You are an expert ELT test designer specializing in vocabulary assessment. You will generate candidate distractors for exactly {len(job_list)} vocabulary questions in a single JSON response with a "candidates" key."""
    
    user_msg = f"""
TASK: Generate 5 candidate distractors for ALL {len(job_list)} VOCABULARY questions.

INPUT FROM STAGE 1:
{json.dumps(stage1_outputs, indent=2)}

GENERATION INSTRUCTIONS:
1. WORD COUNT LIMIT: Max 3 words.
2. EXACT INFLECTIONAL FORM MATCHING: Candidates must match the grammatical form of the correct answer (e.g., if answer is "running", candidates must be gerunds).
3. SEMANTIC FIELD PROXIMITY: Candidates should be from the same semantic field.

MANDATORY OUTPUT FORMAT:
{{
  "candidates": [
    {{
      "Item Number": "...",
      "Candidate A": "...",
      "Candidate B": "...",
      "Candidate C": "...",
      "Candidate D": "...",
      "Candidate E": "..."
    }},
    ...
  ]
}}
"""
    return system_msg, user_msg

def create_sequential_batch_stage3_grammar_prompt(job_list, stage1_outputs, stage2_outputs):
    system_msg = f"""You are an expert English grammar validator. You will evaluate candidate distractors for exactly {len(job_list)} grammar questions and return your validated selections in a JSON object with a "validated" key."""
    
    validation_input = []
    for i, (job, s1, s2) in enumerate(zip(job_list, stage1_outputs, stage2_outputs)):
        validation_input.append({
            "Item Number": s1.get("Item Number", ""),
            "Complete Sentence": s1.get("Complete Sentence", ""),
            "Correct Answer": s1.get("Correct Answer", ""),
            "Candidates": [s2.get(f"Candidate {k}", "") for k in "ABCDE"]
        })
    
    user_msg = f"""
TASK: Validate candidate distractors for ALL {len(job_list)} GRAMMAR questions and select the final three distractors.

INPUT:
{json.dumps(validation_input, indent=2)}

VALIDATION PROCEDURE:
1. GRAMMATICAL CORRECTNESS TEST: Distractor must make the sentence grammatically INCORRECT.
2. PROFICIENCY CHECK: Errors must be appropriate for the CEFR level.

MANDATORY OUTPUT FORMAT:
{{
  "validated": [
    {{
      "Item Number": "...",
      "Selected Distractor A": "...",
      "Selected Distractor B": "...",
      "Selected Distractor C": "...",
      "Validation Notes": "..."
    }},
    ...
  ]
}}
"""
    return system_msg, user_msg

def create_sequential_batch_stage3_vocabulary_prompt(job_list, stage1_outputs, stage2_outputs):
    system_msg = f"""You are an expert English vocabulary validator. You will evaluate candidate distractors for exactly {len(job_list)} vocabulary questions and return your validated selections in a JSON object with a "validated" key."""
    
    validation_input = []
    for i, (job, s1, s2) in enumerate(zip(job_list, stage1_outputs, stage2_outputs)):
        validation_input.append({
            "Item Number": s1.get("Item Number", ""),
            "Complete Sentence": s1.get("Complete Sentence", ""),
            "Correct Answer": s1.get("Correct Answer", ""),
            "Candidates": [s2.get(f"Candidate {k}", "") for k in "ABCDE"]
        })
    
    user_msg = f"""
TASK: Validate candidate distractors for ALL {len(job_list)} VOCABULARY questions and select the final three distractors.

INPUT:
{json.dumps(validation_input, indent=2)}

VALIDATION PROCEDURE:
1. EXAMINER ACCEPTANCE TEST: Distractor must NOT be a valid correct answer.
2. UNIQUENESS CHECK: Distractor must be grammatically correct but semantically wrong.

MANDATORY OUTPUT FORMAT:
{{
  "validated": [
    {{
      "Item Number": "...",
      "Selected Distractor A": "...",
      "Selected Distractor B": "...",
      "Selected Distractor C": "...",
      "Validation Notes": "..."
    }},
    ...
  ]
}}
"""
    return system_msg, user_msg

def create_options_prompt(job, example_banks):
    return "System", "User"

def create_stem_prompt(job, options):
    return "System", "User"

def create_holistic_prompt(job, example_banks):
    return "System", "User"

# =============================================================================
# TAB 5: GRAMMAR LIST GENERATION (New)
# =============================================================================

def create_grammar_list_stage1_prompt(job_list, question_form):
    """
    Generates complete sentences focusing on specific grammar points.
    """
    question_form_instructions = {
        "Random Mix": "Use diverse question forms.",
        "Simple gap fill": "ALL questions must use simple gap fill format via a blank line.",
        "Dialogue completion": "ALL questions must use dialogue completion format.",
        "Error identification": "Create a sentence with an error related to the grammar point (if applicable type).",
        "Sentence transformation": "Provide a sentence and a keyword to transform it."
    }
    
    # Default to gap fill if not specified or complex
    form_instruction = question_form_instructions.get(question_form, question_form_instructions["Simple gap fill"])
    
    system_msg = f"""You are an expert ELT content creator. You will generate exactly {len(job_list)} complete test questions in a single JSON response targeting specific grammar items.

CRITICAL: Your entire response must be a JSON object with a "questions" key containing an array of exactly {len(job_list)} question objects."""
    
    job_specs = []
    for job in job_list:
        job_specs.append({
            "job_id": job['job_id'],
            "cefr": job['cefr'],
            "base_grammar_item": job['base_grammar'],
            "grammar_subtype": job.get('subtype', ''),
        })
 
    user_msg = f"""
TASK: Create exactly {len(job_list)} grammar test questions.

GRAMMAR TARGETS:
{json.dumps(job_specs, indent=2)}

{form_instruction}

GENERATION INSTRUCTIONS:
1. **CONTEXT:** Create a natural sentence that REQUIRES the specific "Base Grammar Item" and "Subtype".
2. **TARGET FOCUS:** The "Correct Answer" must be the specific grammar structure requested.
   - Example Target: "Present Perfect" + "For"
   - Sentence: "I have lived here ____ ten years."
   - Correct Answer: "for"
3. **CLARITY:** Ensure the context makes the answer unambiguous.

MANDATORY OUTPUT FORMAT:
{{
  "questions": [
    {{
      "Item Number": "...",
      "Target Grammar": "...",
      "Subtype": "...",
      "Complete Sentence": "...",
      "Correct Answer": "...",
      "Context Explaination": "...",
      "CEFR rating": "...",
      "Category": "Grammar"
    }},
    ... 
  ]
}}
"""
    return system_msg, user_msg

def create_grammar_list_stage2_prompt(job_list, stage1_outputs):
    """
    Generates distractors for Grammar List items.
    """
    system_msg = f"""You are an expert ELT test designer. You will create exactly 4 candidate distractors for exactly {len(job_list)} grammar questions in JSON format."""
    
    pre_selected_data = []
    
    for i, job in enumerate(job_list):
        stage1_data = stage1_outputs[i]
        
        pre_selected_data.append({
            "Item Number": stage1_data.get("Item Number"),
            "Target Grammar": job['base_grammar'],
            "Subtype": job.get('subtype', ''),
            "Complete Sentence": stage1_data.get("Complete Sentence"),
            "Correct Answer": stage1_data.get("Correct Answer")
        })
    
    user_msg = f"""
TASK: Create a pool of exactly 4 candidate distractors for each question.
OUTPUT FORMAT: JSON

INPUT DATA:
{json.dumps(pre_selected_data, indent=2)}

INSTRUCTIONS FOR DISTRACTOR GENERATION:
1. **COMMON ERRORS:** Focus on common learner mistakes for the specific CEFR level and grammar point.
2. **PLAUSIBILITY:** Distractors should look grammatically possible but be incorrect in the specific context.
3. **RANGE:** 
   - Include wrong tenses.
   - Include wrong prepositions.
   - Include L1 interference errors if common.

MANDATORY OUTPUT FORMAT:
{{
  "candidates": [
    {{
      "Item Number": "...",
      "Candidate A": "...",
      "Candidate B": "...",
      "Candidate C": "...",
      "Candidate D": "...",
      "Distractor Notes": "e.g. Incorrect tense 'have went'"
    }},
    ... 
  ]
}}
"""
    return system_msg, user_msg

def create_grammar_list_stage3_prompt(job_list, stage1_outputs, stage2_outputs):
    """
    Validates Grammar List distractors.
    """
    system_msg = f"""You are an expert English grammar validator. You will filter candidate distractors using strict grammatical rules. Output results in JSON format."""
    
    validation_input = []
    for i, (job, s1, s2) in enumerate(zip(job_list, stage1_outputs, stage2_outputs)):
        candidates = []
        if isinstance(s2, dict):
            # Try to get candidate fields A-D
            candidates = [s2.get(f"Candidate {k}", "") for k in "ABCD"]
            # Filter out empty strings
            candidates = [c for c in candidates if c]
        
        validation_input.append({
            "Item Number": s1.get("Item Number", ""),
            "Complete Sentence": s1.get("Complete Sentence", ""),
            "Correct Answer": s1.get("Correct Answer", ""),
            "Candidates": candidates
        })
    
    user_msg = f"""
TASK: Validate candidates and select the final 3 distractors per question.
OUTPUT FORMAT: JSON

INPUT:
{json.dumps(validation_input, indent=2)}

VALIDATION PROTOCOL:
1. **DEFINITELY INCORRECT:** Ensure the distractor is not a valid alternative answer.
2. **CONTEXTUAL FIT:** The distractor might fit grammatically but be semantically weird (optional), or fit semantically but be grammatically wrong (preferred for grammar tests).

MANDATORY OUTPUT FORMAT:
{{
  "validated": [
    {{
      "Item Number": "...",
      "Selected Distractor A": "...",
      "Selected Distractor B": "...",
      "Selected Distractor C": "...",
      "Validation Notes": "..."
    }},
    ...
  ]
}}
"""
    return system_msg, user_msg
