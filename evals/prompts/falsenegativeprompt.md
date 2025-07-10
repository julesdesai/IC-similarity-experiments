# Instructions for Generating False-Negative Pairs

## Objective
Generate pairs of philosophical positions that appear different in wording but represent identical views (false negatives). These pairs will test whether semantic similarity methods can recognize the same philosophical position expressed in substantially different language.

## Input
You have been provided with a JSON file containing various consequentialist positions, each with:
- `position_id`: A unique identifier
- `summary`: The position name
- `theses`: Four conjunctive statements that together define the position

## Output Format
Generate a JSON file with the following structure:
```json
{
  "false_negative_pairs": [
    {
      "pair_id": "fn_001",
      "original_position": {
        "position_id": "...",
        "summary": "...",
        "theses": [...]
      },
      "reformulated_position": {
        "position_id": "..._reformulated",
        "summary": "...",
        "theses": [...],
        "reformulation_strategies": ["...", "..."],
        "preserves_meaning": true
      }
    }
  ]
}
```

## Reformulation Methods

Apply ALL of the following methods to each thesis to maximize lexical variation while preserving exact philosophical meaning:

### Method 1: Syntactic Transformation
Radically alter sentence structure while preserving logical content.

**Transformation types:**
- Active to passive voice: "X determines Y" → "Y is determined by X"
- Nominalization: "Actions are right if..." → "The rightness of actions depends on..."
- Clause restructuring: "If X then Y" → "Y follows from X" or "X entails Y"
- Predicate inversion: "X maximizes Y" → "Y reaches its maximum through X"

### Method 2: Vocabulary Substitution
Replace philosophical terms with exact synonyms or equivalent phrases.

**Substitution mappings:**
- "consequences" → "outcomes" / "results" / "effects"
- "utility" → "well-being" / "welfare" / "benefit"
- "maximize" → "produce the most" / "optimize" / "achieve the highest level of"
- "rightness" → "moral correctness" / "ethical status"
- "depends on" → "is determined by" / "is a function of" / "derives from"
- "if and only if" → "exactly when" / "precisely in cases where"
- "solely" → "exclusively" / "only" / "entirely"
- "produces" → "brings about" / "causes" / "results in"
- "moral" → "ethical" (and vice versa)
- "action" → "act" / "conduct" / "behavior"

### Method 3: Technical-Colloquial Translation
Convert between formal philosophical language and accessible prose.

**Examples:**
- Technical: "An action is morally permissible if and only if..."
- Colloquial: "It's okay to do something exactly when..."

- Technical: "The moral status of X is determined by its consequential properties"
- Colloquial: "What makes X right or wrong is what happens as a result"

### Method 4: Logical Equivalence Transformation
Express the same logical relationships using different structures.

**Equivalence patterns:**
- "All X are Y" → "There are no X that are not Y" / "Every X is Y"
- "X if and only if Y" → "X exactly when Y" / "X is equivalent to Y"
- "X requires Y" → "Y is necessary for X" / "Without Y, there is no X"
- "The more X, the more Y" → "Y increases with X" / "X and Y are positively correlated"

### Method 5: Expansion and Compression
Vary the level of detail while maintaining the same content.

**Examples:**
- Compressed: "Actions are right if they maximize utility"
- Expanded: "An action possesses the property of moral rightness in those cases where it produces a greater amount of overall utility than any available alternative action"

## Reformulation Requirements

1. **Complete Meaning Preservation**: 
   - Every reformulated position must be logically equivalent to the original
   - No philosophical content may be added or removed
   - The truth conditions must remain identical

2. **Maximum Lexical Distance**:
   - Minimize word overlap between original and reformulated theses
   - Aim for <30% vocabulary overlap (excluding function words)
   - Use different sentence structures for each thesis

3. **Stylistic Consistency**:
   - The four reformulated theses should have a consistent style
   - Choose a different overall register than the original (formal↔informal, technical↔accessible)

4. **Natural Language**:
   - Reformulations must read naturally and fluently
   - Avoid awkward constructions that sacrifice clarity for variation
   - Maintain philosophical precision

## Quality Validation Checklist

For each reformulated thesis, verify:
1. **Logical Equivalence**: Would a philosopher agree these express the same claim?
2. **Lexical Distance**: Do the versions share minimal vocabulary?
3. **Readability**: Does the reformulation flow naturally?
4. **Completeness**: Does it capture all aspects of the original thesis?

## Strategies for Complete Position Reformulation

1. **Choose a Reformulation Theme**:
   - "Academic to Plain English"
   - "British to American philosophical style"
   - "Textbook to Research Paper"
   - "Positive to Negative framing" (e.g., "maximizes good" → "minimizes opportunity cost of forgone good")

2. **Systematic Variation**:
   - If the original uses "if...then", use different logical connectives in the reformulation
   - If the original is agent-focused, make the reformulation outcome-focused
   - If the original uses examples, use abstractions (and vice versa)

3. **Cross-Thesis Consistency**:
   - Ensure terminology is used consistently across all four reformulated theses
   - Maintain parallel structure where the original has it, but with different syntax

## Example Generation

**Original Thesis**: 
"An action is right if and only if it produces the greatest overall utility compared to alternative actions"

**Reformulated Thesis**:
"The moral correctness of conduct is established exactly when that conduct generates more total well-being than any other available option"

**Changes Applied**:
- "action" → "conduct"
- "right" → "moral correctness"
- "if and only if" → "exactly when"
- "produces" → "generates"
- "greatest overall utility" → "more total well-being"
- "compared to alternative actions" → "than any other available option"
- Active to passive transformation in the subject

## Quantity and Coverage

- Generate at least 20 false-negative pairs
- Include positions from different subfamilies of consequentialism
- Vary the reformulation intensity (some moderate changes, some radical restructuring)
- Document which reformulation strategies were dominant for each pair

## Additional Guidelines

1. **Avoid Trivial Changes**: Don't just swap word order or make minor substitutions
2. **Preserve Philosophical Precision**: Technical distinctions must be maintained even in colloquial reformulations
3. **Test the Equivalence**: Imagine explaining why these are the same position to a philosophy student
4. **Creative Liberty**: Feel free to completely restructure how the ideas are presented as long as the content is preserved