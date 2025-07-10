# Instructions for Generating False-Positive Pairs

## Objective
Generate pairs of philosophical positions that appear similar in wording but represent materially different views (false positives). These pairs will test whether semantic similarity methods can distinguish between superficially similar but philosophically distinct positions.

## Input
You have been provided with a JSON file containing various consequentialist positions, each with:
- `position_id`: A unique identifier
- `summary`: The position name
- `theses`: Four conjunctive statements that together define the position

## Output Format
Generate a JSON file with the following structure:
```json
{
  "false_positive_pairs": [
    {
      "pair_id": "fp_001",
      "original_position": {
        "position_id": "...",
        "summary": "...",
        "theses": [...]
      },
      "modified_position": {
        "position_id": "..._modified",
        "summary": "...",
        "theses": [...],
        "modification_type": "...",
        "key_difference": "..."
      }
    }
  ]
}
```

## Generation Methods

For each original position, create a modified version using ONE of these methods:

### Method 1: Critical Term Substitution
Identify the most philosophically load-bearing term in one thesis and replace it with a closely related but distinctly different term.

**Examples of substitutions:**
- "maximize" → "satisfice" or "optimize"
- "all" → "most" or "relevant"
- "solely" → "primarily" or "largely"  
- "if and only if" → "if" or "when"
- "must" → "should" or "ought"
- "actual" → "expected" or "probable"
- "intrinsic" → "instrumental" or "final"
- "right" → "permissible" or "praiseworthy"

### Method 2: Scope Modification
Alter the scope of a claim while maintaining similar vocabulary.

**Examples:**
- "all consequences" → "direct consequences"
- "every individual" → "each affected individual"
- "actions" → "intentional actions"
- "total utility" → "aggregate utility"
- "produces" → "tends to produce"

### Method 3: Logical Operator Adjustment
Modify logical relationships between concepts.

**Examples:**
- "depends solely on" → "depends primarily on"
- "requires" → "permits" or "recommends"
- "if and only if" → "only if" or "if"
- "necessary and sufficient" → "necessary" or "sufficient"

### Method 4: Threshold Introduction/Removal
Add or remove threshold concepts that fundamentally alter the position.

**Examples:**
- Add "sufficient" before a maximizing claim
- Add "above a threshold" to universal claims
- Remove "only" from exclusive claims
- Add "typically" or "generally" to absolute claims

## Constraints and Guidelines

1. **Philosophical Validity**: The modified position should either:
   - Represent a real philosophical position (even if not in the original dataset)
   - Be a coherent theoretical possibility that philosophers might defend

2. **Subtlety Requirement**: 
   - Change exactly ONE thesis
   - The modification should affect 1-3 words maximum
   - The overall structure and vocabulary should remain similar

3. **Preservation Requirements**:
   - Keep the same number of theses (4)
   - Maintain grammatical correctness
   - Preserve the logical flow between theses

4. **Documentation**:
   - Specify which modification method was used in `modification_type`
   - Clearly state what makes the positions different in `key_difference`

## Quality Checks

Before including a pair, verify:
1. The positions are genuinely different (would a philosopher distinguish them?)
2. The wording similarity is high (would a naive text similarity metric conflate them?)
3. The modified position is coherent and defensible
4. Only one substantive change has been made

## Example Generation

**Original**: Act Utilitarianism
- "An action is right if and only if it produces the greatest overall utility compared to alternative actions"

**Modified**: (Method 1 - Term Substitution)
- "An action is right if and only if it produces sufficient overall utility compared to alternative actions"
- `key_difference`: "Changed from maximizing to satisficing requirement"

## Quantity and Distribution

- Generate at least 100 false-positive pairs
- Use each modification method at least 4 times
- Include a variety of difficulty levels (some obvious, some subtle)
- Avoid using the same original position more than twice

## Additional Notes

- If a modification would create a position already in the dataset, document this by referencing the existing position_id
- Prioritize modifications that create philosophically interesting distinctions
- Ensure the modified summary name reflects the change when appropriate