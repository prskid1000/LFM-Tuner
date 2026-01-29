# Initial dataset formats

Use these with `training.reasoning_mode` in config: `none`, `full`, or `masked`.

## Non–CoT (reasoning_mode: none)

- **File:** `initial_dataset_non_cot_example.json`
- Assistant content is the final answer only (no `<reasoning>` / `<final>`).
- Use when you do not want Chain-of-Thought.

## CoT (reasoning_mode: full or masked)

- **File:** `initial_dataset_cot_example.json`
- Assistant content uses tags:
  - `<reasoning>` … `</reasoning>` — step-by-step reasoning
  - `<final>` … `</final>` — user-facing answer
- **full:** train on both reasoning and final.
- **masked:** train only on `<final>`; `<reasoning>` tokens get loss mask -100.

## Mixed (allowed)

- **File:** `initial_dataset_mixed_example.json`
- Some samples with plain answers, some with `<reasoning>` + `<final>`.
- With `reasoning_mode: masked`, samples without `<reasoning>` are unchanged; samples with it have reasoning masked.

## Rules

- Same `messages` schema: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`.
- Assistant must be the last message.
- `<final>` must come after `<reasoning>` when CoT is used.
- Tags are plain text (no special tokens).
