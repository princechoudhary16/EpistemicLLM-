"""Minimal supervised fine-tuning utilities (plan.md section 7: "ordinary LoRA if
memory permits" as the first sanity check before anything else).

Answer-only loss: the prompt (system + question) contributes 0 to the loss; only the
completion (answer) tokens are trained on. This matches plan.md section 7 ("Use
answer-only language-model loss, excluding prompt/padding tokens") and the Q->A training
format P1's synthetic generator also uses, so this trainer can later run on P1's real
corpus unchanged.

Supports batch_size > 1 (right-padded, with an attention mask) so this trainer can
exactly reproduce a different team member's recipe for training-parity comparisons
(e.g. P1/Aryaman's ablation recipe -- same padding/collation approach as
p1_exposure/ablation/run_ablation.py's encode_example/collate) -- not because P6's own
canonical recipe needs batching for throughput.
"""

import torch


def encode_example(tokenizer, system_prompt, question, answer):
    """Tokenizes one (question, answer) pair with an answer-only loss mask.

    Returns (input_ids, labels) as plain Python lists (not yet padded/batched/tensored
    -- collate() does that). labels is -100 (PyTorch's "ignore this position" sentinel
    for cross-entropy) over the prompt, then the answer's own token ids (+ EOS).
    transformers' CausalLM forward does the next-token shift internally, so labels are
    aligned 1:1 with input_ids here, not pre-shifted.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Same call style as ModelRunner.generate_one's prompt tokenization, so training
    # and inference encode the prompt identically.
    prompt_ids = tokenizer(prompt)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        answer_ids = answer_ids + [eos_id]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels


def collate(batch, pad_id, device):
    """Right-pads a list of (input_ids, labels) pairs to the batch's max length and
    builds the matching attention mask. Padded label positions are -100 (ignored)."""
    n = max(len(ids) for ids, _ in batch)
    ids = [i + [pad_id] * (n - len(i)) for i, _ in batch]
    lab = [l + [-100] * (n - len(l)) for _, l in batch]
    att = [[1] * len(i) + [0] * (n - len(i)) for i, _ in batch]
    return (
        torch.tensor(ids, device=device),
        torch.tensor(lab, device=device),
        torch.tensor(att, device=device),
    )


def train_lora_sft(model, tokenizer, examples, system_prompt, epochs, lr, device, seed=0, batch_size=1):
    """Trains `model` (already LoRA-wrapped) in place, batch_size examples per step
    (default 1: one example per step, no padding needed).

    Returns (losses, n_trainable): the per-step loss in order (to check it actually
    decreases -- a basic learnability check, not a claim about final result quality),
    and the number of trainable parameters (to confirm the adapter actually attached;
    a base model with everything frozen would train 0 parameters and silently do
    nothing while still "running" without error).
    """
    torch.manual_seed(seed)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if n_trainable == 0:
        raise RuntimeError("no trainable parameters -- LoRA adapter did not attach correctly")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = [encode_example(tokenizer, system_prompt, q, a) for q, a in examples]

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    for _ in range(epochs):
        for i in range(0, len(encoded), batch_size):
            input_ids, labels, attention_mask = collate(encoded[i:i + batch_size], pad_id, device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            losses.append(out.loss.item())
    model.eval()
    return losses, n_trainable
