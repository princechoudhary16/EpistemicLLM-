"""Minimal supervised fine-tuning utilities (plan.md section 7: "ordinary LoRA if
memory permits" as the first sanity check before anything else).

Answer-only loss: the prompt (system + question) contributes 0 to the loss; only the
completion (answer) tokens are trained on. This matches plan.md section 7 ("Use
answer-only language-model loss, excluding prompt/padding tokens") and the Q->A training
format P1's synthetic generator also uses, so this trainer can later run on P1's real
corpus unchanged.

Deliberately minimal: batch size 1, no gradient accumulation, no LR schedule. Enough for
Week 1's "run a tiny standard SFT" sanity check -- proving the training pipeline itself
works -- not the throughput-tuned loop the main experiments will need (those settings
get frozen only after measuring real throughput, per plan.md section 3).
"""

import torch


def build_example(tokenizer, system_prompt, question, answer, device):
    """Tokenizes one (question, answer) pair with an answer-only loss mask.

    Returns (input_ids, labels), both shape (1, seq_len). labels is -100 (PyTorch's
    "ignore this position" sentinel for cross-entropy) over the prompt, then the
    answer's own token ids (+ EOS). transformers' CausalLM forward does the
    next-token shift internally, so labels are aligned 1:1 with input_ids here, not
    pre-shifted.
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
    return (
        torch.tensor(input_ids, device=device).unsqueeze(0),
        torch.tensor(labels, device=device).unsqueeze(0),
    )


def train_lora_sft(model, tokenizer, examples, system_prompt, epochs, lr, device, seed=0):
    """Trains `model` (already LoRA-wrapped) in place, one example per step.

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

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    for _ in range(epochs):
        for question, answer in examples:
            input_ids, labels = build_example(tokenizer, system_prompt, question, answer, device)
            out = model(input_ids=input_ids, labels=labels)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            losses.append(out.loss.item())
    model.eval()
    return losses, n_trainable
