# OserAI Guide v0.1

This PEFT LoRA adapter uses Qwen3-0.6B at the revision recorded in
`adapter_config.json`. It was trained locally on 48 examples based on the
OserAI and Colibri guides, with 144 optimizer steps on an Apple M2.

On held-out prompts, both the adapter and the base model passed one of ten
OserAI keyword checks. The adapter passed two of four general checks and the
base passed three. Guide is available as a labeled preview for evaluation.

The [deployment guide](../../docs/chat-cpanel.md#offer-oserai-guide-v01-to-invited-testers)
explains how to export and serve the model for `/guide` replies.
