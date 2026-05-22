---
layout: default
title: Can an Autoregressive VLM Control a Robot in Real Time? (Part 1)
description: lalala

citation_title: "Can an Autoregressive VLM Control a Robot in Real Time? (Part 1)"
citation_authors: 
  - "Oleg Balakhnov"
  - "Sergei Skvortsov"
citation_date: "2026/05/22"
math: true

header_links:
  - text: View on GitHub
    url: https://github.com/acting-machines/lerobot/tree/dev
  - text: Model weights
    url: https://huggingface.co/olegbalakhnov/vla-0-smol-spec

---

# Introduction

Robots need to react fast. A policy that takes three seconds to decide what to do next is not a policy you can deploy on a real arm. This creates an obvious tension with large vision-language models: they are powerful, general, and increasingly capable of understanding and acting in the world - but autoregressive decoding is slow by nature.

In our previous work, VLA-0-Smol [[1]](https://robot-learning-collective.github.io/vla-0-smol/), we showed that a standard VLM can learn to output robot actions directly as text tokens, with no architectural changes and no custom action heads. This "actions-as-text" setup has a lot of practical advantages:
- No architecture changes. We don't need special action heads or a custom vocabulary.
- Training is stable and fast. Inputs and targets stay in the same domain the model was pretrained on (tokens), and we can reuse mature VLM training stacks.
- One model for reasoning + control. The same backbone can describe what it sees, plan in language, and output actions.
- The software ecosystem is great. Debugging, logging, evaluation, and serving infrastructure is much more developed for VLMs than for many robotics-specific policies.

But we left one big question open: can it actually run fast enough to control a robot in real time? Autoregressive decoding is slow by nature, and in robotics, latency is not just "nice to have" - it directly limits the control rate, the smoothness of motion, and how well the policy can react to changes.

This post is our attempt to answer that question concretely. We set a measurable target: 
<div style="
  background: rgba(59,130,246,0.08);
  border-left: 6px solid #3b82f6;
  padding: 16px 20px;
  margin: 24px 0;
  border-radius: 8px;
">
  Keep the delay between receiving a camera image and sending an action to the motors below 100 ms, on a 6-DoF arm with a 1-DoF gripper.
</div>

And we treated it as a systems problem rather than a modeling one - keeping the same VLM backbone and actions-as-text representation throughout.

Two changes got us close. 
- Streaming actions out of the model as they are generated, rather than waiting for a full chunk to be decoded before the robot moves.
- Adding an EAGLE-style speculative decoder to make each individual generation step cheaper. Together, they bring latency from over 3.5 seconds down to around 100 ms, while keeping LIBERO task performance competitive [[5]](https://arxiv.org/abs/2306.03310).

The takeaway is that the bottleneck was never the actions-as-text idea itself. It was how inference was scheduled and executed. Fix those, and real-time VLM-based robot control starts to look practical.

# Baseline: Blocking Autoregressive VLA Control

Our baseline policy follows the VLA-0-Smol setup: a standard vision–language model is fine-tuned to output robot actions as ordinary text tokens. Instead of adding a continuous action head or a robotics-specific decoder, we express each action as a short sequence of discrete numbers in the model’s text vocabulary. The policy is therefore trained exactly like a language model: given an image, task description, and robot state, predict the next action token.

The target model is based on **SmolVLM2**, a compact vision–language model with an image encoder and an autoregressive language-model decoder. During training, the camera observation is inserted into the prompt as image tokens, while the task, robot state, and action prefix are represented as text. The target output is a sequence of action tokens corresponding to the next robot command, or to a short chunk of future commands.

The sequence looks like this:

<div style="font-family: 'SF Mono', Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; overflow-x: auto; line-height: 2.4; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
  <div style="margin-bottom: 12px; border-left: 4px solid #94a3b8; padding-left: 12px;">
    <span style="background: #f1f5f9; color: #475569; padding: 4px 8px; border-radius: 4px; #93c5fd; font-size: 0.9em; margin-right: 4px;">&lt;|im_start|&gt;User:</span>
    <span style="background: #eff6ff; color: #1e40af; padding: 4px 8px; border-radius: 4px; border: 1px solid #bfdbfe; font-size: 0.9em; margin-right: 4px;">&lt;fake_token_around_image&gt;&lt;global-img&gt;</span>
    <span style="background: #dbeafe; color: #1e40af; padding: 4px 8px; border-radius: 4px; border: 1px solid #93c5fd; font-size: 0.9em; margin-right: 4px;">
    <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: middle; margin-right: 2px;"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg> [Image Embeddings (64 patches)]</span> <span style="background: #eff6ff; color: #1e40af; padding: 4px 8px; border-radius: 4px; border: 1px solid #bfdbfe; font-size: 0.9em; margin-right: 8px;">&lt;fake_token_around_image&gt;</span>
    <span style="background: #f0fdf4; color: #166534; padding: 4px 8px; border-radius: 4px; border: 1px solid #bbf7d0; font-size: 0.9em; margin-right: 4px;">Task: Description of the task to be performed,</span>
    <span style="background: #fff7ed; color: #9a3412; padding: 4px 8px; border-radius: 4px; border: 1px solid #ffedd5; font-size: 0.9em; margin-right: 4px;">State: 341 182,</span>
    <span style="background: #fff1f2; color: #be123c; padding: 4px 8px; border-radius: 4px; border: 1px solid #fecdd3; font-size: 0.9em; margin-right: 4px; font-size: 0.9em;">Actions:</span>
    <span style="background: #f1f5f9; color: #475569; padding: 4px 8px; border-radius: 4px; font-size: 0.9em;">&lt;end_of_utterance&gt;</span>
  </div>
  <div style="border-left: 4px solid #818cf8; padding-left: 12px;">
    <span style="background: #eef2ff; color: #3730a3; padding: 4px 8px; border-radius: 4px; margin-right: 8px; font-size: 0.9em; ">Assistant:</span>
    <span style="background: #f5f3ff; color: #5b21b6; padding: 4px 8px; border-radius: 4px; border: 1px solid #ddd6fe; font-size: 0.9em;">227 232 223 191</span> <span style="background: #f1f5f9; color: #475569; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; margin-left: 6px;">&lt;end_of_utterance&gt; &lt;|im_end|&gt;</span>
  </div>
</div>
<p style="text-align: center; font-size: 0.9rem; color: #667; margin-top: 10px;"><em>Figure 1: The multimodal input sequence formatted for the SmolVLM2 chat template.</em></p>

The important detail for this post is that the action sequence is generated autoregressively. The model does not predict the full action sequence in one forward pass. Instead, it predicts one token, appends it to the context, predicts the next token, and repeats this process until the whole action chunk has been produced.
The naïve control loop therefore looks like this:

<div class="algorithm-box">
  <div class="algorithm-title">Algorithm 1: Naïve autoregressive VLA control loop</div>
  <p class="algorithm-note">Repeat while the robot is running:</p>
  <ol>
    <li>Receive a new camera observation and the current robot state.</li>
    <li>Build the VLM prompt.</li>
    <li>Run the model prefill on the image and text context.</li>
    <li>Decode the full action chunk autoregressively.</li>
    <li>Parse the generated tokens into robot actions.</li>
    <li>Execute the action chunk on the robot.</li>
  </ol>
</div>

This is the simplest way to run an autoregressive VLA policy, but it is also the most blocking one. The robot does not start moving until the full chunk has been decoded. If generation takes hundreds of milliseconds, the robot simply waits during that time. This creates visible pauses and makes the policy unsuitable for high-frequency control.

This problem becomes worse when we predict action chunks. Chunking is useful because it lets the policy generate several future actions from one observation, which can make behaviour smoother and more stable. But in an autoregressive text-token setup, predicting a longer chunk also means generating more tokens sequentially. For example, if one action contains several discretised values and we predict multiple future actions, the model may need many decoding steps before the robot receives anything.

In the previous VLA-0-Smol work, we also used temporal ensembling, which improved task performance. However, temporal ensembling is expensive in this setting because it requires repeatedly predicting overlapping future trajectories. Since the focus of this post is real-time control, we do not use temporal ensembling in the main runtime setup. We still report it as a reference point in the results, but our main baseline is the plain autoregressive policy without temporal ensembling.

With this blocking implementation, latency is far above our target. 

<div style="
  background: rgba(59,130,246,0.08);
  border-left: 6px solid #3b82f6;
  padding: 16px 20px;
  margin: 24px 0;
  border-radius: 8px;
">
  Generating a full action sequence of 8 actions for a 7-DoF robot takes 3625 ms, while our target is to keep the delay between receiving a camera image and sending an action to the motors below 100 ms. 
</div>

The policy itself is strong, but the naïve execution loop is too slow by more than an order of magnitude.

This gives us a clean systems problem: keep the same target VLA policy and the same actions-as-text representation, but change how inference is scheduled and accelerated. The next sections describe two steps in that direction: first, overlapping inference with robot control, and second, using speculative decoding to reduce the cost of autoregressive generation.

# Parallel Inference and Control

## Method

The blocking baseline wastes a lot of time because inference and execution happen sequentially. The model first generates a full action chunk,

\\[
A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}],
\\]

and only after the whole chunk has been decoded does the robot start executing it.
This is inefficient for an autoregressive policy. The model produces the action sequence token by token, and in our representation each robot action is available before the full chunk is finished. In other words, once the tokens for \\(a_t\\) have been generated and parsed, we already know the next action that should be sent to the robot. We do not need to wait for \\(a_{t+1}, \ldots, a_{t+H-1}\\).
The simplest improvement is therefore to overlap decoding and execution. Instead of treating the action chunk as something that must be fully generated before control starts, we treat it as a stream of actions.
In our implementation, the system is split into two parallel loops:

<div class="algorithm-box">
  <div class="algorithm-title">Algorithm 2: Parallel inference and control</div>
  <p class="algorithm-note">Run the following loops in parallel:</p>
  <div class="algorithm-grid">
    <div>
      <div class="algorithm-subtitle">Inference loop</div>
      <ol>
        <li>Observe the camera image and current robot state.</li>
        <li>Run the VLM prefill.</li>
        <li>Generate actions autoregressively until the EOS token is produced.</li>
        <li>Push each generated action into the action buffer.</li>
      </ol>
    </div>
    <div>
      <div class="algorithm-subtitle">Control loop</div>
      <ol>
        <li>Run at the fixed robot control rate.</li>
        <li>Read the next action from the action buffer.</li>
        <li>Send the action to the robot.</li>
      </ol>
    </div>
  </div>
</div>

The action buffer is the only communication point between the two loops. The inference loop continuously produces actions and pushes them into the buffer. The control loop runs at the desired control frequency, for example 10 Hz, and consumes one action at a time.

This does not make the model itself faster. The total compute required to generate the full chunk is unchanged. However, it changes what the robot is waiting for. In the blocking baseline, the robot waits for the whole chunk. With parallel inference and control, the robot only needs to wait for the first executable action.

For a chunk of H actions, this can be a large improvement. In our setup, the chunk contains 8 actions, so overlapping inference and control can reduce the initial waiting time by roughly the chunk size, assuming actions are generated at a similar rate.

There is one important trade-off. All actions in the chunk are still generated from the observation used at the start of decoding. This means that later actions are based on slightly older visual information. However, this is already a common trade-off in action-chunking and trajectory-following systems. As long as the chunk horizon is short and the control loop keeps receiving new observations frequently, the system can remain responsive while avoiding large idle gaps.

The key point is that this optimisation is almost free. It does not require changing the model, retraining the policy, or modifying the action representation. It only changes the runtime scheduling.

## Results

With the blocking autoregressive baseline, we measured:

<div style="margin: 24px 0; overflow-x: auto;">
  <table style="display: inline-table; width: auto; min-width: 520px; border-collapse: collapse; background: #ffffff; border: 0;">
    <caption style="caption-side: bottom; padding-top: 8px; color: #4b5563; font-size: 0.9rem; text-align: left;">Table 1: Autoregressive baseline latency.</caption>
    <thead>
      <tr style="background: #f3f4f6;">
        <th style="text-align: left; padding: 12px 14px; color: #111827; font-size: 0.9rem; font-weight: 700; border: 0; border-bottom: 1px solid #d1d5db;">Measurement</th>
        <th style="text-align: right; padding: 12px 14px; color: #111827; font-size: 0.9rem; font-weight: 700; border: 0; border-bottom: 1px solid #d1d5db;">Latency (ms)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827;">Full action chunk</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">3625</td>
      </tr>
      <tr>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827;">First executable action</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">489</td>
      </tr>
      <tr>
        <td style="padding: 12px 14px; border: 0; color: #111827;">Later actions</td>
        <td style="padding: 12px 14px; border: 0; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">448</td>
      </tr>
    </tbody>
  </table>
</div>

By starting execution as soon as the first action is available, the robot no longer needs to wait for the full 3625 ms sequence generation. The effective image-to-first-action delay becomes approximately 489 ms.

This is still far from our target of less than 100 ms, so parallel inference and control alone is not enough for real-time control. However, it gives a large improvement without any loss in model quality or any change to the policy itself.

It also composes naturally with later optimisations. If we can reduce the time to generate one action, the benefit of streaming execution remains: the robot still avoids waiting for the full chunk, and each generated action can be executed as soon as it is ready. In other words, this scheduling change turns future decoding speedups directly into lower control latency.

# Speculative Decoding

## Method

After overlapping inference and control, the robot no longer waits for the full action chunk. However, it still waits for the model to generate the next action. In the baseline, this is still too slow: generating the first action takes around 489 ms, while our target is below 100 ms.

The next question is therefore:

<div style="
  background: rgba(59,130,246,0.08);
  border-left: 6px solid #3b82f6;
  padding: 16px 20px;
  margin: 24px 0;
  border-radius: 8px;
">
  Can we reduce the cost of autoregressive decoding itself?
</div>

The fundamental bottleneck is that autoregressive models generate one token per forward pass. Each pass runs the full network, attention over the entire context, all layers, all parameters just to produce a single token. For a large VLM, this is expensive, and there is no way around it within standard autoregressive decoding.

Several techniques exist to reduce this cost. The most direct approaches: quantization, pruning, and optimized attention kernels make each forward pass cheaper. But they still generate one token per pass. Speculative decoding takes a different angle: instead of making each step cheaper, it makes each step produce more tokens. It has become a common acceleration technique for LLM inference [[4]](https://arxiv.org/abs/2211.17192). Recent models such as DeepSeek-V3 use multi-token prediction during training [[2]](https://arxiv.org/abs/2412.19437).

We focus on speculative decoding because it is lossless in the standard setting and composes well with other optimizations. The core idea is simple: instead of asking the large target model to generate one token at a time, we add a smaller and cheaper draft model that proposes several future tokens, and let the target model verify them in a single pass.

The standard speculative decoding loop is:

<div class="algorithm-box">
  <div class="algorithm-title">Algorithm 3: Standard speculative decoding</div>
  <p class="algorithm-note">Repeat until generation is complete:</p>
  <ol>
    <li>Use the draft model to propose several candidate tokens.</li>
    <li>Run the target model once on the full candidate sequence.</li>
    <li>Accept the longest prefix that matches what the target model would have generated.</li>
    <li>If a token is rejected, resample at that position from the target model's distribution.</li>
  </ol>
</div>

The benefit comes from accepting more than one token per expensive target-model step. If the draft model proposes 5 tokens and 4 are accepted, we effectively generate 4 tokens for the cost of one verification pass plus a cheap draft step. The key metric is the acceptance rate: the higher it is, the closer we get to the theoretical maximum speedup.

Speculative decoding is particularly interesting for robot control. In natural language, tokens are highly context-dependent, the right next word depends heavily on everything that came before. Robot action sequences have a different structure. Consecutive actions within a chunk are related, but the individual dimensions within a single action (joint angles, gripper state) are largely independent of each other given the overall motion context. Our hypothesis is that this structure makes action tokens easier for a small draft model to predict accurately, which should translate into higher acceptance rates and larger speedups than are typically seen in open-ended text generation.

## EAGLE-3

EAGLE-3 is a speculative decoding method built on top of the original EAGLE framework [[3]](https://arxiv.org/abs/2503.01840).

Standard speculative decoding uses a separate smaller LLM as the draft model, which operates independently of the target model. EAGLE-3 improves on this by giving the draft model access to internal features from the target model — specifically, a fusion of low-, mid-, and high-level hidden states — projected through a small FC layer. This richer signal makes the draft model's predictions much more accurate than a standalone smaller model could achieve, which translates directly into higher acceptance rates and larger speedups.

<div style="margin: 28px 0;">
  <img src="{{ '/_assets/eagle_inference.png' | relative_url }}" style="display: block; width: min(50%, 720px); height: auto; margin: 0 auto;">
  <p style="text-align: center; font-size: 0.9rem; color: #667; margin-top: 10px;"><em>Figure 2: EAGLE-3 inference, from [3]. The target model exposes hidden states from several depths, these states are fused by a small projection layer, and the resulting features condition a lightweight draft decoder that proposes multiple future tokens before target-model verification.</em></p>
</div>

The figure shows the EAGLE-3 draft path used during inference. The target model produces the usual next-token distribution, but it also supplies low-, middle-, and high-level representations for the current prefix. These representations are concatenated and projected into draft features, which the small decoder reuses across several speculative steps. Each step extends the tentative sequence, so the target model can later verify several candidate tokens with one expensive pass instead of generating them one at a time.

The other key innovation is training-time test: during training, the draft model's own outputs are fed back as inputs, simulating exactly what happens during multi-step drafting at inference time. This prevents error accumulation when the draft model runs for several steps without target-model corrections between them. It also removes the need for a feature prediction loss, freeing the draft model to focus entirely on token prediction and allowing it to scale more effectively with additional training data.


In practice, EAGLE-3 achieves up to 6.5× speedup over vanilla autoregressive decoding on standard benchmarks.

## Our modifications

Our implementation differs from the standard EAGLE-3 setup in two important ways.

First, we train the base policy and the draft module together. Because of this, we compute the loss on the target action tokens, not on reproducing the base model’s own predictions. The reason is simple: we ultimately care about generating correct actions, not about imitating the current base model. Also DeepSeek-V3 authors claim that multi-token prediction loss densifies the training signal and enable the model to pre-plan its representation for better prediction of future tokens [[2]](https://arxiv.org/abs/2412.19437).

Second, in this experiment we evaluate a version without verification. In normal text generation, verification is important because a wrong token becomes part of the generated prefix and can permanently change the rest of the sentence. In robot control, the situation is a bit different. A slightly imperfect action is not necessarily catastrophic: the robot receives new observations, the policy runs again, and future actions can correct small errors. But we still evaluate performance of the base model to see a difference in downstream tasks performance.

The total training loss combines the base model's next-token prediction loss with the draft heads' multi-token prediction losses:

\\[
L = L_{\mathrm{VLM}} + \sum_{k=1}^{K} L_{\mathrm{MTP}}^{(k)}
\\]

where \\(L_{\mathrm{VLM}}\\) is the mean cross-entropy loss over action tokens for the base model, \\(K\\) is the number of draft heads (5 in our setup), and \\(L_{\mathrm{MTP}}^{(k)}\\) is the cross-entropy loss of the \\(k\\)-th draft head predicting tokens \\(k\\) steps ahead. All losses are masked to action tokens only — image tokens and task description tokens do not contribute to the gradient.



This version should therefore be read as an approximation to speculative decoding, not as a complete replacement for verified EAGLE-3 inference, therefore we will call it EAGLE-style. In the next step, when using an inference engine with built-in speculative decoding support, verification can be added back.

## Results

We trained the target model and draft model jointly, with gradients flowing through both. All VLM components: vision encoder, connector, and language backbone were unfrozen, as we found this consistently improves task performance. Joint training is motivated by DeepSeek-V3's finding that multi-token prediction loss densifies the gradient signal and encourages the model to build representations that are better suited for predicting future tokens.

The draft model is a single Transformer decoder layer with the same hidden dimension as the target model, augmented with the multi-layer feature fusion head described above. It was trained to predict 5 tokens ahead. Both the target and draft models are trained with cross-entropy loss, summed together into a single objective.
Training ran for 100k steps on the LIBERO dataset with batch size 192, learning rate 5e-5, across 8 H100 GPUs with DDP.

### Latency results
We measured latency on an H100 GPU [[6]](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306).

<div style="margin: 24px 0; overflow-x: auto;">
  <table style="display: inline-table; width: auto; min-width: 760px; border-collapse: collapse; background: #ffffff; border: 0;">
    <caption style="caption-side: bottom; padding-top: 8px; color: #4b5563; font-size: 0.9rem; text-align: left;">Table 2: Decoder latency comparison.</caption>
    <thead>
      <tr style="background: #f3f4f6;">
        <th style="text-align: left; padding: 12px 14px; color: #111827; font-size: 0.9rem; font-weight: 700; border: 0; border-bottom: 1px solid #d1d5db;">Decoder</th>
        <th style="text-align: right; padding: 12px 14px; color: #111827; font-size: 0.9rem; font-weight: 700; border: 0; border-bottom: 1px solid #d1d5db;">First action (ms)</th>
        <th style="text-align: right; padding: 12px 14px; color: #111827; font-size: 0.9rem; font-weight: 700; border: 0; border-bottom: 1px solid #d1d5db;">Later actions (ms)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827;">Autoregressive baseline</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">489</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">448</td>
      </tr>
      <tr>
        <td style="padding: 12px 14px; border: 0; color: #111827;">EAGLE-style, 5 heads, unverified</td>
        <td style="padding: 12px 14px; border: 0; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">154</td>
        <td style="padding: 12px 14px; border: 0; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">106</td>
      </tr>
    </tbody>
  </table>
</div>

This is a large improvement. With a new observation, latency drops from 489 ms to 154 ms. When reusing the same observation context, it drops from 448 ms to 106 ms.
That is still slightly above the 100 ms target, but it is much closer. More importantly, this result composes with the previous scheduling change. Parallel inference and control makes the robot wait only for the next action, and speculative decoding makes that next action much cheaper to generate.

### Task performance

The main risk of skipping verification is that speed comes at the cost of policy quality. To check this, we evaluated all variants on two LIBERO benchmarks.

<div style="margin: 24px 0; overflow-x: auto;">
  <table style="display: inline-table; width: auto; min-width: 720px; border-collapse: collapse; background: #ffffff; border: 0;">
    <caption style="caption-side: bottom; padding-top: 8px; color: #4b5563; font-size: 0.9rem; text-align: left;">Table 3: LIBERO task performance.</caption>
    <thead>
      <tr style="background: #f3f4f6;">
        <th style="text-align: left; padding: 12px 14px; color: #111827; font-size: 0.9rem; font-weight: 700; border: 0; border-bottom: 1px solid #d1d5db;">Model</th>
        <th style="text-align: right; padding: 12px 14px; color: #111827; font-size: 0.9rem; font-weight: 700; border: 0; border-bottom: 1px solid #d1d5db;">LIBERO-Goal</th>
        <th style="text-align: right; padding: 12px 14px; color: #111827; font-size: 0.9rem; font-weight: 700; border: 0; border-bottom: 1px solid #d1d5db;">LIBERO-Long</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827;">Autoregressive baseline</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">91.6</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">88.2</td>
      </tr>
      <tr>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827;">Baseline + temporal ensembling</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">95.6</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">91.2</td>
      </tr>
      <tr>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827;">EAGLE-style, unverified</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">94.8</td>
        <td style="padding: 12px 14px; border: 0; border-bottom: 1px solid #e5e7eb; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">88.4</td>
      </tr>
      <tr>
        <td style="padding: 12px 14px; border: 0; color: #111827;">Target model only</td>
        <td style="padding: 12px 14px; border: 0; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">93.2</td>
        <td style="padding: 12px 14px; border: 0; color: #111827; text-align: right; font-variant-numeric: tabular-nums;">88.8</td>
      </tr>
    </tbody>
  </table>
</div>

The results are encouraging on both counts. First, the unverified EAGLE-style model is competitive across the board, on LIBERO-Goal it actually outperforms the plain baseline and approaches the temporal-ensembling reference, while on LIBERO-Long it stays within the same range. Second, and perhaps more tellingly, there is essentially no gap between the EAGLE-style model and the target model alone. This supports the intuition that small action errors are not catastrophic in closed-loop control: the robot receives a new observation at every step, and the policy can correct minor deviations before they compound.

Also worth noting is that the jointly trained model, optimised with multi-token prediction loss, outperforms the plain autoregressive baseline even when evaluated without the draft heads. This is consistent with DeepSeek-V3's finding that predicting multiple future tokens encourages the model to build more forward-looking representations, which appears to transfer to downstream task performance. Whether the same effect carries over to diffusion-based action heads is an open question.

# Conclusion and Discussion

We started with a question that sounds like a modeling problem — can an autoregressive VLM control a robot in real time? — but it turned out to be a systems problem. The actions-as-text representation was never the bottleneck. The bottleneck was how inference was scheduled and executed. Once we treated the model output as an action stream rather than a completed text sequence, and added a speculative draft model to reduce per-token cost, latency dropped from over 3.5 seconds to around 100 ms without changing the policy or retraining from scratch.

This reframing has a practical consequence worth emphasising. VLM inference is a heavily optimised field: better kernels, KV-cache management, batching strategies, and speculative decoding are all active areas of engineering with mature tooling. A robotics policy built on a standard VLM backbone gets to inherit all of that, essentially for free. A custom architecture with a specialised action head does not. As inference stacks continue to improve, an actions-as-text policy should get faster without any additional work on the robotics side.

The other finding we find interesting is how closed-loop control changes the cost of being wrong. In text generation, a single bad token corrupts the context and can derail everything that follows. In robot control, the loop closes at every timestep: the robot observes the world again, the policy reruns, and small errors can be absorbed before they compound. This is why unverified speculative decoding — which would be considered an approximation in language generation — stays competitive on task performance here. It also raises a broader question about where that tolerance breaks down. Faster, more dynamic tasks, longer horizons, or situations where a single bad action is irreversible may require verification after all. Understanding exactly where the boundary lies is an open question.

What comes next is mostly engineering. The remaining gap to 100 ms is small, and adding a proper inference engine — with optimised decoding kernels and KV-cache handling — should close it. Beyond latency, the more interesting open questions are whether the multi-token prediction benefit generalises beyond LIBERO, and whether the same approximate-inference argument holds on a real robot where errors have physical consequences.

# References

1. Balakhnov, O., & Skvortsov, S. (2025). *VLA-0-Smol*.  
   https://robot-learning-collective.github.io/vla-0-smol/

2. DeepSeek-AI. (2024). *DeepSeek-V3 Technical Report*. arXiv:2412.19437.  
   https://arxiv.org/abs/2412.19437

3. Li et al. (2025). *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*. arXiv:2503.01840.  
   https://arxiv.org/abs/2503.01840

4. Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.  
   https://arxiv.org/abs/2211.17192

5. Liu et al. (2023). *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. CoRL 2023.  
   https://arxiv.org/abs/2306.03310

6. NVIDIA. (2024). *H100 Tensor Core GPU Datasheet*.  
   https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306
