# free-model-aggregator-2

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/cocotvoros/free-model-aggregator-2/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/cocotvoros/free-model-aggregator-2/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE) over the best free OpenRouter model that can serve each request. Every completion comes from a real free model dialled through the grader's gateway — never an answer of its own.

## Run it

```sh
node main.ts --list   # print the catalog of free models and exit
node main.ts          # serve (needs MAKEFUN_PROMPT_FILE + MAKEFUN_ENV_URL)
```

Run it with Node 22 and nothing to install.

How it picks a model: every free model meeting the request's requirements is ranked by the sealed quality table's rating, best first, with no load-balancing spread: on the practice trials a spread cost far more weighted rating than the failovers it saved. A JSON-mode prompt goes to the best JSON-capable model; any other prompt goes to the best model overall. A model that answers 402, 429 or a 5xx, times out, or answers empty is never the answer: a 429 or 5xx is retried once on the same model first, then the request moves to the next model in the ranking, up to four models a prompt, and a model that fails twice in a row is set aside for the rest of the trial. Refusals a retry recovered from are reported on the response in the `x-makefun-upstream-failures` header.

Every translation preserves the provider echo exactly: system text, user text, tool names, `max_tokens`, stop sequences, images and JSON mode travel verbatim, and the usage the client sees is the usage the provider reported — estimated from the text only when the provider reported none. The `model` field always names the model really dialled.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
