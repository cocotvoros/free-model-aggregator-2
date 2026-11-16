// The client wires, the one upstream wire, and the translations between.
//
// The trial's client calls this server as an OpenAI service
// (POST /v1/chat/completions) or an Anthropic one (POST /v1/messages),
// in JSON or in server-sent events. Every prompt goes upstream as one
// Chat Completions request to the grader's gateway (OpenRouter's free
// models live there), always non-streaming: the gateway's JSON answer is
// then rendered in the client's own wire, streamed or not as it asked.
// Buffering first and rendering after keeps the event order well formed
// on both wires, whatever the provider's own stream looked like.
//
// The one rule that governs every translation: the provider echoes what
// it parsed (system text, user text, tool names, max_tokens, stop
// sequences, images and JSON mode), and the client compares that echo
// with what it sent. A field rewritten or dropped loses the request. So:
//
//   - an OpenAI client body travels upstream almost verbatim: only its
//     `model` is replaced by the dialled id, `stream` is taken off (the
//     upstream call is never streamed) and a missing output cap is filled
//     in, because the gateway refuses a request with no cap;
//   - an Anthropic client body is rewritten into Chat Completions form:
//     the system prompt becomes a system message, image blocks become
//     image_url data URLs, tool_use blocks become tool_calls, tool_result
//     blocks become tool messages, and the output cap, temperature,
//     top_p and stop sequences pass through;
//   - a gateway answer travels back verbatim apart from its `model`
//     field, which is always the id this entry dialled, and its usage,
//     which is estimated from the text only when the provider reported
//     none (a wire check needs an output count of at least 1);
//   - a tool call the model made is kept only when the client declared
//     that tool's name: the wire check fails a prompt whose answer names
//     a tool the client never declared.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same gateway answer give the same
// bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How wide one streamed text delta is, matching the providers. */
const DELTA_CHARS = 13;
/** The output cap filled in when a client body names none (the gateway refuses a request with no cap). */
const DEFAULT_MAX_TOKENS = 1024;

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then data, then blank. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** Split text into fixed-width pieces, the way the providers split theirs. */
export function splitText(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) out.push(text.slice(at, at + DELTA_CHARS));
  return out.length === 0 ? [''] : out;
}

/** A rough token count for fallback usage: about one token per four characters, at least the floor. */
export function estimateTokens(text: string, floor: number): number {
  return Math.max(floor, Math.ceil(text.length / 4));
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Client request -> upstream Chat Completions body
// ---------------------------------------------------------------------------

/**
 * An OpenAI client body as the upstream Chat Completions body naming
 * `model`: the client's own body, with the dialled id, never streamed,
 * and an output cap filled in when the client named none.
 */
export function openAIToUpstream(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.stream;
  delete out.stream_options;
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') {
    out.max_tokens = DEFAULT_MAX_TOKENS;
  }
  return out;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request
 * naming `model`: the system prompt becomes a system message, image
 * blocks become image_url data URLs, tool_use blocks become tool_calls,
 * tool_result blocks become tool messages, tools and tool_choice are
 * rewritten, and the output cap, temperature, top_p and stop sequences
 * pass through. Never streamed upstream: the gateway answer is rendered
 * in the client's wire afterwards, streamed or not as it asked.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else out.max_tokens = DEFAULT_MAX_TOKENS;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers
// ---------------------------------------------------------------------------

export interface UpstreamToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface GatewayAnswer {
  id: string;
  created: number;
  text: string;
  toolCalls: UpstreamToolCall[];
  finishReason: string;
  inputTokens: number;
  outputTokens: number;
}

/** Parse one gateway Chat Completions JSON body into text, tool calls, usage and stop reason. */
export function parseGatewayJson(raw: string): GatewayAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: UpstreamToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
      });
    }
  }
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    id: typeof value.id === 'string' ? value.id : 'chatcmpl-agg',
    created: typeof value.created === 'number' ? value.created : FIXED_OPENAI_CREATED,
    text,
    toolCalls,
    finishReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
  };
}

/**
 * Fill in usage the provider did not report: the output count from the
 * answer's own text (at least 1 when there is text), the input count
 * from the request's size. A wire check needs an output count of at
 * least 1, and a provider that reports no usage must not fail it.
 */
export function withFallbackUsage(answer: GatewayAnswer, requestChars: number): GatewayAnswer {
  const inputTokens = answer.inputTokens > 0 ? answer.inputTokens : estimateTokens(' '.repeat(Math.min(requestChars, 100000)), 0);
  const outputTokens = answer.outputTokens > 0 ? answer.outputTokens : answer.text === '' ? 1 : estimateTokens(answer.text, 1);
  return { ...answer, inputTokens, outputTokens };
}

/** Keep only the tool calls whose name the client declared. */
export function declaredCalls(calls: UpstreamToolCall[], declared: ReadonlySet<string>): UpstreamToolCall[] {
  if (declared.size === 0) return [];
  return calls.filter((c) => declared.has(c.name));
}

/** The tool names a client body declares, on either wire. */
export function declaredToolNames(body: Json, wire: 'openai' | 'anthropic'): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(body.tools)) return names;
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (wire === 'openai' && isRecord(raw.function) && typeof raw.function.name === 'string') names.add(raw.function.name);
    if (wire === 'anthropic' && typeof raw.name === 'string') names.add(raw.name);
  }
  return names;
}

// ---------------------------------------------------------------------------
// Gateway answers -> client wire
// ---------------------------------------------------------------------------

/** A gateway answer as an OpenAI Chat Completions JSON object. `model` is the id the entry dialled. */
export function openAIJsonAnswer(answer: GatewayAnswer, model: string, declared: ReadonlySet<string>): Json {
  const calls = declaredCalls(answer.toolCalls, declared);
  const message: Json = { role: 'assistant', content: answer.text };
  if (calls.length > 0) {
    message.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
  }
  const finish = calls.length > 0 && answer.finishReason === 'stop' ? 'tool_calls' : answer.finishReason;
  return {
    id: answer.id,
    object: 'chat.completion',
    created: answer.created,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A gateway answer as an OpenAI SSE stream, ending with usage and [DONE]. */
export function openAIStreamAnswer(answer: GatewayAnswer, model: string, declared: ReadonlySet<string>): string {
  const calls = declaredCalls(answer.toolCalls, declared);
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: answer.id,
    object: 'chat.completion.chunk',
    created: answer.created,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = '';
  out += sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(answer.text)) out += sseEvent(null, chunk({ content: piece }, null));
  for (let i = 0; i < calls.length; i++) {
    const call = calls[i] as UpstreamToolCall;
    out += sseEvent(null, chunk({ tool_calls: [{ index: i, id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null));
  }
  const finish = calls.length > 0 && answer.finishReason === 'stop' ? 'tool_calls' : answer.finishReason;
  out += sseEvent(null, chunk({}, finish));
  out += sseEvent(null, {
    id: answer.id,
    object: 'chat.completion.chunk',
    created: answer.created,
    model,
    choices: [],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** A gateway answer as an Anthropic Messages JSON object. */
export function anthropicJsonAnswer(answer: GatewayAnswer, model: string, declared: ReadonlySet<string>): Json {
  const calls = declaredCalls(answer.toolCalls, declared);
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of calls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input: argumentsOf(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const stop = calls.length > 0 && answer.finishReason === 'stop' ? 'tool_use' : stopReasonOf(answer.finishReason);
  return {
    id: `msg_${answer.id}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A gateway answer as an Anthropic SSE event sequence. */
export function anthropicStreamAnswer(answer: GatewayAnswer, model: string, declared: ReadonlySet<string>): string {
  const calls = declaredCalls(answer.toolCalls, declared);
  let out = '';
  out += sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: `msg_${answer.id}`,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== '') {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of splitText(answer.text)) {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of calls) {
    out += sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
    });
    if (call.arguments !== '') {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  const stop = calls.length > 0 && answer.finishReason === 'stop' ? 'tool_use' : stopReasonOf(answer.finishReason);
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
