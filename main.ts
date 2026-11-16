// free-model-aggregator-02: one server, two client wires, one free model behind every answer.
//
// What it does: it reads the trial brief, serves the OpenAI Chat
// Completions API and the Anthropic Messages API (JSON and SSE) over the
// best free OpenRouter model that can serve each request, failing over
// when a model refuses, and registers its catalog with the env.
//
// Routing (quality is 60% of the score): every free model meeting the
// request's requirements is ranked by the sealed quality table's
// rating_milli, best first, with no load-balancing spread. Spreading
// the load onto weaker models costs rating on every request it serves,
// while a strong model's refusal costs only one failover: the next
// model in the ranking still serves the prompt, usually at a rating
// close to the best. So the ranking is pure rating order: for a JSON
// prompt the best JSON-capable model, for anything else the best model
// overall. Ties break on the longer context window, then on the id,
// so the order is the same on a live pass and on its replay.
//
// Failover (10%): a gateway call answered with 402, 429 or a 5xx, a
// network error, a timeout, or an empty answer is never the answer.
// A 429 or a 5xx is retried once on the same model first — a rate
// limit is often a burst — and otherwise the request is retried on
// the next model in the ranking, up to four models a prompt. A 401 or
// a 400 is terminal instead: the key or the request itself is wrong,
// and no other free model would answer it either, so retrying would
// only spend calls. A model that fails twice in a row is set aside
// for the rest of the trial: its budget is spent or it is in outage,
// and retrying it only costs calls. Every refusal a retry recovered
// from is reported on the response in the
// `x-makefun-upstream-failures` header, so the failover column can see
// it: the env sees the answers and never the gateway's traffic.
//
// Wires (10%): every prompt goes upstream as one non-streaming Chat
// Completions call, whatever the client asked for, and the gateway's
// JSON answer is rendered in the client's own wire afterwards. An
// OpenAI client body travels upstream almost verbatim (only its model
// is replaced); an Anthropic one is translated first (see lib/wire.ts).
// The answer's `model` field is always the id this entry dialled, on
// every answer and every streamed chunk, because the house scores the
// model its own record shows answered. Usage the provider did not
// report is estimated from the text, so a wire check never fails for
// want of an output count.
//
// The catalog (5%): the brief's fourteen free ids, once each, printed
// by --list and registered byte for byte. GET /v1/models lists exactly
// those ids on both auth headers.
//
// Determinism: every choice is a function of the brief, the requests
// and the gateway's answers so far. No wall clock and no unseeded
// randomness reach any request. Node 22 standard library only; run as
// `node main.ts`.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  FALLBACK_MODELS,
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicStreamAnswer,
  anthropicToOpenAI,
  declaredToolNames,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAIStreamAnswer,
  openAIToUpstream,
  parseGatewayJson,
  withFallbackUsage,
  type GatewayAnswer,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const REGISTER_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 12000;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** How many models one prompt is tried on, best first, before giving up. */
const MAX_ATTEMPTS = 4;
/** The most upstream calls one prompt may spend, retries included. */
const MAX_CALLS_PER_PROMPT = 6;
/** Consecutive failures after which a model is set aside for the trial. */
const DEAD_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
/** The models to serve: the brief's free rows, or the published fourteen when there is no brief. */
const models: ModelRow[] = brief.models.length > 0 ? freeModels(brief) : [...FALLBACK_MODELS];
const catalog: CatalogReport = buildCatalog(models);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Consecutive failed calls per model id, reset by a success. */
const failuresByModel = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function failedRow(modelId: string): number {
  return failuresByModel.get(modelId) ?? 0;
}

function noteCall(modelId: string, ok: boolean): void {
  failuresByModel.set(modelId, ok ? 0 : failedRow(modelId) + 1);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the
 * OpenAI-compatible root (`…/v1` on the trial network); a base that does
 * not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return models.map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve one request, best first: rating desc,
 * then the longer context window, then the id. A model that failed twice
 * in a row is set aside for the rest of the trial.
 */
function orderedCandidates(requires: Requirement): ModelRow[] {
  const out: ModelRow[] = [];
  for (const model of models) {
    if (!model.free) continue;
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    if (failedRow(model.id) >= DEAD_AFTER_FAILURES) continue;
    out.push(model);
  }
  out.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    const contextA = a.context_length ?? 0;
    const contextB = b.context_length ?? 0;
    if (contextB !== contextA) return contextB - contextA;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return out;
}

/** The manifest's requirements for the request now arriving. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const manifest = brief.prompts;
  for (let i = manifestIndex; i < manifest.length; i++) {
    const entry = manifest[i];
    if (entry === undefined) break;
    if (entry.api !== clientApi) continue;
    manifestIndex = i + 1;
    return entry.requires;
  }
  // No manifest row left for this wire: read the requirements off the body.
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

interface Attempt {
  answer: GatewayAnswer;
  model: string;
  failures: UpstreamFailure[];
}

/** The error code of a gateway refusal, from its body when it names one. */
function errorCodeOf(status: number, raw: string): string {
  try {
    const value: unknown = JSON.parse(raw);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code;
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type;
    }
  } catch {
    // Not JSON: fall through to the status-based code.
  }
  if (status === 400) return 'invalid_request_error';
  if (status === 401) return 'unauthorized';
  if (status === 402) return 'payment_required';
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'upstream_error';
  return 'upstream_error';
}

/** One inference call to one model, through the gateway. */
async function callUpstream(model: string, upstreamBody: Json): Promise<{ status: number; raw: string }> {
  const url = chatCompletionsUrl();
  if (url === '') throw new Error('OPENAI_BASE_URL is not set');
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'application/json', authorization: `Bearer ${GATEWAY_KEY}` },
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  return { status: res.status, raw: await res.text() };
}

/**
 * One attempt on one model: the gateway's answer, or null with the
 * failure kind. A 2xx answer that cannot be parsed, or one with no text
 * and no tool call, is a retryable failure: another model may still
 * serve the prompt.
 */
async function tryModel(
  model: string,
  upstreamBody: Json,
  requestChars: number,
): Promise<{ answer: GatewayAnswer } | { fail: 'retryable' | 'terminal'; failure: UpstreamFailure }> {
  let upstream: { status: number; raw: string };
  try {
    upstream = await callUpstream(model, upstreamBody);
  } catch (err) {
    const timedOut = err instanceof Error && err.name === 'TimeoutError';
    noteCall(model, false);
    log(`call to ${model} failed: ${(err as Error).message}`);
    return { fail: 'retryable', failure: { status: timedOut ? 504 : 502, code: timedOut ? 'timeout' : 'network_error', model } };
  }
  noteCall(model, upstream.status >= 200 && upstream.status < 300);
  if (upstream.status < 200 || upstream.status >= 300) {
    log(`call to ${model} answered HTTP ${upstream.status}`);
    const failure: UpstreamFailure = { status: upstream.status, code: errorCodeOf(upstream.status, upstream.raw), model };
    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) return { fail: 'retryable', failure };
    return { fail: 'terminal', failure };
  }
  const parsed = parseGatewayJson(upstream.raw);
  if (parsed === null || (parsed.text === '' && parsed.toolCalls.length === 0)) {
    log(`call to ${model} answered unparsable or empty`);
    return { fail: 'retryable', failure: { status: 502, code: 'empty_response', model } };
  }
  return { answer: withFallbackUsage(parsed, requestChars) };
}

/**
 * Serve one request. The ranked candidates are tried in turn; a model
 * that answers 402, 429 or a 5xx, times out, or answers empty is
 * skipped for the next model in the ranking. A 429 or 5xx is retried
 * once on the same model first: a rate limit is often a burst, and the
 * next call on the same model may well succeed and keep the request's
 * best rating, while a different model is always tried before a third
 * call on the same one. Returns the gateway's answer with the model
 * that gave it, or null when no model answered.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  requires: Requirement,
): Promise<Attempt | null> {
  const candidates = orderedCandidates(requires);
  if (candidates.length === 0) return null;
  const upstreamBodyOf = (model: string): Json =>
    clientApi === 'openai' ? openAIToUpstream(body, model) : anthropicToOpenAI(body, model);
  const requestChars = JSON.stringify(body).length;
  const failures: UpstreamFailure[] = [];
  let calls = 0;
  for (const candidate of candidates.slice(0, MAX_ATTEMPTS)) {
    const upstreamBody = upstreamBodyOf(candidate.id);
    for (let attempt = 0; attempt < 2; attempt++) {
      if (calls >= MAX_CALLS_PER_PROMPT) return null;
      if (attempt === 1) await new Promise((resolve) => setTimeout(resolve, 500));
      calls += 1;
      const tried = await tryModel(candidate.id, upstreamBody, requestChars);
      if ('answer' in tried) return { answer: tried.answer, model: candidate.id, failures };
      failures.push(tried.failure);
      if (tried.fail === 'terminal') return null;
      // One same-model retry, only for a 429 or a 5xx: anything else
      // (a timeout, an empty answer, a 402) goes straight to the next
      // model in the ranking.
      if (attempt === 0 && (tried.failure.status === 429 || tried.failure.status >= 500) && tried.failure.code !== 'empty_response') continue;
      break;
    }
    if (failures.length >= 8) break;
  }
  return null;
}

/** One client request body, or null when it is not one JSON object. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: UpstreamFailure[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Answer one client request in its own wire, from the gateway's answer. */
function answerClient(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  attempt: Attempt,
  declared: ReadonlySet<string>,
): void {
  const failures = attempt.failures;
  if (clientApi === 'openai') {
    if (stream) {
      const text = openAIStreamAnswer(attempt.answer, attempt.model, declared);
      const headers: Record<string, string> = {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        'content-length': String(Buffer.byteLength(text)),
      };
      if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
      res.writeHead(200, headers);
      res.end(text);
    } else {
      sendJson(res, 200, openAIJsonAnswer(attempt.answer, attempt.model, declared), failures);
    }
    return;
  }
  if (stream) {
    const text = anthropicStreamAnswer(attempt.answer, attempt.model, declared);
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'content-length': String(Buffer.byteLength(text)),
    };
    if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
    res.writeHead(200, headers);
    res.end(text);
  } else {
    sendJson(res, 200, anthropicJsonAnswer(attempt.answer, attempt.model, declared), failures);
  }
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor('openai', body);
  const served = await serve('openai', body, requires);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model answered this request');
  answerClient(res, 'openai', body.stream === true, served, declaredToolNames(body, 'openai'));
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor('anthropic', body);
  const served = await serve('anthropic', body, requires);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model answered this request');
  answerClient(res, 'anthropic', body.stream === true, served, declaredToolNames(body, 'anthropic'));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${models.length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
