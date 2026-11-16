// The trial brief: the free model list and the six prompts, plus the catalog.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) is prose with two JSON
// documents in it. This file finds each document by its `schema` member
// under this competition's prefix (`free-model-aggregator-02/`):
//
//   - the free model list (`free-model-aggregator-02/models/v1`): one row
//     per model, keyed by the id the gateway routes
//     (`openrouter/<vendor>/<model>:free`), with the house's published
//     rating in milli-points, what the model can do (tools, vision, JSON
//     mode, context length) and its free flag. Only a free model may serve
//     a prompt.
//   - the six prompts (`free-model-aggregator-02/prompts/v1`): the prompts
//     the env's client sends, in order, each with its id, its api
//     (`openai` or `anthropic`), whether it streams, its weight (1 or 2)
//     and the requirements a model must meet to serve it.
//
// Everything here is deterministic: no clock and no randomness, so
// `node main.ts --list` and the register body build the same document,
// byte for byte, on a live pass and on its replay.
//
// Outside a trial there is no brief to read. Then the catalog falls back
// to the fourteen free ids of the competition's published model list, so
// `--list` still prints the catalog the server can serve, and a real
// client can point at the server and use it. Inside a trial the brief
// always wins: if the sealed list ever differed, the brief's rows are the
// ones registered and served.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this competition's brief writes names this prefix. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The env var an entrant's OpenRouter key is pushed under (kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this server reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/**
 * The competition's published fourteen free ids, in the brief's order,
 * with the context lengths the brief publishes. Used only when no brief
 * is present (outside a trial); a trial always uses its own brief.
 */
export const FALLBACK_MODELS: ModelRow[] = [
  { id: 'openrouter/baidu/ernie-5.5-flash-preview:free', provider: 'openrouter', display_name: 'Baidu: ERNIE-5.5-Flash-Preview (free)', rating_milli: 1150, context_length: 1024000, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/cohere/north-mini-code:free', provider: 'openrouter', display_name: 'Cohere: North Mini Code (free)', rating_milli: 1200, context_length: 256000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/dots-studio/dots-3-note-preview:free', provider: 'openrouter', display_name: 'Dots Studio: Dots3-Note Preview (free)', rating_milli: 1150, context_length: 512000, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/google/gemma-4-26b-a4b-it:free', provider: 'openrouter', display_name: 'Google: Gemma 4 26B A4B (free)', rating_milli: 1150, context_length: 262144, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/google/gemma-4-31b-it:free', provider: 'openrouter', display_name: 'Google: Gemma 4 31B (free)', rating_milli: 1210, context_length: 262144, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/inclusionai/ling-3.2-flash:free', provider: 'openrouter', display_name: 'inclusionAI: Ling 3.2 Flash (free)', rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/liquid/lfm-3-1.7b:free', provider: 'openrouter', display_name: 'LiquidAI: LFM3-1.7B (free)', rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', provider: 'openrouter', display_name: 'NVIDIA: Nemotron 3 Nano Omni (free)', rating_milli: 1150, context_length: 256000, tools: true, vision: true, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free', provider: 'openrouter', display_name: 'NVIDIA: Nemotron 3 Ultra (free)', rating_milli: 1285, context_length: 1000000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3.5-lightning:free', provider: 'openrouter', display_name: 'NVIDIA: Nemotron 3.5 Lightning (free)', rating_milli: 1221, context_length: 1000000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3.5-super-vl:free', provider: 'openrouter', display_name: 'NVIDIA: Nemotron 3.5 Super VL (free)', rating_milli: 1303, context_length: 1000000, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/particle-ai/particle-1:free', provider: 'openrouter', display_name: 'Particle AI: Particle-1 (free)', rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/recursive/flint-v1.0:free', provider: 'openrouter', display_name: 'Recursive: Flint (free)', rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/thinkingmachines/inkling-2-small:free', provider: 'openrouter', display_name: 'Thinking Machines: Inkling 2 Small (free)', rating_milli: 1475, context_length: 1048576, tools: true, vision: true, json: false, free: true },
];

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this competition's prefix that carries one of
 * `keys` as an array: the document and the key it was found under.
 */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the
 * provider that names it: a row that gives the bare OpenRouter id
 * (`vendor/model:free`) and names `openrouter` is routed as the gateway
 * spells it (`openrouter/vendor/model:free`). A row is free unless it
 * says `free: false`; a paid row is never routed.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this competition's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's
 * trial-local address: the gateway's origin differs between a live pass
 * and its replay, and the catalog is part of the state export a replay
 * must reproduce byte for byte.
 */
export function buildCatalog(models: ModelRow[]): CatalogReport {
  const seen = new Set<string>();
  const list: CatalogModel[] = [];
  for (const model of models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    list.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (list.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models: list,
      },
    ],
  };
}
