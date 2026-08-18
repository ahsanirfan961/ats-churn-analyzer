import { getApiBaseUrl } from '../config'

const BASE_URL = getApiBaseUrl()

export type Thread = {
  thread_id: string
  title: string
  created_at: string
  updated_at: string
}

export type Problem = {
  text: string
  verdict: 'grounded' | 'mislabeled' | 'fabricated'
  reason: string | null
}

export type Verification = {
  passed: boolean
  problems: Problem[]
  parse_error: string | null
  retrying?: boolean
}

export type ToolCall = {
  name: string
  args: unknown
  output?: string
}

export type Message = {
  role: 'user' | 'assistant' | 'tool' | 'correction'
  content: string
  name?: string
  tool_calls?: ToolCall[]
  verification?: Verification | null
}

function attachToolOutput(toolCalls: ToolCall[], name: string, content: string): ToolCall[] {
  const calls = toolCalls.map((call) => ({ ...call }))
  for (let i = calls.length - 1; i >= 0; i -= 1) {
    if (calls[i].name === name && calls[i].output === undefined) {
      calls[i] = { ...calls[i], output: content }
      return calls
    }
  }
  return [...calls, { name, args: {}, output: content }]
}

/** Fold tool-role messages into the final assistant reply for each turn. */
export function normalizeMessages(messages: Message[]): Message[] {
  const result: Message[] = []
  let pendingTools: ToolCall[] = []

  for (const message of messages) {
    if (message.role === 'user' || message.role === 'correction') {
      pendingTools = []
      result.push(message)
      continue
    }

    if (message.role === 'tool') {
      pendingTools = attachToolOutput(pendingTools, message.name ?? '', message.content)
      continue
    }

    if (message.role === 'assistant') {
      const toolCalls = [
        ...pendingTools,
        ...(message.tool_calls ?? []).map((call) => ({ ...call })),
      ]
      pendingTools = []

      if (message.content.trim()) {
        result.push({
          ...message,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        })
      } else if (toolCalls.length > 0) {
        pendingTools = toolCalls
      } else {
        result.push(message)
      }
    }
  }

  if (pendingTools.length > 0) {
    result.push({ role: 'assistant', content: '', tool_calls: pendingTools })
  }

  return result
}

export type ClaimCheck = {
  text: string
  verdict: 'grounded' | 'mislabeled' | 'fabricated'
  reason: string | null
}

export type StreamEvent =
  | { type: 'token'; text: string }
  | { type: 'tool_call'; name: string; args: unknown }
  | { type: 'tool_result'; name: string; output: string }
  | { type: 'status'; phase: 'verifying' }
  | { type: 'claim_check'; claim: ClaimCheck }
  | ({ type: 'verification' } & Verification)
  | { type: 'retry'; reason: string }
  | { type: 'done'; answer: string }
  | { type: 'error'; message: string }

export async function createThread(): Promise<Thread> {
  const response = await fetch(`${BASE_URL}/threads`, { method: 'POST' })
  return response.json()
}

export async function listThreads(): Promise<Thread[]> {
  const response = await fetch(`${BASE_URL}/threads`)
  return response.json()
}

export async function getMessages(threadId: string): Promise<Message[]> {
  const response = await fetch(`${BASE_URL}/threads/${threadId}/messages`)
  const body = await response.json()
  return normalizeMessages(body.messages)
}

export async function* streamTurn(
  threadId: string,
  message: string,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${BASE_URL}/threads/${threadId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  if (!response.ok || !response.body) {
    yield { type: 'error', message: `The server responded with ${response.status}.` }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const parseBuffer = function* () {
    let split = buffer.indexOf('\n\n')
    while (split !== -1) {
      const block = buffer.slice(0, split)
      buffer = buffer.slice(split + 2)
      const dataLine = block.split('\n').find((line) => line.startsWith('data: '))
      if (dataLine) yield JSON.parse(dataLine.slice(6)) as StreamEvent
      split = buffer.indexOf('\n\n')
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    yield* parseBuffer()
  }

  buffer += decoder.decode()
  yield* parseBuffer()

  const trailing = buffer.split('\n').find((line) => line.startsWith('data: '))
  if (trailing) yield JSON.parse(trailing.slice(6)) as StreamEvent
}
