import { useCallback, useEffect, useState } from 'react'
import {
  getMessages,
  streamTurn,
  type Message,
  type ToolCall,
  type Verification,
} from '../api/chatClient'

export type PendingAnswer = {
  content: string
  toolCalls: ToolCall[]
  verification: Verification | null
  rewritten: boolean
  phase: 'streaming' | 'verifying' | null
}

const emptyPending: PendingAnswer = {
  content: '',
  toolCalls: [],
  verification: null,
  rewritten: false,
  phase: 'streaming',
}

export function useChatStream(threadId: string | null) {
  const [messages, setMessages] = useState<Message[]>([])
  const [pending, setPending] = useState<PendingAnswer | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  useEffect(() => {
    setPending(null)
    if (!threadId) {
      setMessages([])
      return
    }
    getMessages(threadId).then(setMessages)
  }, [threadId])

  const send = useCallback(
    async (text: string) => {
      if (!threadId || isStreaming) return
      setMessages((current) => [...current, { role: 'user', content: text }])
      setIsStreaming(true)

      let answer: PendingAnswer = { ...emptyPending }
      let completed = false
      setPending(answer)

      for await (const event of streamTurn(threadId, text)) {
        if (event.type === 'token') {
          answer = { ...answer, content: answer.content + event.text, phase: 'streaming' }
        } else if (event.type === 'status') {
          answer = { ...answer, phase: event.phase }
        } else if (event.type === 'tool_call') {
          answer = {
            ...answer,
            toolCalls: [...answer.toolCalls, { name: event.name, args: event.args }],
          }
        } else if (event.type === 'tool_result') {
          const toolCalls = [...answer.toolCalls]
          for (let i = toolCalls.length - 1; i >= 0; i -= 1) {
            if (toolCalls[i].name === event.name && toolCalls[i].output === undefined) {
              toolCalls[i] = { ...toolCalls[i], output: event.output }
              break
            }
          }
          answer = { ...answer, toolCalls }
        } else if (event.type === 'verification') {
          answer = { ...answer, verification: event, phase: null }
        } else if (event.type === 'retry') {
          answer = { ...answer, content: '', rewritten: true, phase: 'streaming' }
        } else if (event.type === 'done') {
          setMessages((current) => [
            ...current,
            {
              role: 'assistant',
              content: event.answer,
              tool_calls:
                answer.toolCalls.length > 0 ? [...answer.toolCalls] : undefined,
              verification: answer.verification,
            },
          ])
          completed = true
          setPending(null)
          continue
        } else if (event.type === 'error') {
          setMessages((current) => [
            ...current,
            {
              role: 'assistant',
              content: event.message,
              tool_calls:
                answer.toolCalls.length > 0 ? [...answer.toolCalls] : undefined,
            },
          ])
          setPending(null)
          continue
        }
        setPending({ ...answer })
      }

      if (completed) {
        const loaded = await getMessages(threadId)
        setMessages(loaded)
      }
      setPending(null)
      setIsStreaming(false)
    },
    [threadId, isStreaming],
  )

  return { messages, pending, isStreaming, send }
}
