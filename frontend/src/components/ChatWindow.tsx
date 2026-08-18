import { useEffect, useRef } from 'react'
import type { Message } from '../api/chatClient'
import type { PendingAnswer } from '../hooks/useChatStream'
import MarkdownContent from './MarkdownContent'
import MessageBubble from './MessageBubble'
import { ToolCallCard, VerificationBadge } from './ToolCallCard'

const EXAMPLES = [
  'Which contract type has the highest churn rate?',
  'Why is customer 4424-TKOPW at risk?',
  'What happens to their risk if they move to a two-year contract?',
  'Top 10 highest-paying customers without tech support',
]

export default function ChatWindow({
  messages,
  pending,
  isStreaming,
  onExample,
}: {
  messages: Message[]
  pending: PendingAnswer | null
  isStreaming: boolean
  onExample: (text: string) => void
}) {
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, pending])

  const visible = messages.filter((m) => m.role !== 'tool')

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-8">
        {visible.length === 0 && !pending && (
          <div className="mt-16 space-y-4">
            <h1 className="text-xl">Ask about churn in the customer base.</h1>
            <div className="flex flex-col items-start gap-2">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  onClick={() => onExample(example)}
                  className="rounded-full border border-edge px-3 py-1.5 text-sm text-muted hover:border-accent hover:text-accent"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {visible.map((message, index) => (
          <MessageBubble key={index} message={message} />
        ))}

        {pending && (
          <div className="space-y-2">
            {pending.toolCalls.map((call, index) => (
              <ToolCallCard key={index} call={call} />
            ))}
            <VerificationBadge
              verification={pending.verification}
              rewritten={pending.rewritten}
            />
            <div>
              {pending.content && <MarkdownContent content={pending.content} />}
              {pending.phase === 'verifying' ? (
                <div className="mt-2 flex items-center gap-2 text-xs text-muted">
                  <span
                    className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-edge border-t-accent"
                    aria-hidden
                  />
                  Verifying answer against tool results…
                </div>
              ) : (
                isStreaming && <span className="animate-pulse text-muted">▍</span>
              )}
            </div>
          </div>
        )}

        <div ref={bottom} />
      </div>
    </div>
  )
}
