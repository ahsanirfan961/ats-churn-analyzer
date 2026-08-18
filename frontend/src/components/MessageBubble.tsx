import type { Message } from '../api/chatClient'
import { ToolCallCard, VerificationBadge } from './ToolCallCard'

export default function MessageBubble({ message }: { message: Message }) {
  if (message.role === 'tool') return null

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-accent/15 px-4 py-2 whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.role === 'correction') {
    return (
      <div className="rounded-md border border-edge bg-panel/40 px-3 py-2 text-xs text-muted">
        <span className="text-amber-300">self-correction prompt</span> — the agent was asked to
        rewrite its previous answer.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {(message.tool_calls ?? []).map((call, index) => (
        <ToolCallCard key={index} call={call} />
      ))}
      <VerificationBadge verification={message.verification ?? null} />
      {message.content && <div className="whitespace-pre-wrap">{message.content}</div>}
    </div>
  )
}
