import { useState } from 'react'
import type { ToolCall, Verification } from '../api/chatClient'

function summarize(args: unknown): string {
  const text = JSON.stringify(args ?? {})
  return text.length > 70 ? `${text.slice(0, 70)}...` : text
}

function pretty(output: string | undefined): string {
  if (!output) return 'running...'
  try {
    return JSON.stringify(JSON.parse(output.replace(/'/g, '"')), null, 2)
  } catch {
    return output
  }
}

export function ToolCallCard({ call }: { call: ToolCall }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-md border border-edge bg-panel/60 text-xs">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-panel"
      >
        <span className="text-muted">{open ? '▾' : '▸'}</span>
        <span className="font-mono text-accent">{call.name}</span>
        <span className="truncate text-muted">{summarize(call.args)}</span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-edge px-3 py-2">
          <div>
            <div className="mb-1 text-muted">arguments</div>
            <pre className="overflow-x-auto font-mono text-[11px]">
              {JSON.stringify(call.args ?? {}, null, 2)}
            </pre>
          </div>
          <div>
            <div className="mb-1 text-muted">result</div>
            <pre className="max-h-64 overflow-auto font-mono text-[11px]">
              {pretty(call.output)}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

export function VerificationBadge({
  verification,
  rewritten,
}: {
  verification: Verification | null
  rewritten?: boolean
}) {
  if (!verification) return null

  if (verification.passed) {
    return (
      <div className="text-xs text-emerald-400">
        Self-check passed{rewritten ? ' after a rewrite' : ''}
        {verification.parse_error ? ' (judge output unreadable, answer not verified)' : ''}
      </div>
    )
  }

  return (
    <div className="space-y-1 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
      <div>
        Self-check flagged {verification.problems.length} claim
        {verification.problems.length === 1 ? '' : 's'}
        {verification.retrying ? ' — rewriting the answer' : ''}
      </div>
      {verification.problems.map((problem, index) => (
        <div key={index} className="text-amber-200/80">
          <span className="font-mono">{problem.verdict}</span>: "{problem.text}"
          {problem.reason ? ` — ${problem.reason}` : ''}
        </div>
      ))}
    </div>
  )
}
