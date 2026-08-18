import { useState } from 'react'
import type { ClaimCheck, ToolCall, Verification } from '../api/chatClient'

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

const CLAIM_STYLES: Record<ClaimCheck['verdict'], { icon: string; className: string }> = {
  grounded: { icon: '✓', className: 'text-emerald-400' },
  mislabeled: { icon: '⚠', className: 'text-amber-300' },
  fabricated: { icon: '✗', className: 'text-red-400' },
}

function ClaimCheckRow({ claim }: { claim: ClaimCheck }) {
  if (!claim?.verdict) return null
  const style = CLAIM_STYLES[claim.verdict] ?? CLAIM_STYLES.grounded
  return (
    <div className={`flex items-start gap-2 text-xs ${style.className}`}>
      <span className="mt-px shrink-0 font-mono">{style.icon}</span>
      <span>
        "{claim.text}"
        {claim.reason ? <span className="text-muted"> — {claim.reason}</span> : null}
      </span>
    </div>
  )
}

export function ClaimCheckList({
  claims,
  waiting,
}: {
  claims: ClaimCheck[]
  waiting: boolean
}) {
  return (
    <div className="mt-2 space-y-1.5 rounded-md border border-edge bg-panel/40 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted">Verifying claims</div>
      {claims.filter(Boolean).map((claim, index) => (
        <ClaimCheckRow key={index} claim={claim} />
      ))}
      {waiting && (
        <div className="flex items-center gap-2 text-xs text-muted">
          <span
            className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-edge border-t-accent"
            aria-hidden
          />
          {claims.length === 0 ? 'Checking answer against tool results…' : 'Checking next claim…'}
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
