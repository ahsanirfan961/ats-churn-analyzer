import { useState } from 'react'

export default function Composer({
  disabled,
  onSend,
}: {
  disabled: boolean
  onSend: (text: string) => void
}) {
  const [text, setText] = useState('')

  const submit = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
  }

  return (
    <div className="border-t border-edge bg-ink px-6 py-4">
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <textarea
          value={text}
          rows={1}
          disabled={disabled}
          placeholder={disabled ? 'Thinking...' : 'Ask about the customer base'}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          className="max-h-40 flex-1 resize-none rounded-xl border border-edge bg-panel px-4 py-3 outline-none focus:border-accent disabled:opacity-50"
        />
        <button
          onClick={submit}
          disabled={disabled || !text.trim()}
          className="rounded-xl bg-accent px-4 py-3 font-medium text-ink disabled:opacity-30"
        >
          Send
        </button>
      </div>
    </div>
  )
}
