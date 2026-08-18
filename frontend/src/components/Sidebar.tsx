import type { Thread } from '../api/chatClient'

export default function Sidebar({
  threads,
  activeId,
  onSelect,
  onNew,
}: {
  threads: Thread[]
  activeId: string | null
  onSelect: (threadId: string) => void
  onNew: () => void
}) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-edge bg-panel">
      <div className="px-4 py-4">
        <div className="mb-3 text-sm font-medium">Churn Analyst</div>
        <button
          onClick={onNew}
          className="w-full rounded-lg border border-edge px-3 py-2 text-sm hover:border-accent hover:text-accent"
        >
          New chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-2 pb-4">
        {threads.map((thread) => (
          <button
            key={thread.thread_id}
            onClick={() => onSelect(thread.thread_id)}
            className={`mb-1 w-full truncate rounded-lg px-3 py-2 text-left text-sm ${
              thread.thread_id === activeId ? 'bg-edge text-white' : 'text-muted hover:bg-edge/50'
            }`}
          >
            {thread.title}
          </button>
        ))}
      </div>
    </aside>
  )
}
