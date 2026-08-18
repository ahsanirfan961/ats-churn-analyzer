import { useCallback, useEffect, useState } from 'react'
import { createThread, listThreads, type Thread } from './api/chatClient'
import ChatWindow from './components/ChatWindow'
import Composer from './components/Composer'
import Sidebar from './components/Sidebar'
import { useChatStream } from './hooks/useChatStream'

export default function App() {
  const [threads, setThreads] = useState<Thread[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const { messages, pending, isStreaming, send } = useChatStream(activeId)

  useEffect(() => {
    listThreads().then((loaded) => {
      setThreads(loaded)
      setActiveId((current) => current ?? loaded[0]?.thread_id ?? null)
    })
  }, [])

  const startThread = useCallback(async () => {
    const thread = await createThread()
    setThreads((current) => [thread, ...current])
    setActiveId(thread.thread_id)
    return thread.thread_id
  }, [])

  const handleSend = useCallback(
    async (text: string) => {
      if (!activeId) {
        await startThread()
        return
      }
      await send(text)
      setThreads(await listThreads())
    },
    [activeId, send, startThread],
  )

  return (
    <div className="flex h-full">
      <Sidebar
        threads={threads}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={startThread}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        {activeId ? (
          <>
            <ChatWindow
              messages={messages}
              pending={pending}
              isStreaming={isStreaming}
              onExample={handleSend}
            />
            <Composer disabled={isStreaming} onSend={handleSend} />
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <button
              onClick={startThread}
              className="rounded-xl bg-accent px-5 py-3 font-medium text-ink"
            >
              Start a chat
            </button>
          </div>
        )}
      </main>
    </div>
  )
}
