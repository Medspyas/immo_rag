import { Navigate, Route, Routes } from 'react-router-dom'

import { Layout } from '@/components/layout'
import { PageChat } from '@/pages/chat'
import { PageDocuments } from '@/pages/documents'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<PageChat />} />
        <Route path="documents" element={<PageDocuments />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
