import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { AdminDashboard, AdminLogin, AdminReview } from './pages/Admin'
import { Details } from './pages/Details'
import { Home } from './pages/Home'
import { Register, Sent } from './pages/Register'

export default function App() { return <Routes><Route element={<Layout/>}><Route index element={<Home/>}/><Route path="gatos/:id" element={<Details/>}/><Route path="cadastrar" element={<Register/>}/><Route path="enviado/:id" element={<Sent/>}/><Route path="admin" element={<AdminLogin/>}/><Route path="admin/dashboard" element={<AdminDashboard/>}/><Route path="admin/revisar/:id" element={<AdminReview/>}/><Route path="*" element={<Navigate to="/" replace/>}/></Route></Routes> }

