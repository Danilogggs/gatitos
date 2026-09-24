import { Link, NavLink, Outlet } from 'react-router-dom'

export function Layout() {
  return <div className="site-shell"><header className="site-header"><div className="container header-inner">
    <Link className="brand" to="/"><span className="brand-icon">✦</span> catcare<span>ai</span><small>ENCONTROS QUE TRANSFORMAM</small></Link>
    <nav aria-label="Navegação principal"><NavLink to="/">Encontrar um gato</NavLink><NavLink to="/cadastrar">Cadastrar gato</NavLink><NavLink to="/admin">Área administrativa</NavLink></nav>
  </div></header><main><Outlet /></main><footer className="site-footer"><div className="container footer-inner"><div><strong>catcare<span>ai</span></strong><p>Tecnologia e cuidado para aproximar gatos de novos lares.</p></div><span>Feito com cuidado, revisado por pessoas. ♡</span></div></footer></div>
}

