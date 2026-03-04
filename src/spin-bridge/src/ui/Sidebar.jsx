import { MessageSquare, Settings, LayoutDashboard } from "lucide-react";
import { Link, useLocation } from "react-router-dom";

export default function Sidebar() {
  const location = useLocation();

  // Función para determinar si la ruta es la activa
  const isActive = (path) => location.pathname === path;

  // Estilos base para los botones
  const navItemStyles =
    "flex items-center gap-3 px-4 py-3 rounded-2xl transition-all duration-200 group";

  // Estilo cuando NO está seleccionado
  const inactiveStyles =
    "text-slate-500 hover:bg-slate-50 hover:text-brand-primary";

  // Estilo cuando SÍ está seleccionado
  const activeStyles =
    "bg-brand-primary text-white shadow-lg shadow-brand-primary/20 scale-[1.02]";

  return (
    <div className="w-64 h-screen bg-white text-brand-dark p-6 flex flex-col fixed border-r border-slate-100 font-montserrat shadow-sm">
      {/* Logo Section */}
      <div className="flex items-center gap-2 mb-12 px-2">
        <div className="w-8 h-8 bg-brand-primary rounded-lg flex items-center justify-center text-white font-bold">
          SB
        </div>
        <h1 className="text-xl font-black text-brand-primary tracking-tight">
          Spin Bridge
        </h1>
      </div>
      <nav className="flex flex-col gap-3 font-bold text-sm">
        {/* Dashboard */}
        <Link
          to="/"
          className={`${navItemStyles} ${isActive("/") ? activeStyles : inactiveStyles}`}
        >
          <LayoutDashboard
            size={20}
            className={
              isActive("/")
                ? "text-white"
                : "text-slate-400 group-hover:text-brand-primary"
            }
          />
          <span>Dashboard</span>
        </Link>

        {/* Análisis de conversaciones - Texto corregido */}
        <Link
          to="/chats"
          className={`${navItemStyles} ${isActive("/chats") ? activeStyles : inactiveStyles}`}
        >
          <MessageSquare
            size={20}
            className={
              isActive("/chats")
                ? "text-white"
                : "text-slate-400 group-hover:text-brand-primary"
            }
          />
          <span className="leading-tight">
            Análisis de <br /> Conversaciones
          </span>
        </Link>

        {/* Configuración */}
        <Link
          to="/config"
          className={`${navItemStyles} ${isActive("/config") ? activeStyles : inactiveStyles}`}
        >
          <Settings
            size={20}
            className={
              isActive("/config")
                ? "text-white"
                : "text-slate-400 group-hover:text-brand-primary"
            }
          />
          <span>Configuración del Agente</span>
        </Link>
      </nav>
      {/* Footer del Sidebar (Opcional) */}
      <div className="mt-auto p-4 bg-slate-50 rounded-2xl border border-slate-100">
        <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest text-center">
          Developed by AI Products Chapter. 2026
        </p>
      </div>
    </div>
  );
}
