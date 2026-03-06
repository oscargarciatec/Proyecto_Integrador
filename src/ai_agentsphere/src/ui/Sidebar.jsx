import React from "react";
import {
  MessageSquare,
  Settings,
  LayoutDashboard,
  CircleGauge,
} from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useThemeState } from "../context/ThemeContextCore";

const Sidebar = React.memo(() => {
  const location = useLocation();
  const activePath = location.pathname;
  // Sidebar responds to theme changes automatically via CSS classes.
  // We call useThemeState() to ensure the component re-renders when the state updates.
  useThemeState();

  const navItemStyles =
    "flex items-center gap-3 px-4 py-3 rounded-2xl transition-all duration-300 group font-bold text-sm";

  // Agregamos dark:hover para que el hover se vea bien en oscuro
  const inactiveStyles =
    "text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 hover:text-brand-primary dark:hover:text-brand-primary";

  const activeStyles =
    "bg-brand-primary text-white shadow-lg shadow-brand-primary/20 scale-[1.02]";

  return (
    <div className="w-68 h-screen bg-white dark:bg-slate-900 text-brand-dark dark:text-slate-100 p-6 flex flex-col fixed border-r border-slate-100 dark:border-slate-800 font-montserrat shadow-sm transition-colors duration-300">
      <div className="flex items-center justify-center mb-2 px-2">
        <Link to="/" className="block">
          {/* Logo para Modo Claro: Se oculta cuando hay clase .dark */}
          <img
            src="/logo-light.png"
            alt="AI AgentSphere Logo"
            className="h-30 w-auto dark:hidden object-contain"
          />
          {/* Logo para Modo Oscuro: Solo se muestra cuando hay clase .dark */}
          <img
            src="/logo-light.png"
            alt="AI AgentSphere Logo"
            className="h-30 w-auto hidden dark:block object-contain"
          />
        </Link>
      </div>

      <div className="text-center mb-12">
        <h1 className="text-2xl font-bold">
          AI Agent<span className="text-brand-primary">Sphere</span>
        </h1>
      </div>

      <nav className="flex flex-col gap-3">
        {/* Dashboard */}
        <Link
          to="/"
          className={`${navItemStyles} ${activePath === "/" ? activeStyles : inactiveStyles}`}
        >
          <CircleGauge
            size={20}
            className={
              activePath === "/"
                ? "text-white"
                : "text-slate-400 dark:text-slate-500 group-hover:text-brand-primary"
            }
          />
          <span>Metrics Center</span>
        </Link>

        {/* Análisis de conversaciones */}
        <Link
          to="/chats"
          className={`${navItemStyles} ${activePath === "/chats" ? activeStyles : inactiveStyles}`}
        >
          <MessageSquare
            size={20}
            className={
              activePath === "/chats"
                ? "text-white"
                : "text-slate-400 dark:text-slate-500 group-hover:text-brand-primary"
            }
          />
          <span className="leading-tight">
            Conversations <br /> Analysis
          </span>
        </Link>

        {/* Configuración */}
        <Link
          to="/config"
          className={`${navItemStyles} ${activePath === "/config" ? activeStyles : inactiveStyles}`}
        >
          <Settings
            size={20}
            className={
              activePath === "/config"
                ? "text-white"
                : "text-slate-400 dark:text-slate-500 group-hover:text-brand-primary"
            }
          />
          <span>Agent Configuration</span>
        </Link>
      </nav>

      <div className="mt-auto p-4 bg-slate-50 dark:bg-slate-800/40 rounded-2xl border border-slate-100 dark:border-slate-700 transition-colors">
        <p className="text-[10px] text-slate-400 dark:text-slate-500 font-bold uppercase tracking-widest text-center">
          Developed by AI Products <br /> Chapter. 2026
        </p>
      </div>
    </div>
  );
});

export default Sidebar;
