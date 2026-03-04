import { Sun, Moon } from "lucide-react";
import { useDarkMode } from "../hooks/useDarkMode";

export const ThemeToggle = () => {
  const [isDark, setIsDark] = useDarkMode();

  return (
    <button
      onClick={() => setIsDark(!isDark)}
      className="group relative flex items-center justify-center w-12 h-12 rounded-2xl bg-white dark:bg-slate-800 shadow-xl shadow-slate-200/50 dark:shadow-none border border-slate-100 dark:border-slate-700 transition-all duration-300 hover:scale-110 active:scale-95"
    >
      <div className="relative w-6 h-6 overflow-hidden">
        {/* Icono de Sol que sube/baja */}
        <Sun
          className={`absolute inset-0 text-brand-orange transition-transform duration-500 ${isDark ? "translate-y-10" : "translate-y-0"}`}
          size={24}
        />
        {/* Icono de Luna que sube/baja */}
        <Moon
          className={`absolute inset-0 text-brand-primary transition-transform duration-500 ${isDark ? "translate-y-0" : "-translate-y-10"}`}
          size={24}
        />
      </div>

      {/* Tooltip pequeño al hacer hover */}
      <span className="absolute right-14 scale-0 group-hover:scale-100 transition-all bg-slate-800 text-white text-[10px] py-1 px-2 rounded-lg font-bold">
        {isDark ? "Cambiar a modo claro" : "Cambiar a modo oscuro"}
      </span>
    </button>
  );
};
