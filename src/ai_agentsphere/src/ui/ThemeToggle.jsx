import { Sun, Moon } from "lucide-react";
import { useThemeState, useThemeDispatch } from "../context/ThemeContextCore";

export const ThemeToggle = () => {
  const isDark = useThemeState();
  const { toggleTheme } = useThemeDispatch();

  return (
    <button
      onClick={toggleTheme}
      className="group relative flex items-center justify-center w-12 h-12 rounded-2xl bg-white dark:bg-slate-800 shadow-xl shadow-slate-200/50 dark:shadow-none border border-slate-100 dark:border-slate-700 transition-all duration-300 hover:scale-110 active:scale-95"
    >
      <div className="relative w-6 h-6 overflow-hidden">
        <Sun
          className={`absolute inset-0 text-brand-orange transition-transform duration-500 ${isDark ? "translate-y-10" : "translate-y-0"}`}
          size={24}
        />
        <Moon
          className={`absolute inset-0 text-brand-primary transition-transform duration-500 ${isDark ? "translate-y-0" : "-translate-y-10"}`}
          size={24}
        />
      </div>

      <span className="absolute right-14 scale-0 group-hover:scale-100 transition-all bg-slate-800 text-white text-[10px] py-1 px-2 rounded-lg font-bold">
        {isDark ? "Change to light mode" : "Change to dark mode"}
      </span>
    </button>
  );
};
